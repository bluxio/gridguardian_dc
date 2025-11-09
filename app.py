import json
from functools import lru_cache
import copy
from agents.monitor import power_balance, thermal_violations, power_deficits
from agents.planner import plan_actions
from agents.executor import apply_plan
from agents.verifier import verify
from agents.narrator import narrate_react
from agents.scenario_gen import nemotron_generate_scenarios
from agents.critic import nemotron_grade

CLUSTERS_DEFAULT = ["GPU_A", "CPU_B", "STORAGE_C", "EDGE_D"]


def _load_clusters_meta():
    import json as _json
    import os

    path = os.path.join(os.path.dirname(__file__), "data", "clusters.json")
    with open(path, "r") as fh:
        return _json.load(fh)


def _cluster_keys(meta=None):
    if isinstance(meta, dict) and "clusters" in meta:
        return list(meta["clusters"])
    return CLUSTERS_DEFAULT


def _ensure_map(val, keys, default=0.0):
    if isinstance(val, dict):
        return {k: float(val.get(k, default)) for k in keys}
    try:
        v = float(val)
        return {k: (v if i == 0 else default) for i, k in enumerate(keys)}
    except Exception:
        return {k: default for k in keys}


@lru_cache(maxsize=8)
def _cached_load(meta_path: str, scenario_path: str):
    import json as _json

    with open(meta_path) as f:
        meta = _json.load(f)
    with open(scenario_path) as f:
        state = _json.load(f)
    return meta, state


def load_state(meta_path, scenario_path):
    meta, state = _cached_load(meta_path, scenario_path)
    meta = copy.deepcopy(meta)
    state = copy.deepcopy(state)
    state["base_grid_kw"] = meta["base_grid_kw"]
    state["cooling_capacity_kw"] = meta["cooling_capacity_kw"]
    state["battery_max_kw"] = meta["battery_max_kw"]
    return state, meta.get("clusters") or meta.get("regions")


def run_dc(scenario="A", tau=-2.0, induce_failure=False, use_llm=False):
    state, clusters = load_state("data/clusters.json", f"data/scenario_DC_{scenario}.json")
    base_keys = clusters or list(state.get("power_draw_kw", {}).keys())
    meta_for_keys = {"clusters": base_keys} if base_keys else None
    keys = _cluster_keys(meta_for_keys)
    state["battery_out_kw"] = {k: 0.0 for k in keys}
    if induce_failure:
        state["power_draw_kw"]["GPU_A"] += 8.0
        state["temp_c"]["GPU_A"] += 4.0
    bal0 = power_balance(state["base_grid_kw"], state["power_draw_kw"], state["battery_out_kw"])
    therm0 = thermal_violations(state["temp_c"])
    powdef0 = power_deficits(bal0)
    plan, reasoning = plan_actions(
        powdef0,
        therm0,
        bal0,
        state["battery_kw"],
        state["cooling_capacity_kw"],
        state["cooling_online_kw"],
        state["base_grid_kw"],
        state["power_draw_kw"].copy(),
        use_llm=use_llm,
    )
    logs, state2 = apply_plan(state, plan, state["cooling_capacity_kw"])
    ver = verify(state2, tau)
    trace = narrate_react(reasoning, logs, ver)
    return {
        "balance_before": bal0,
        "temp_before": state["temp_c"],
        "plan": plan,
        "logs": logs,
        "verify": ver,
        "balance_after": ver["balance_after"],
        "temp_after": state2["temp_c"],
        "react_trace": trace,
    }


def _summarize_for_critic(res, scenario, induce_failure, tau, planner):
    return {
        "scenario": scenario,
        "induce_failure": induce_failure,
        "tau": tau,
        "stable": res["verify"]["stable"],
        "balance_after": res["verify"]["balance_after"],
        "thermal_violations": res["verify"]["thermal_violations"],
        "power_deficits": res["verify"]["power_deficits"],
        "plan": res["plan"],
        "logs": res["logs"],
        "planner": planner,
    }


def evaluate_dc(tau=-2.0, use_llm=False):
    """
    Run multiple scenarios with and without induced failure.
    Returns a summary dict with pass rate and per-run results.
    """
    runs = [
        ("A", False),
        ("A", True),
        ("B", False),
        ("B", True),
    ]
    results = []
    passed = 0
    for sc, fail in runs:
        res = run_dc(scenario=sc, tau=tau, induce_failure=fail, use_llm=use_llm)
        ok = res["verify"]["stable"]
        if ok:
            passed += 1
        results.append(
            {
                "scenario": sc,
                "induce_failure": fail,
                "stable": ok,
                "balance_after": res["verify"]["balance_after"],
                "thermal_violations": res["verify"]["thermal_violations"],
                "power_deficits": res["verify"]["power_deficits"],
            }
        )
    return {"passed": passed, "total": len(runs), "score": passed / len(runs), "runs": results}


def evaluate_dc_nemotron(tau=-2.0, use_llm=False, n_scenarios=3):
    meta_all = _load_clusters_meta()
    keys = _cluster_keys(meta_all)
    scenarios, notes = nemotron_generate_scenarios(meta_all, n=n_scenarios)
    runs = []
    passed = 0
    for idx, snap in enumerate(scenarios):
        state = {
            "timestep": 0,
            "power_draw_kw": snap["power_draw_kw"],
            "cooling_online_kw": snap["cooling_online_kw"],
            "battery_kw": snap["battery_kw"],
            "utilization": snap["utilization"],
            "temp_c": snap["temp_c"],
            "cooling_capacity_kw": meta_all["cooling_capacity_kw"],
            "battery_max_kw": meta_all["battery_max_kw"],
            "base_grid_kw": meta_all["base_grid_kw"],
        }
        state["base_grid_kw"] = _ensure_map(state.get("base_grid_kw"), keys, 0.0)
        state["power_draw_kw"] = _ensure_map(state.get("power_draw_kw"), keys, 0.0)
        state["cooling_online_kw"] = _ensure_map(state.get("cooling_online_kw"), keys, 0.0)
        state["battery_kw"] = _ensure_map(state.get("battery_kw"), keys, 0.0)
        state["utilization"] = _ensure_map(state.get("utilization"), keys, 0.0)
        state["temp_c"] = _ensure_map(state.get("temp_c"), keys, 0.0)
        state["battery_out_kw"] = _ensure_map(state.get("battery_out_kw"), keys, 0.0)
        state["cooling_capacity_kw"] = _ensure_map(state.get("cooling_capacity_kw"), keys, 0.0)
        bal0 = power_balance(state["base_grid_kw"], state["power_draw_kw"], state["battery_out_kw"])
        therm0 = thermal_violations(state["temp_c"])
        powdef0 = power_deficits(bal0)
        plan, reasoning = plan_actions(
            powdef0,
            therm0,
            bal0,
            state["battery_kw"],
            state["cooling_capacity_kw"],
            state["cooling_online_kw"],
            state["base_grid_kw"],
            state["power_draw_kw"].copy(),
            use_llm=use_llm,
        )
        logs, state2 = apply_plan(state, plan, state["cooling_capacity_kw"])
        ver = verify(state2, tau)
        res = {"plan": plan, "logs": logs, "verify": ver}
        if ver["stable"]:
            passed += 1
        run_summary = _summarize_for_critic(
            res,
            scenario=f"NEMO_{idx}",
            induce_failure=False,
            tau=tau,
            planner="nemotron" if use_llm else "greedy",
        )
        grade = nemotron_grade({"result": run_summary})
        runs.append({"result": run_summary, "grade": grade})
    return {"notes": notes, "passed": passed, "total": len(runs), "score": passed / len(runs), "runs": runs}
