import copy
import json
import os
from functools import lru_cache
from pathlib import Path

from agents.monitor import power_balance, thermal_violations, power_deficits
from agents.planner import plan_actions
from agents.executor import apply_plan
from agents.verifier import verify
from agents.narrator import narrate_react
from agents.scenario_gen import nemotron_generate_scenarios
from agents.critic import nemotron_grade

CLUSTERS_DEFAULT = ["GPU_A", "CPU_B", "STORAGE_C", "EDGE_D"]
DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_clusters_meta():
    with open(DATA_DIR / "clusters.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def _cluster_keys(meta=None):
    if isinstance(meta, dict) and meta.get("clusters"):
        return list(meta["clusters"])
    return CLUSTERS_DEFAULT[:]


def _ensure_map(val, keys, default=0.0):
    if isinstance(val, dict):
        return {k: float(val.get(k, default)) for k in keys}
    try:
        scalar = float(val)
        return {k: (scalar if idx == 0 else default) for idx, k in enumerate(keys)}
    except Exception:
        return {k: default for k in keys}


@lru_cache(maxsize=8)
def _cached_load(meta_path: str, scenario_path: str):
    with open(meta_path, "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    with open(scenario_path, "r", encoding="utf-8") as sf:
        state = json.load(sf)
    return meta, state


def _load_state(scenario_id: str):
    meta_path = DATA_DIR / "clusters.json"
    scenario_path = DATA_DIR / f"scenario_DC_{scenario_id}.json"
    meta, state = _cached_load(str(meta_path), str(scenario_path))
    return copy.deepcopy(meta), copy.deepcopy(state)


def _normalize_state(meta, state):
    keys = _cluster_keys(meta)
    state["base_grid_kw"] = _ensure_map(meta.get("base_grid_kw") or state.get("base_grid_kw"), keys, 0.0)
    state["cooling_capacity_kw"] = _ensure_map(meta.get("cooling_capacity_kw") or state.get("cooling_capacity_kw"), keys, 0.0)
    state["battery_max_kw"] = _ensure_map(meta.get("battery_max_kw") or state.get("battery_max_kw"), keys, 0.0)
    state["power_draw_kw"] = _ensure_map(state.get("power_draw_kw"), keys, 0.0)
    state["cooling_online_kw"] = _ensure_map(state.get("cooling_online_kw"), keys, 0.0)
    state["battery_kw"] = _ensure_map(state.get("battery_kw"), keys, 0.0)
    state["utilization"] = _ensure_map(state.get("utilization"), keys, 0.0)
    state["temp_c"] = _ensure_map(state.get("temp_c"), keys, 0.0)
    state["battery_out_kw"] = {k: 0.0 for k in keys}
    state["clusters"] = keys
    return state


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


def run_dc(scenario_id="A", tau=-2.0, induce_failure=False, use_llm=False):
    meta, state = _load_state(scenario_id)
    state = _normalize_state(meta, state)
    keys = state["clusters"]
    if induce_failure and "GPU_A" in keys:
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
    logs, new_state = apply_plan(state, plan, state["cooling_capacity_kw"])
    verification = verify(new_state, tau)
    trace = narrate_react(reasoning, logs, verification)
    return {
        "scenario": scenario_id,
        "balance_before": bal0,
        "balance_after": verification["balance_after"],
        "temp_before": state["temp_c"].copy(),
        "temp_after": new_state["temp_c"].copy(),
        "plan": plan,
        "logs": logs,
        "verify": verification,
        "react_trace": trace,
    }


def evaluate_dc(tau=-2.0, use_llm=False):
    configs = [("A", False), ("A", True), ("B", False), ("B", True)]
    runs = []
    passed = 0
    for scenario_id, induce in configs:
        res = run_dc(scenario_id, tau=tau, induce_failure=induce, use_llm=use_llm)
        ok = res["verify"]["stable"]
        if ok:
            passed += 1
        runs.append(
            {
                "scenario": scenario_id,
                "induce_failure": induce,
                "stable": ok,
                "balance_after": res["verify"]["balance_after"],
                "thermal_violations": res["verify"]["thermal_violations"],
                "power_deficits": res["verify"]["power_deficits"],
            }
        )
    total = len(runs)
    return {"passed": passed, "total": total, "score": passed / total if total else 0.0, "runs": runs}


def evaluate_dc_nemotron(tau=-2.0, use_llm=False, n_scenarios=3):
    meta_all = _load_clusters_meta()
    keys = _cluster_keys(meta_all)
    scenarios, notes = nemotron_generate_scenarios(meta_all, n=n_scenarios)
    runs = []
    passed = 0
    for idx, snap in enumerate(scenarios):
        state = {
            "timestep": snap.get("timestep", 0),
            "power_draw_kw": snap.get("power_draw_kw", {}),
            "cooling_online_kw": snap.get("cooling_online_kw", {}),
            "battery_kw": snap.get("battery_kw", {}),
            "utilization": snap.get("utilization", {}),
            "temp_c": snap.get("temp_c", {}),
            "cooling_capacity_kw": meta_all.get("cooling_capacity_kw", {}),
            "battery_max_kw": meta_all.get("battery_max_kw", {}),
            "base_grid_kw": meta_all.get("base_grid_kw", {}),
        }
        state["base_grid_kw"] = _ensure_map(state["base_grid_kw"], keys, 0.0)
        state["power_draw_kw"] = _ensure_map(state["power_draw_kw"], keys, 0.0)
        state["cooling_online_kw"] = _ensure_map(state["cooling_online_kw"], keys, 0.0)
        state["battery_kw"] = _ensure_map(state["battery_kw"], keys, 0.0)
        state["utilization"] = _ensure_map(state["utilization"], keys, 0.0)
        state["temp_c"] = _ensure_map(state["temp_c"], keys, 0.0)
        state["battery_out_kw"] = _ensure_map(state.get("battery_out_kw"), keys, 0.0)
        state["cooling_capacity_kw"] = _ensure_map(state["cooling_capacity_kw"], keys, 0.0)
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
        logs, new_state = apply_plan(state, plan, state["cooling_capacity_kw"])
        verification = verify(new_state, tau)
        if verification["stable"]:
            passed += 1
        result = {
            "plan": plan,
            "logs": logs,
            "verify": verification,
        }
        summary = _summarize_for_critic(
            result,
            scenario=f"NEMO_{idx}",
            induce_failure=False,
            tau=tau,
            planner="nemotron" if use_llm else "greedy",
        )
        grade = nemotron_grade({"result": summary})
        runs.append({"result": summary, "grade": grade})
    total = len(runs)
    return {
        "notes": notes,
        "passed": passed,
        "total": total,
        "score": passed / total if total else 0.0,
        "runs": runs,
    }
