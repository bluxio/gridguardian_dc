import json
import os

import requests
from jsonschema import validate, ValidationError
from core.state import TEMP_LIMIT, ALPHA

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["type", "cluster", "kw"],
                        "properties": {
                            "type": {"const": "cooling"},
                            "cluster": {"type": "string"},
                            "kw": {"type": "number", "minimum": 0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["type", "cluster", "kw"],
                        "properties": {
                            "type": {"const": "battery"},
                            "cluster": {"type": "string"},
                            "kw": {"type": "number", "minimum": 0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["type", "from", "to", "kw"],
                        "properties": {
                            "type": {"const": "redistribute"},
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "kw": {"type": "number", "minimum": 0},
                        },
                    },
                ]
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}


def _nemotron_plan(
    power_defs,
    therm_viol,
    balance,
    battery_kw,
    cooling_cap,
    cooling_on,
    base_grid,
    power_draw,
):
    api_key = os.getenv("NEMOTRON_KEY")
    if not api_key:
        raise RuntimeError("Missing NEMOTRON_KEY")
    system = (
        "You are a datacenter planner. Choose actions from {cooling,battery,redistribute}.\n"
        "Obey capacities and nonnegativity. Prefer moving workload OFF hot/deficit clusters.\n"
        "Respond strictly as JSON with schema: {\"actions\": [{...}]}"
    )
    user = json.dumps(
        {
            "power_defs": power_defs,
            "thermal_viol": therm_viol,
            "balance": balance,
            "caps": {
                "battery_kw": battery_kw,
                "cooling_cap": cooling_cap,
                "cooling_on": cooling_on,
            },
            "grid": {"base_grid": base_grid},
            "power_draw": power_draw,
            "goal": "Eliminate thermal violations and raise balances >= tau with minimal battery/cooling.",
        }
    )
    payload = {
        "model": "nvidia/nvidia-nemotron-nano-9b-v2",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    plan = json.loads(content)
    validate(plan, PLAN_SCHEMA)
    reasoning = [
        "Nemotron planned actions with constraints enforced.",
        f"Thermal clusters: {list(therm_viol.keys())}",
        f"Power deficits: { {k: -v for k, v in power_defs.items()} }",
    ]
    return plan, reasoning


def greedy_plan(power_defs, therm_viol, balance, battery_kw, cooling_cap, cooling_on, base_grid, power_draw):
    plan = []
    # 1) Thermal corrections sized to hit TEMP_LIMIT
    for c in therm_viol:
        need_kw = max(0.0, (therm_viol[c] - TEMP_LIMIT) / ALPHA)
        head = max(0.0, cooling_cap[c] - cooling_on[c])
        if head > 0 and need_kw > 0:
            plan.append({"type": "cooling", "cluster": c, "kw": round(min(need_kw, head), 2)})

    # 2) Power: move workload away from deficits, then use battery/cooling
    for c, bal in power_defs.items():
        need = -bal  # kW needed to close the gap at c

        donors = {d: max(0.0, base_grid[d] - power_draw[d]) for d in balance.keys() if d != c}

        for d, s in list(donors.items()):
            if s <= 0:
                continue
            take = min(need, s, power_draw[c])
            if take > 0:
                plan.append({"type": "redistribute", "from": c, "to": d, "kw": round(take, 2)})
                power_draw[c] = max(0.0, power_draw[c] - take)
                power_draw[d] += take
                need -= take
            if need <= 0:
                break

        if need > 0:
            take = min(need, battery_kw[c])
            if take > 0:
                plan.append({"type": "battery", "cluster": c, "kw": round(take, 2)})
                need -= take

        if c in therm_viol and need > 0:
            head = max(0.0, cooling_cap[c] - cooling_on[c])
            if head > 0:
                plan.append({"type": "cooling", "cluster": c, "kw": round(min(need, head, 5.0), 2)})
    return {"actions": plan}


def validate_plan(p):
    validate(p, PLAN_SCHEMA)
    return True


def plan_actions(
    power_defs,
    therm_viol,
    balance,
    battery_kw,
    cooling_cap,
    cooling_on,
    base_grid,
    power_draw,
    use_llm=False,
    llm=None,
):
    reasoning = []
    if use_llm:
        try:
            return _nemotron_plan(
                power_defs,
                therm_viol,
                balance,
                battery_kw,
                cooling_cap,
                cooling_on,
                base_grid,
                power_draw,
            )
        except Exception as exc:
            reasoning.append(
                f"Nemotron error: {type(exc).__name__}: {exc}. Falling back to greedy planner."
            )
    plan = greedy_plan(
        power_defs,
        therm_viol,
        balance,
        battery_kw,
        cooling_cap,
        cooling_on,
        base_grid,
        power_draw,
    )
    try:
        validate_plan(plan)
    except ValidationError:
        reasoning.append("Greedy plan invalid; returning empty plan.")
        plan = {"actions": []}
    if not reasoning:
        reasoning = ["Greedy planner selected minimal valid actions."]
    return plan, reasoning
