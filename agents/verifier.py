import json
from agents.monitor import power_balance, thermal_violations


def verify(state, tau=-2.0):
    bal = power_balance(state["base_grid_kw"], state["power_draw_kw"], state["battery_out_kw"])
    therm_bad = thermal_violations(state["temp_c"])
    power_bad = {c: v for c, v in bal.items() if v < tau}
    ok = len(therm_bad) == 0 and len(power_bad) == 0
    result = {
        "stable": ok,
        "tau": tau,
        "balance_after": bal,
        "thermal_violations": therm_bad,
        "power_deficits": power_bad,
    }
    if not ok:
        result["alert"] = {
            "level": "CRITICAL",
            "message": f"Issues detected in {list(therm_bad.keys()) + list(power_bad.keys())}",
        }
    else:
        result["alert"] = {"level": "OK", "message": "All clusters stable"}
    with open("alerts.log", "a") as f:
        f.write(json.dumps(result["alert"]) + "\n")
    return result
