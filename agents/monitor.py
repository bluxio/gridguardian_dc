from core.state import TEMP_LIMIT, POWER_MARGIN


def power_balance(base_grid, power_draw, battery_out):
    return {c: (base_grid[c] + battery_out.get(c, 0.0) - power_draw[c]) for c in power_draw}


def thermal_violations(temp):
    if isinstance(temp, dict):
        return {c: t for c, t in temp.items() if t > TEMP_LIMIT}
    try:
        value = float(temp)
    except Exception:
        return {}
    return {"GPU_A": value} if value > TEMP_LIMIT else {}


def power_deficits(balance):
    return {c: v for c, v in balance.items() if v < -POWER_MARGIN}


def donor_headroom(balance):
    return {c: max(0.0, v) for c, v in balance.items()}
