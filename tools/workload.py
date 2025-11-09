from core.state import BETA, MAX_UTIL


def redistribute(state, src, dst, kw):
    # reduce src load and temp; increase dst if util allows
    if state["utilization"][dst] >= MAX_UTIL:
        return f"Redistribute skipped: {dst} at max util.", 0.0
    kw = max(0.0, min(kw, state["power_draw_kw"][src]))  # can't move more than draw
    state["power_draw_kw"][src] = max(0.0, state["power_draw_kw"][src] - kw)
    state["temp_c"][src] -= BETA * kw
    state["power_draw_kw"][dst] += kw
    state["temp_c"][dst] += BETA * kw
    # utilization nudge (mock)
    state["utilization"][src] = max(0.0, state["utilization"][src] - 0.01 * kw)
    state["utilization"][dst] = min(1.0, state["utilization"][dst] + 0.01 * kw)
    return f"Redistributed {kw:.1f} kW {src}→{dst}.", kw
