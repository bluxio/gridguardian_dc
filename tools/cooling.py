from core.state import ALPHA


def boost(state, cluster, kw, cooling_cap):
    head = cooling_cap[cluster] - state["cooling_online_kw"][cluster]
    kw = max(0.0, min(kw, head))
    state["cooling_online_kw"][cluster] += kw
    state["temp_c"][cluster] -= ALPHA * kw
    return f"Boosted cooling {kw:.1f} kW on {cluster}.", kw
