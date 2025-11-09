def discharge(state, cluster, kw):
    kw = min(kw, state["battery_kw"][cluster])
    state["battery_kw"][cluster] -= kw
    state["battery_out_kw"][cluster] += kw
    return f"Discharged {kw:.1f} kW battery on {cluster}.", kw
