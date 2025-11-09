def narrate_react(reasoning, tool_logs, verify):
    trace = []
    for r in reasoning:
        trace.append({"phase": "Reason", "text": r})
    for entry in tool_logs:
        args = {k: v for k, v in entry.items() if k not in ["t", "tool", "msg"]}
        trace.append({"phase": "Act", "text": f"{entry['tool']} args={args}"})
        trace.append({"phase": "Observe", "text": entry.get("msg", "done")})
    trace.append(
        {
            "phase": "Observe",
            "text": "PASS"
            if verify["stable"]
            else f"FAIL {verify['power_deficits']} {verify['thermal_violations']}",
        }
    )
    return trace
