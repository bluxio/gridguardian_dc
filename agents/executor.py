import time
from tools.cooling import boost
from tools.battery import discharge
from tools.workload import redistribute


def _log(logs, tool, **kw):
    entry = {"t": time.time(), "tool": tool}
    entry.update(kw)
    logs.append(entry)


def apply_plan(state, plan, cooling_cap):
    logs = []
    for action in plan["actions"]:
        action_type = action["type"]
        if action_type == "cooling":
            msg, actual = boost(state, action["cluster"], action["kw"], cooling_cap)
            _log(
                logs,
                "cooling_tool",
                cluster=action["cluster"],
                kw=action["kw"],
                actual=actual,
                msg=msg,
            )
        elif action_type == "battery":
            msg, actual = discharge(state, action["cluster"], action["kw"])
            _log(
                logs,
                "battery_tool",
                cluster=action["cluster"],
                kw=action["kw"],
                actual=actual,
                msg=msg,
            )
        elif action_type == "redistribute":
            msg, actual = redistribute(state, action["from"], action["to"], action["kw"])
            _log(
                logs,
                "redistribute_tool",
                src=action["from"],
                dst=action["to"],
                kw=action["kw"],
                actual=actual,
                msg=msg,
            )
        else:
            _log(logs, "unknown_tool", raw=action, msg="Unknown action")
    return logs, state
