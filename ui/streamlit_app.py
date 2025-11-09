import json
import os
import sys
import importlib

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.append(proj_root)

try:
    core = importlib.import_module("core_app")
except Exception as e:
    import streamlit as st  # type: ignore

    st.error(f"Failed to import core_app: {type(e).__name__}: {e}")
    raise

import streamlit as st, pandas as pd

assert (
    hasattr(core, "run_dc")
    and hasattr(core, "evaluate_dc")
    and hasattr(core, "evaluate_dc_nemotron")
), "core_app missing required functions"

st.set_page_config(page_title="GridGuardian DC", layout="wide")
st.title("GridGuardian DC — Datacenter Energy & Thermal Control")
st.caption("Nemotron integration available via OpenRouter. Set NEMOTRON_KEY to enable live calls; safe fallbacks used otherwise.")

scenario = st.selectbox("Scenario", ["A", "B"])
tau = st.slider("Power stability threshold (min kW balance)", -10.0, 0.0, -2.0, 0.5)
colA, colB = st.columns(2)
with colA:
    induce_failure = st.toggle("Induce failure (demo)", value=False)
with colB:
    use_llm = st.toggle("Use Nemotron planner (stub)", value=False)

if st.button("Run Controller"):
    res = core.run_dc(scenario, tau, induce_failure=induce_failure, use_llm=use_llm)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Power Balance (kW) — Before")
        st.bar_chart(pd.DataFrame.from_dict(res["balance_before"], orient="index", columns=["kW"]))
        st.subheader("Power Balance (kW) — After")
        st.bar_chart(pd.DataFrame.from_dict(res["balance_after"], orient="index", columns=["kW"]))
    with c2:
        st.subheader("Temperatures (°C)")
        df = pd.DataFrame({"Before": res["temp_before"], "After": res["temp_after"]})
        st.bar_chart(df)
        st.subheader("Plan (JSON)")
        st.json(res["plan"])
        st.subheader("Verification")
        st.json(res["verify"])
        if "alert" in res["verify"]:
            if res["verify"]["alert"]["level"] == "CRITICAL":
                st.error(f"⚠️ {res['verify']['alert']['message']}")
            else:
                st.success("✅ All clusters stable")
        st.download_button(
            "Download Plan JSON",
            data=json.dumps(res["plan"], indent=2),
            file_name=f"plan_{scenario}.json",
        )
        if st.button("Reset Scenario"):
            st.session_state.clear()
            st.rerun()

    st.subheader("ReAct Trace")
    for row in res["react_trace"]:
        st.write(f"{row['phase']}: {row['text']}")

st.divider()
st.subheader("Evaluator")
col1, col2 = st.columns(2)
with col1:
    eval_tau = st.slider("Eval τ (min kW balance)", -10.0, 0.0, -2.0, 0.5, key="eval_tau")
with col2:
    eval_use_llm = st.toggle("Use Nemotron planner (stub) in eval", value=False, key="eval_use_llm")


@st.cache_data(show_spinner=False)
def _cached_eval(tau, use_llm):
    return core.evaluate_dc(tau=tau, use_llm=use_llm)


if st.button("Run Evaluator"):
    summary = _cached_eval(eval_tau, eval_use_llm)
    st.write(f"Pass rate: {summary['passed']}/{summary['total']} = {summary['score']:.2f}")
    st.json(summary["runs"])

st.divider()
st.subheader("Nemotron Scenario Generator + Critic")
colg1, colg2, colg3 = st.columns(3)
with colg1:
    nemo_n = st.number_input("Num scenarios", min_value=1, max_value=10, value=3)
with colg2:
    nemo_tau = st.slider("Nemotron Eval τ", -10.0, 0.0, -2.0, 0.5, key="nemo_tau")
with colg3:
    nemo_use_llm = st.toggle("Use Nemotron planner in eval", value=False, key="nemo_use_llm")

if st.button("Run Nemotron Eval"):
    summary = core.evaluate_dc_nemotron(tau=nemo_tau, use_llm=nemo_use_llm, n_scenarios=int(nemo_n))
    st.write(f"Pass rate: {summary['passed']}/{summary['total']} = {summary['score']:.2f}")
    for idx, run in enumerate(summary["runs"]):
        score = run["grade"].get("score", 0.0)
        st.write(f"Scenario #{idx}: score={score:.2f}")
        st.json(run)
