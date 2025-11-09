# GridGuardian DC

Lightweight datacenter controller prototype showcasing monitor, planner, executor, and verifier agents that balance power, cooling, and workload between clusters.

## Structure
- `app.py` wires agents, loads scenarios, and exposes `run_dc` for the UI.
- `agents/` contains monitor/planner/executor/verifier logic.
- `tools/` hosts actuation helpers for cooling, battery, and workload redistribution.
- `data/` defines cluster metadata plus scenario snapshots A/B.
- `ui/streamlit_app.py` offers a Streamlit dashboard to run the controller.

## Usage
```
cd gridguardian_dc
pip install jsonschema streamlit pandas
streamlit run ui/streamlit_app.py
```

## Nemotron Integration
- **Planner:** Live JSON plans via `nvidia/Nemotron-nano-9b-v2` on OpenRouter; falls back to the greedy heuristic if unavailable or invalid.
- **Scenario Generator:** Nemotron proposes stress snapshots; fallback sampler produces realistic loads locally.
- **Critic:** Nemotron grades outcomes with `score/notes/risks/suggestions`; fallback issues 1.0 for stable runs, else 0.0.
- **Keys:**
  - macOS/Linux: `export NEMOTRON_KEY=sk-or-xxxxxxxx`
  - Windows PowerShell: `setx NEMOTRON_KEY "sk-or-xxxxxxxx"`
- **Security:** Never log the secret key. Responses are schema-validated and always have safe fallback paths.

## Nemotron (Live)
- Requires an OpenRouter API key with access to Nemotron models.
- Run the UI toggle “Use Nemotron planner” (and evaluator toggles) to enable live calls.
- If the call fails or schema validation rejects the output, the system automatically reverts to the greedy planner with a log entry.
