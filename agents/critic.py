import os
import json
import requests


def nemotron_grade(run_summary):
    """Ask Nemotron to critique controller outcomes; fallback deterministic."""
    key = os.getenv("NEMOTRON_KEY")
    stable_flag = run_summary.get("stable")
    if stable_flag is None and isinstance(run_summary.get("result"), dict):
        stable_flag = run_summary["result"].get("stable")
    if not key:
        return {
            "score": 1.0 if stable_flag else 0.0,
            "notes": "Fallback critic: pass if stable",
            "risks": [],
            "suggestions": [],
        }
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    system = (
        "You are a datacenter operations auditor. "
        "Given plan/actions and final metrics, return JSON with fields: score(0..1), notes, risks[], suggestions[]."
    )
    user = json.dumps(run_summary)
    payload = {
        "model": "nvidia/nvidia-nemotron-nano-9b-v2",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as exc:
        return {
            "score": 1.0 if stable_flag else 0.0,
            "notes": f"Nemotron error: {type(exc).__name__}",
            "risks": [],
            "suggestions": [],
        }
