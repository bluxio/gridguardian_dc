import os
import json
import requests
import random


def _nemotron_gen_payload(meta, seed=None):
    clusters = meta["clusters"]
    rnd = random.Random(seed)
    base = {
        "power_draw_kw": {c: rnd.randint(10, 65) for c in clusters},
        "cooling_online_kw": {c: rnd.randint(5, 20) for c in clusters},
        "battery_kw": {c: rnd.randint(3, 12) for c in clusters},
        "utilization": {c: round(rnd.uniform(0.2, 0.9), 2) for c in clusters},
        "temp_c": {c: rnd.randint(55, 90) for c in clusters},
    }
    return base


def nemotron_generate_scenarios(meta, n=3, seed=None):
    """Calls Nemotron to propose scenarios; fallback to random sampler."""
    key = os.getenv("NEMOTRON_KEY")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    system = (
        "You generate realistic datacenter stress snapshots. "
        "Output strictly JSON with fields: power_draw_kw, cooling_online_kw, battery_kw, utilization, temp_c. "
        "Raise heat and load on 1-2 clusters per snapshot; others moderate."
    )
    scenarios = []
    if not key:
        for i in range(n):
            scenarios.append(_nemotron_gen_payload(meta, None if seed is None else seed + i))
        return scenarios, ["Nemotron key missing; returned locally sampled scenarios."]
    headers = {"Authorization": f"Bearer {key}"}
    notes = []
    for i in range(n):
        user = json.dumps({"clusters": meta["clusters"], "hint": "one hot GPU cluster, one donor"})
        payload = {
            "model": "nvidia/nvidia-nemotron-nano-9b-v2",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            scenarios.append(json.loads(content))
        except Exception as exc:
            notes.append(f"Nemotron error on scenario {i}: {type(exc).__name__}")
            scenarios.append(_nemotron_gen_payload(meta, None if seed is None else seed + i))
    if not notes:
        notes.append("Generated scenarios via Nemotron where possible; fell back to local sampler if errors.")
    return scenarios, notes
