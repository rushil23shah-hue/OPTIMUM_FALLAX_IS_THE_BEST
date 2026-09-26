"""NASim experiment contracts. Kept dependency-free for dashboard discovery."""
ENVIRONMENT = "NASim-Pentesting-v0"
AGENT = "graph_ppo"
SOURCE_SHA256 = "4ba93663cde54ab87a7719517b56bbb19877358f9c256e884fa102f989a966e6"
ADAPTER_VERSION = "nasim_graph_ppo_v1"
PROFILES = {
    "nasim_current": {
        "label": "Current notebook shaping",
        "scan_cost": 0.0, "shaping": True,
        "description": "The uploaded notebook's current reward, with free scans and explicit shaping. Not assumed clean.",
        "historical_verified": False,
    },
    "nasim_scan_penalty": {
        "label": "Native reward / scan penalty 1 (reconstruction)",
        "scan_cost": 1.0, "shaping": False,
        "description": "Native NASim reward with a cost of +1 (reward penalty -1) for every scan. A reconstruction, not a verified copy of the earlier notebook.",
        "historical_verified": False,
    },
}
SPLITS = {"train": ([12, 12, 12], 10000), "validation": ([3, 4, 5], 50000), "test": ([5, 5, 10], 90000)}


def reward_metadata(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown NASim reward profile: {profile}")
    p = PROFILES[profile]
    return {"profile": profile, "parameters": {
        "scan_cost": p["scan_cost"], "notebook_shaping": p["shaping"],
        "ids_enabled": False, "explicit_honeypot_penalty_enabled": False,
        "source_sha256": SOURCE_SHA256, "adapter_version": ADAPTER_VERSION,
    }, "historical_verified": p["historical_verified"]}


def make_manifest(profile="nasim_current", steps=102400, seed=2026):
    return {"environment": ENVIRONMENT, "agents": [AGENT],
            "reward": reward_metadata(profile), "initialization": "fresh",
            "requested_steps": steps, "budgets": {AGENT: steps}, "seed": seed,
            "rollout_steps": 1024, "ppo_epochs": 4, "minibatch_size": 256,
            "evaluation": {"split": "test", "repeats": 3, "deterministic": True},
            "scenario_bank": {name: {"counts": counts, "seed_offset": offset}
                              for name, (counts, offset) in SPLITS.items()},
            "adapter_version": ADAPTER_VERSION, "source_sha256": SOURCE_SHA256}
