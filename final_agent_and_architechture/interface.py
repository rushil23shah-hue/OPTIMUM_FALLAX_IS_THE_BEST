"""Train all six agents in a fresh, isolated reward experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

AGENTS = {
    "ppo": ("ppo_simple", "train_ppo"),
    "ppo_icm": ("ppo_icm", "train_ppo_icm"),
    "td3": ("td3FINAL", "train_td3"),
    "td3_rnd": ("td3_rndFINAL", "train"),
    "sac": ("sac_main", "main"),
    "model_based": ("train", "main"),
}
BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-profile", choices=("original", "optimized_v1"), default="optimized_v1")
    parser.add_argument("--experiment", type=Path, help="New output directory; defaults to experiments/<profile>")
    args = parser.parse_args()
    from reward_wrapper import reward_metadata
    output = (args.experiment or BASE / "experiments" / args.reward_profile).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Experiment directory is not empty: {output}. Choose a new --experiment directory for a fresh run.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"environment": "BipedalWalker-v3", "reward": reward_metadata(args.reward_profile),
                "agents": AGENTS, "initialization": "fresh", "python": sys.executable}
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(WALKER_REWARD_PROFILE=args.reward_profile, MPLBACKEND="Agg",
               PYTHONPATH=str(BASE) + os.pathsep + env.get("PYTHONPATH", ""))
    results = {}
    for name, (module, function) in AGENTS.items():
        print(f"=== {name}: {args.reward_profile} -> {output} ===", flush=True)
        result = subprocess.run([sys.executable, "-u", "-c",
                                 f"from {module} import {function}; {function}()"], cwd=output, env=env)
        results[name] = "ok" if result.returncode == 0 else f"failed: exit {result.returncode}"
        (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return int(any(value != "ok" for value in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
