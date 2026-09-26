"""Owned subprocess jobs. No training runs when this module is imported."""
import argparse
import json
import os
from pathlib import Path
import sys

AGENTS = ("ppo", "ppo_icm", "td3", "td3_rnd", "sac", "model_based")


def train_agent(agent, steps):
    if agent == "ppo":
        from ppo_simple import train_ppo
        train_ppo(total_steps=steps, resume_checkpoint=False)
    elif agent == "ppo_icm":
        from ppo_icm import train_ppo_icm
        train_ppo_icm(total_steps=steps, resume_checkpoint=False)
    elif agent == "td3":
        from td3FINAL import train_td3
        train_td3(total_steps=steps)
    elif agent == "td3_rnd":
        from td3_rndFINAL import train
        train(max_timesteps=steps, n_episodes=steps)
    elif agent == "sac":
        # SAC parses CLI arguments on import. Give it only its own arguments.
        sys.argv = ["sac_main", "--Max_train_steps", str(steps), "--save_interval", str(min(steps, 100000))]
        from sac_main import main
        main()
    elif agent == "model_based":
        from train import main
        main(max_updates=max(1, steps // 2048), resume_checkpoint=False)


def report(agents):
    import numpy as np
    import torch
    import run_hackability_report as engine
    from reward_wrapper import reward_metadata
    manifest = json.loads(Path("experiment.json").read_text(encoding="utf-8"))
    profile = manifest["reward"]["profile"]
    if manifest["reward"] != reward_metadata(profile):
        raise ValueError("Reward parameters changed after training; restore the recorded configuration.")
    os.environ["WALKER_REWARD_PROFILE"] = profile
    np.random.seed(0)
    torch.manual_seed(0)
    results = {}
    path = Path("runs/hackability_report.json")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        results = json.loads(path.read_text(encoding="utf-8"))
    for agent in agents:
        print(f"Evaluating {agent}: collecting trajectory and adversarial sweep", flush=True)
        result, _ = engine.run_agent(agent, engine.AGENTS[agent])
        results[agent] = result
        print(f"{agent}: score {result['hackability_score']:.4f}", flush=True)
    results["_experiment"] = manifest
    results["_limitations"] = [
        "Diagnostic signals are not proof of reward hacking.",
        "PPO normalization statistics are reconstructed during evaluation.",
        "Curiosity critics include intrinsic reward while diagnostics use extrinsic reward.",
        "Legacy TD3 reports may have used an untrained critic.",
        "Timeline indices refer to evaluation rollouts, not training timesteps.",
    ]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    temporary.replace(path)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "report"))
    parser.add_argument("--agents", nargs="+", choices=AGENTS, required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()
    if args.mode == "train":
        train_agent(args.agents[0], args.steps)
    else:
        report(args.agents)


if __name__ == "__main__":
    main()
