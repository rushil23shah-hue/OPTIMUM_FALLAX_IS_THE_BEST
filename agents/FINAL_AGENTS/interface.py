"""Train selected agents, evaluate held-out rollouts, and compare policies."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

# Existing trainers use sibling imports; support script and -m execution.
AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
from metrics import cross_agent_policy_distance
from metrics_integration import evaluate_agent, write_json

AGENTS = {
    "ppo": ("ppo_simple", "train_ppo"),
    "ppo_icm": ("ppo_icm", "train_ppo_icm"),
    "td3": ("td3FINAL", "train_td3"),
    "td3_rnd": ("td3_rndFINAL", "train"),
    "sac": ("main", "main"),
    "model_based": ("train", "main"),
}


def training_kwargs(name, env_id, seed, steps=None):
    options = {"env_id": env_id, "seed": seed}
    if name == "td3":
        options["output_dir"] = "."
    if name == "sac":
        options["argv"] = []
    if steps is not None:
        if name == "td3_rnd":
            options["max_timesteps"] = steps
        elif name == "model_based":
            options.update(n_steps=min(2048, steps), max_updates=(steps + 2047) // 2048)
        else:
            options["total_steps"] = steps
            if name in ("ppo", "ppo_icm"):
                options["n_steps"] = min(2048, steps)
                options["n_epochs"] = 1
                options["batch_size"] = min(256, steps)
    return options


def run_agents(names=None, env_id="BipedalWalker-v3", steps=None, episodes=5,
               seed=0, run_dir="runs", eval_max_steps=None, shared_state_count=256):
    names = list(AGENTS) if names is None else list(names)
    if not names or len(names) != len(set(names)) or any(n not in AGENTS for n in names):
        raise ValueError("select distinct registered agents")
    if steps is not None and steps < 2:
        raise ValueError("steps must be at least 2")
    if episodes < 1 or shared_state_count < 1:
        raise ValueError("episodes and shared_state_count must be positive")
    if eval_max_steps is not None and eval_max_steps < 1:
        raise ValueError("eval_max_steps must be positive")
    from datetime import datetime
    root = Path(run_dir).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    results, adapters, state_pools = {}, {}, []
    for name in names:
        output = root / name
        output.mkdir()
        previous_cwd = Path.cwd()
        try:
            os.chdir(output)
            module_name, callable_name = AGENTS[name]
            model = getattr(importlib.import_module(module_name), callable_name)(
                **training_kwargs(name, env_id, seed, steps))
            context = getattr(model, "metrics_context", {})
            context.update(training_steps=steps, training_seed=seed,
                           policy_updates=max(1, (steps or 1) // 2048),
                           reward_id="environment_raw_extrinsic", run_id=str(root))
            model.metrics_context = context
            adapter, trajectories, _ = evaluate_agent(name, model, output, episodes, seed + 10000, eval_max_steps)
            adapters[name] = adapter
            pool = np.concatenate([t.obs[:-1] for t in trajectories])
            indices = np.linspace(0, len(pool) - 1, min(len(pool), shared_state_count), dtype=int)
            state_pools.append(pool[indices])
            results[name] = {"status": "ok", "metrics": str(output / "metrics.json")}
        except Exception as exc:
            traceback.print_exc()
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.chdir(previous_cwd)
        write_json(root / "summary.json", results)
    if len(adapters) >= 2:
        pool = np.concatenate(state_pools)
        rng = np.random.default_rng(seed)
        shared = pool[rng.choice(len(pool), min(shared_state_count, len(pool)), replace=False)]
        comparison = cross_agent_policy_distance(shared, {name: a.policy for name, a in adapters.items()})
        comparison.update(env_id=env_id, policy_mode="deterministic executed actions",
                          interpretation="Low distance indicates similar actions, not proof of a shared exploit.")
        np.save(root / "shared_observations.npy", shared)
        write_json(root / "cross_agent_metrics.json", comparison)
    print(json.dumps({"output_dir": str(root), "agents": results}, indent=2))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=list(AGENTS))
    parser.add_argument("--env-id", default="BipedalWalker-v3")
    parser.add_argument("--steps", type=int, help="training budget per agent; defaults to each trainer's budget")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--eval-max-steps", type=int)
    args = vars(parser.parse_args(argv))
    args["names"] = args.pop("agents")
    results = run_agents(**args)
    return int(any(item["status"] != "ok" for item in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
