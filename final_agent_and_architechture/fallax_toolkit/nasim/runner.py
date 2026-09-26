"""Explicit NASim commands. Default preflight does not train or update weights."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile
import time
import numpy as np
import torch

from . import notebook_model as model
from .adapter import Episode, create_bank, create_models, bank_fingerprint, validate_bank
from .config import AGENT, ENVIRONMENT, PROFILES, ADAPTER_VERSION, make_manifest, reward_metadata


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(serializable(data), indent=2), encoding="utf-8")
    temp.replace(path)


def runtime_versions():
    return {n: importlib.metadata.version(n) for n in ("nasim", "torch", "torch-geometric", "numpy", "gymnasium")}


def read_manifest(output):
    manifest = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("environment") != ENVIRONMENT or manifest.get("agents") != [AGENT]:
        raise ValueError("This is not a NASim / Graph PPO experiment.")
    if manifest["reward"] != reward_metadata(manifest["reward"]["profile"]):
        raise ValueError("Reward configuration does not match the training manifest.")
    if manifest.get("adapter_version") != ADAPTER_VERSION:
        raise ValueError("Unsupported NASim adapter version.")
    return manifest


def collect(models, specs, profile, seed=2026, repeats=1, step_cap=None, deterministic=True):
    """Inference-only collection, including frozen pre-action graph embeddings."""
    for network in models:
        network.eval()
    embeddings, rows, episodes = [], [], []
    episode_number = 0
    for spec in specs:
        for repeat in range(repeats):
            episode_number += 1
            # Identical scenario/transition seeds for paired reward comparisons.
            episode_seed = seed + int(spec["seed"]) * 10 + repeat
            torch.manual_seed(episode_seed)
            episode = Episode(spec, profile, episode_seed)
            total, native_total, honeypot = 0.0, 0.0, False
            try:
                limit = min(400, step_cap) if step_cap else 400
                with torch.no_grad():
                    for _ in range(limit):
                        decision = episode.choose(*models, deterministic=deterministic)
                        reward, terminated, truncated, row = episode.step(decision)
                        embeddings.append(decision["embedding"].numpy().copy())
                        row.update(episode=episode_number, repeat=repeat, scenario_seed=spec["seed"], transition_seed=episode_seed)
                        rows.append(row)
                        total += reward
                        native_total += row["native_reward"]
                        honeypot |= row["honeypot_targeted"]
                        if terminated or truncated:
                            break
                # Test-only short collection caps are explicit administrative truncations.
                if rows and not (rows[-1]["terminated"] or rows[-1]["truncated"]):
                    rows[-1].update(truncated=True, boundary_reason="diagnostic_step_cap")
                episodes.append({"scenario": spec["name"], "difficulty": spec["difficulty"],
                     "repeat": repeat, "seed": episode_seed, "reward": total, "native_reward": native_total,
                     "goal_success": bool(episode.env.goal_reached()), "honeypot_targeted": honeypot,
                     "length": episode.steps, "boundary_reason": rows[-1]["boundary_reason"]})
            finally:
                episode.close()
    return embeddings, rows, episodes


def save_checkpoint(output, models, optimizer, manifest, steps, fingerprint, complete=False):
    payload = {"format": ADAPTER_VERSION, "environment": ENVIRONMENT, "agent": AGENT,
               "reward": manifest["reward"], "steps": steps, "training_complete": complete,
               "scenario_fingerprint": fingerprint, "seed": manifest["seed"],
               "state_encoder": models[0].state_dict(), "actor": models[1].state_dict(),
               "critic": models[2].state_dict(), "optimizer": optimizer.state_dict(),
               "runtime_versions": runtime_versions()}
    path = output / "graph_ppo.pt"
    temp = output / "graph_ppo.tmp"
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(output, manifest, fingerprint):
    path = output / "graph_ppo.pt"
    if not path.exists():
        raise FileNotFoundError("No Graph PPO checkpoint. Train this experiment first.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != ADAPTER_VERSION or checkpoint.get("reward") != manifest["reward"]:
        raise ValueError("Checkpoint adapter/reward does not match this experiment.")
    if checkpoint.get("scenario_fingerprint") != fingerprint:
        raise ValueError("Scenario bank differs from the trained checkpoint.")
    if checkpoint.get("steps", 0) <= 0 or not checkpoint.get("training_complete"):
        raise ValueError("A completed trained checkpoint is required for a report.")
    models = create_models(manifest["seed"])
    for net, key in zip(models, ("state_encoder", "actor", "critic")):
        net.load_state_dict(checkpoint[key])
        net.eval()
    return models, checkpoint


def train(output):
    """Called only by the user's explicit Train action; never by preflight."""
    manifest = read_manifest(output)
    if (output / "graph_ppo.pt").exists():
        raise ValueError("Checkpoint already exists. Start a new experiment; implicit resume is disabled.")
    profile = manifest["reward"]["profile"]
    banks = create_bank(output / "scenario_bank", profile)
    validate_bank(banks)
    fingerprint = bank_fingerprint(banks)
    manifest.update(scenario_fingerprint=fingerprint, runtime_versions=runtime_versions())
    save_json(output / "experiment.json", manifest)
    seed = manifest["seed"]
    models = create_models(seed)
    encoder, actor, critic = models
    parameters = [p for net in models for p in net.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    rng = np.random.default_rng(seed + 1)
    rollout_steps = manifest["rollout_steps"]
    iterations = max(1, manifest["budgets"][AGENT] // rollout_steps)
    buffer = model.RolloutBuffer()
    history, total_steps, episode_count = [], 0, 0
    start = time.monotonic()
    for iteration in range(iterations):
        for net in models:
            net.train()
        buffer.clear()
        progress = (iteration + 1) / iterations
        difficulty = 0 if progress < .2 else 1 if progress < .5 else 2
        eligible = [s for s in banks["train"] if s["difficulty"] <= difficulty]

        def start_episode():
            nonlocal episode_count
            episode_count += 1
            spec = eligible[int(rng.integers(len(eligible)))]
            return Episode(spec, profile, seed + episode_count)

        episode = start_episode()
        try:
            for _ in range(rollout_steps):
                with torch.no_grad():
                    decision = episode.choose(*models)
                observation = episode.obs.copy()
                reward, terminated, truncated, row = episode.step(decision)
                done = terminated or truncated
                buffer.store(observation, decision["chosen"].detach(), decision["log_prob"].detach(),
                             decision["value"].detach(), reward, done, decision["pyg"], decision["mask"])
                total_steps += 1
                if done:
                    episode.close()
                    episode = start_episode()
            with torch.no_grad():
                if buffer.dones[-1]:
                    last_value = torch.tensor(0.0)
                else:
                    graph, pyg, _ = episode.graph()
                    nodes = encoder.gnn(pyg.x, pyg.edge_index, pyg.edge_attr)
                    last_value = critic(torch.cat([nodes.mean(0), nodes.max(0).values])).squeeze()
        finally:
            episode.close()
        returns, advantages = model.compute_returns_and_advantages(buffer, last_value, gamma=.99, lam=.95)
        losses = []
        for _ in range(manifest["ppo_epochs"]):
            permutation = torch.randperm(len(buffer))
            for offset in range(0, len(buffer), manifest["minibatch_size"]):
                indices = permutation[offset:offset + manifest["minibatch_size"]]
                loss = model.compute_ppo_loss(actor, critic, encoder, buffer, returns, advantages, indices,
                                               epsilon=.2, ent_coef=.03)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Graph PPO loss.")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, .5)
                optimizer.step()
                losses.append(float(loss.detach()))
        item = {"iteration": iteration + 1, "steps": total_steps, "curriculum_difficulty": difficulty,
                "mean_reward": float(np.mean(buffer.rewards)), "loss": float(np.mean(losses)),
                "elapsed_seconds": time.monotonic() - start}
        history.append(item)
        print(f"Graph PPO {iteration+1}/{iterations} | steps {total_steps} | reward {item['mean_reward']:.4f} | loss {item['loss']:.4f}", flush=True)
        save_json(output / "training_history.json", history)
        if (iteration + 1) % 10 == 0 or iteration + 1 == iterations:
            save_checkpoint(output, models, optimizer, manifest, total_steps, fingerprint, complete=iteration + 1 == iterations)
    save_json(output / "summary.json", {AGENT: "ok"})
    print("Training complete. Generate a report separately from the dashboard.", flush=True)


def evaluate(output):
    manifest = read_manifest(output)
    banks = create_bank(output / "scenario_bank", manifest["reward"]["profile"])
    validate_bank(banks)
    models, checkpoint = load_checkpoint(output, manifest, bank_fingerprint(banks))
    print("Evaluating Graph PPO on 20 held-out scenarios, 3 transition seeds each.", flush=True)
    embeddings, rows, episodes = collect(models, banks["test"], manifest["reward"]["profile"],
                  seed=manifest["seed"], repeats=manifest["evaluation"]["repeats"])
    from .report import analyze, PROFILE
    result = analyze(embeddings, rows, models[2])
    result.update(reward_profile=manifest["reward"]["profile"], episode_results=episodes,
                  checkpoint_steps=checkpoint["steps"], checkpoint_sha256=hashlib.sha256((output / "graph_ppo.pt").read_bytes()).hexdigest())
    report = {AGENT: result, "_experiment": manifest, "_score_profile": PROFILE,
              "_limitations": [
                  "This is a partial shared-diagnostic index, not the BipedalWalker score or a calibrated probability of hacking.",
                  "The historical scan-penalty profile is a reconstruction; no earlier notebook was supplied.",
                  "The notebook's graph uses simulator topology and its trackers sometimes inspect simulator truth; information leakage can affect interpretation.",
                  "No-valid-action states are explicitly truncated; time limits retain the copied PPO GAE convention.",
                  "Domain progress flags are context, not ground-truth exploit labels. A well-learned exploit can have low critic error.",
                  "Adversarial graph perturbation, critic sensitivity and continuous-action saturation were not evaluated.",
              ]}
    save_json(output / "runs/hackability_report.json", report)
    print(f"Saved NASim report: {output / 'runs/hackability_report.json'}", flush=True)
    return report


def preflight():
    """Short inference-only walks and forward loss check, zero optimizer steps."""
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="fallax-nasim-check-") as folder:
        banks = create_bank(Path(folder) / "scenarios", "nasim_current")
        models = create_models()
        before = [p.detach().clone() for net in models for p in net.parameters()]
        # Short inference walks on one network of each difficulty, no model updates.
        specs = [next(s for s in banks["validation"] if s["difficulty"] == d) for d in (0, 1, 2)]
        embeddings, rows, episodes = collect(models, specs, "nasim_current", step_cap=30)
        from .report import analyze
        report = analyze(embeddings, rows, models[2], trained=False)
        assert report["hackability_score"] is None
        # Exercise copied rollout/GAE/loss shapes without backward or optimizer.
        buffer = model.RolloutBuffer()
        episode = Episode(specs[0], "nasim_current", 2026)
        try:
            with torch.no_grad():
                for _ in range(8):
                    obs = episode.obs.copy()
                    decision = episode.choose(*models)
                    reward, terminated, truncated, _ = episode.step(decision)
                    buffer.store(obs, decision["chosen"], decision["log_prob"], decision["value"],
                                 reward, terminated or truncated, decision["pyg"], decision["mask"])
                    if terminated or truncated:
                        break
                returns, advantages = model.compute_returns_and_advantages(buffer, torch.tensor(0.0))
                loss = model.compute_ppo_loss(models[1], models[2], models[0], buffer,
                                             returns, advantages, list(range(len(buffer))))
                assert torch.isfinite(loss), "Forward PPO loss is not finite"
        finally:
            episode.close()
        after = [p for net in models for p in net.parameters()]
        assert all(torch.equal(a, b) for a, b in zip(before, after)), "Preflight changed model weights"
        result = {"status": "passed", "training_performed": False, "optimizer_steps": 0,
                  "scenario_counts": {k: len(v) for k,v in banks.items()},
                  "inference_steps": len(rows), "episodes": len(episodes),
                  "report_schema_checked": True, "ppo_loss_forward_checked": True, "weights_unchanged": True, "runtime": runtime_versions()}
        print(json.dumps(result, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preflight", choices=("preflight", "train", "report"))
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="nasim_current")
    parser.add_argument("--steps", type=int, default=102400)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "preflight":
        return preflight()
    if args.experiment is None:
        parser.error("--experiment is required for train/report")
    output = args.experiment.resolve()
    if args.mode == "train" and not (output / "experiment.json").exists():
        if output.exists() and any(output.iterdir()):
            parser.error("Use a new empty experiment directory.")
        if not 1024 <= args.steps <= 10000000:
            parser.error("--steps must be between 1024 and 10,000,000")
        save_json(output / "experiment.json", make_manifest(args.profile, args.steps))
    if args.mode == "train":
        train(output)
    else:
        evaluate(output)


if __name__ == "__main__":
    main()
