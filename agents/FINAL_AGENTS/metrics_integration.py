"""Adapters and held-out rollout collection for the six FINAL_AGENTS trainers."""
from pathlib import Path
import json
import numpy as np

try:
    from .metrics import (Trajectory, evaluate_trajectory, termination_timing_distribution,
                          return_calibration_gap_variance)
except ImportError:
    from metrics import (Trajectory, evaluate_trajectory, termination_timing_distribution,
                         return_calibration_gap_variance)


def attach_metrics_context(model, env_id, gamma=0.99, obs_rms=None, clip=None,
                          training_steps=None, training_seed=None, policy_updates=None,
                          reward_id="environment_raw_extrinsic", run_id=None):
    """Keep the fitted normalizer with the returned model, without updating it."""
    model.metrics_context = {"env_id": env_id, "gamma": gamma, "clip": clip,
                             "training_steps": training_steps,
                             "training_seed": training_seed,
                             "policy_updates": policy_updates,
                             "reward_id": reward_id, "run_id": run_id}
    if obs_rms is not None:
        model.metrics_context.update(mean=np.array(obs_rms.mean, copy=True),
                                     var=np.array(obs_rms.var, copy=True))
    return model


class AgentAdapter:
    def __init__(self, name, model, low, high):
        self.name, self.model = name, model
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.context = model.metrics_context
        self.gamma = self.context["gamma"]
        self.observation_scale = np.sqrt(self.context["var"] + 1e-8) if "var" in self.context else 1.0
        self.objective = {
            "ppo": "extrinsic value under stochastic training policy; deterministic evaluation",
            "ppo_icm": "extrinsic plus batch-normalized ICM intrinsic reward",
            "model_based": "extrinsic plus batch-normalized ICM; real and dreamed training data",
            "td3": "extrinsic Q, minimum of twin critics",
            "td3_rnd": "extrinsic plus RND intrinsic reward, minimum of twin critics",
            "sac": "adapted reward (fall penalties changed) plus entropy; minimum of twin soft Q critics",
        }[name]

    def _tensor(self, obs):
        import torch
        obs = np.asarray(obs, dtype=np.float32)
        if "mean" in self.context:
            obs = (obs - self.context["mean"]) / self.observation_scale
            if self.context["clip"] is not None:
                obs = np.clip(obs, -self.context["clip"], self.context["clip"])
        module = self.model if self.name in ("ppo", "ppo_icm", "model_based") else self.model.actor
        device = next(module.parameters()).device
        return torch.as_tensor(obs, dtype=torch.float32, device=device)

    def policy(self, obs):
        """Deterministic actions in environment coordinates, without exploration."""
        import torch
        with torch.no_grad():
            x = self._tensor(obs)
            if self.name == "ppo":
                a = self.model.actor_mean(x)
            elif self.name in ("ppo_icm", "model_based"):
                a = self.model.actor_mean(self.model.actor_fc(x))
            elif self.name == "sac":
                a, _ = self.model.actor(x, deterministic=True, with_logprob=False)
                # Existing SAC trainer maps [-1,1] by the scalar high[0].
                a = a * float(self.high[0])
            else:
                a = self.model.actor(x)
            return np.clip(a.cpu().numpy(), self.low, self.high)

    def critic(self, obs, actions=None):
        import torch
        with torch.no_grad():
            x = self._tensor(obs)
            if self.name in ("ppo", "ppo_icm", "model_based"):
                value = self.model.get_value(x)
            else:
                if actions is None:
                    actions = self.policy(obs)
                a = torch.as_tensor(actions, dtype=torch.float32, device=x.device)
                if self.name == "sac":
                    q1, q2 = self.model.critic(x, a / float(self.high[0]))
                elif self.name == "td3":
                    q1, q2 = self.model.q1(x, a), self.model.q2(x, a)
                else:
                    q1, q2 = self.model.critic_1(x, a), self.model.critic_2(x, a)
                value = torch.minimum(q1, q2)
            return value.cpu().numpy().reshape(-1)


def collect_trajectories(env, policy, episodes=5, seed=10000, max_steps=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    horizon = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    limit = max_steps if max_steps is not None else horizon
    if limit is None or limit < 1:
        raise ValueError("provide max_steps for an environment without a time limit")
    trajectories = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        observations, actions, rewards, terminated, truncated = [np.array(obs, copy=True)], [], [], [], []
        for _ in range(limit):
            batch = np.asarray(policy(np.asarray(obs)[None, :]))
            if batch.shape != (1,) + env.action_space.shape or not np.isfinite(batch).all():
                raise ValueError("policy returned invalid actions")
            action = np.clip(batch[0], env.action_space.low, env.action_space.high).astype(env.action_space.dtype)
            obs, reward, term, trunc, _ = env.step(action)
            observations.append(np.array(obs, copy=True))
            actions.append(action.copy())
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            if term or trunc:
                break
        trajectories.append(Trajectory(observations, actions, rewards, terminated, truncated, horizon))
    return trajectories


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_agent(name, model, output_dir, episodes=5, seed=10000, max_steps=None):
    import gymnasium as gym
    env = gym.make(model.metrics_context["env_id"])
    try:
        adapter = AgentAdapter(name, model, env.action_space.low, env.action_space.high)
        trajectories = collect_trajectories(env, adapter.policy, episodes, seed, max_steps)
    finally:
        env.close()
    reports = [evaluate_trajectory(t, adapter.critic, adapter.gamma, adapter.low, adapter.high,
                                   adapter.observation_scale, seed + i) for i, t in enumerate(trajectories)]
    report = {"schema_version": 1, "agent": name, "env_id": adapter.context["env_id"],
              "checkpoint": "end_of_training", "evaluation_seed": seed, "gamma": adapter.gamma,
              "training_steps": adapter.context.get("training_steps"),
              "training_seed": adapter.context.get("training_seed"),
              "policy_updates": adapter.context.get("policy_updates") or 0,
              "reward_id": adapter.context.get("reward_id", "environment_raw_extrinsic"),
              "evaluation_protocol": {
                  "episodes": len(reports), "seed_start": seed,
                  "max_steps": max((len(t.rewards) for t in trajectories), default=0),
                  "deterministic_policy": True, "raw_extrinsic_rewards": True,
              },
              "run_id": adapter.context.get("run_id") or str(output_dir),
              "action_low": adapter.low.tolist(), "action_high": adapter.high.tolist(),
              "observation_preprocessing": {
                  key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in adapter.context.items() if key in ("mean", "var", "clip")},
              "reward_basis": "raw extrinsic environment reward", "critic_objective": adapter.objective,
              "interpretation": "Diagnostics, not a reward-hacking verdict. Critic gaps may include objective or policy mismatch.",
              "episodes": reports,
              "termination_timing_distribution": termination_timing_distribution(trajectories),
              "return_calibration_gap_variance": return_calibration_gap_variance(
                  [r["return_calibration_gap"]["gaps"] for r in reports])}
    report["report_id"] = f"{report['run_id']}::{name}::{report['training_steps']}::{seed}"
    for i, episode in enumerate(report["episodes"]):
        episode["evidence"] = {"trajectory": str(Path(output_dir) / f"trajectory_{i:03d}.npz"),
                                "evaluation_seed": seed + i}
    output_dir = Path(output_dir)
    write_json(output_dir / "metrics.json", report)
    for i, t in enumerate(trajectories):
        np.savez_compressed(output_dir / f"trajectory_{i:03d}.npz", obs=t.obs, actions=t.actions,
                            rewards=t.rewards, terminated=t.terminated, truncated=t.truncated,
                            horizon=-1 if t.horizon is None else t.horizon)
    return adapter, trajectories, report
