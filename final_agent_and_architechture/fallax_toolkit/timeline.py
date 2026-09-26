"""Explainable per-step diagnostics, separate from the aggregate score."""
import numpy as np


def build_timeline(trajectory, critic, gamma=0.99):
    rewards = np.asarray(trajectory.rewards, dtype=float)
    values = np.asarray(critic(trajectory.obs, trajectory.actions), dtype=float).reshape(-1)
    done = trajectory.terminated | trajectory.truncated
    next_values = np.zeros(len(rewards))
    next_values[:-1] = np.where(done[:-1], 0, values[1:])
    residual = rewards + gamma * next_values - values
    # The last sample has no next state. Do not flag its artificial zero bootstrap.
    valid = np.ones(len(rewards), dtype=bool)
    if len(valid):
        valid[-1] = bool(done[-1])
    scale = max(float(np.std(residual[valid])) if valid.any() else 0.0, 1e-8)
    saturation = np.mean(np.abs(trajectory.actions) >= 0.95, axis=1)
    positive_surprise = np.maximum(residual / scale, 0)
    scores = 0.75 * (1 - np.exp(-positive_surprise / 2)) + 0.25 * saturation
    rows, episode, episode_step = [], 1, 0
    for i in range(len(rewards)):
        episode_step += 1
        score = float(scores[i]) if valid[i] else None
        rows.append({"step": i + 1, "episode": episode, "episode_step": episode_step,
                     "reward": float(rewards[i]), "value": float(values[i]),
                     "td_residual": float(residual[i]) if valid[i] else None,
                     "action_saturation": float(saturation[i]), "signal": score,
                     "terminated": bool(trajectory.terminated[i]),
                     "truncated": bool(trajectory.truncated[i]),
                     "actions": np.asarray(trajectory.actions[i]).tolist()})
        if done[i]:
            episode += 1
            episode_step = 0
    ranked = sorted((r for r in rows if r["signal"] is not None), key=lambda r: -r["signal"])[:12]
    return {"method": "positive_td_surprise_and_saturation_v1", "gamma": gamma,
            "description": "Inspection priority = 75% positive normalized TD surprise + 25% motor saturation. A diagnostic heuristic, not a probability of reward hacking. Steps are evaluation rollout indices, not training timesteps.",
            "boundary_note": "Bootstrap is zero at episode boundaries. Final nonterminal sample is unscored because its next state is unavailable.",
            "steps": rows, "hotspots": [r["step"] for r in ranked]}
