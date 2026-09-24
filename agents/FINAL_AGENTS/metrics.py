"""Environment-independent diagnostics; values are evidence, not hacking verdicts.

Critics accept (observations, actions=None) and return one scalar per row.
A trajectory is one episode or contiguous fragment: T actions and T+1 states.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


def _array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


@dataclass
class Trajectory:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    horizon: int | None = None

    def __post_init__(self):
        self.obs = _array(self.obs, "obs")
        self.actions = _array(self.actions, "actions")
        self.rewards = _array(self.rewards, "rewards")
        n = len(self.rewards)
        if n == 0 or self.rewards.shape != (n,):
            raise ValueError("rewards must be a nonempty vector")
        if self.obs.ndim != 2 or len(self.obs) != n + 1:
            raise ValueError("obs must have shape (T+1, observation_dim)")
        if self.actions.ndim != 2 or len(self.actions) != n or self.actions.shape[1] == 0:
            raise ValueError("actions must have shape (T, action_dim)")
        for name in ("terminated", "truncated"):
            flags = np.asarray(getattr(self, name))
            if flags.shape != (n,) or not np.isin(flags, [0, 1]).all():
                raise ValueError(f"{name} must contain T boolean flags")
            setattr(self, name, flags.astype(bool))
        if np.any(self.terminated[:-1] | self.truncated[:-1]):
            raise ValueError("split trajectories at every reset boundary")
        if self.horizon is not None and (self.horizon < n or int(self.horizon) != self.horizon):
            raise ValueError("horizon must be an integer >= trajectory length")


def _values(critic, obs, actions=None):
    values = _array(critic(obs, actions), "critic output")
    if values.shape not in ((len(obs),), (len(obs), 1)):
        raise ValueError("critic must return one scalar per observation")
    return values.reshape(-1)


def _discount(gamma):
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")


def _summary(x):
    x = _array(x, "samples")
    return {"mean": float(x.mean()), "std": float(x.std()),
            "variance": float(x.var()), "mean_abs": float(np.abs(x).mean()),
            "max_abs": float(np.abs(x).max())}


def td_error_anomaly(trajectory, critic, gamma=0.99):
    """Q critics must supply their policy's continuation value when actions=None."""
    _discount(gamma)
    t = trajectory
    current = _values(critic, t.obs[:-1], t.actions)
    # Never evaluate terminal states (some critics have no valid terminal input).
    following = np.zeros(len(t.rewards))
    mask = ~t.terminated
    if mask.any():
        following[mask] = _values(critic, t.obs[1:][mask])
    delta = t.rewards + gamma * following - current
    result = _summary(delta)
    result["skewness"] = float(np.mean(((delta - delta.mean()) / delta.std()) ** 3)) if delta.std() > 0 else 0.0
    result["residuals"] = delta.tolist()
    return result


def return_calibration_gap(trajectory, critic, gamma=0.99):
    _discount(gamma)
    t = trajectory
    # A truncation/collector cutoff is not an MDP terminal; retain its bootstrap.
    tail = 0.0 if t.terminated[-1] else float(_values(critic, t.obs[-1:])[0])
    returns = np.empty(len(t.rewards))
    for i in reversed(range(len(returns))):
        tail = t.rewards[i] + gamma * tail
        returns[i] = tail
    gap = returns - _values(critic, t.obs[:-1], t.actions)
    x = np.arange(len(gap), dtype=float)
    x -= x.mean()
    result = _summary(gap)
    result.update(gaps=gap.tolist(), returns=returns.tolist(),
                  trend_per_step=float(x @ gap / (x @ x)) if len(gap) > 1 else 0.0,
                  bootstrapped=not bool(t.terminated[-1]))
    return result


def reward_concentration(rewards):
    """Gini/entropy of absolute reward (also accepts absolute TD surprise)."""
    mass = np.abs(_array(rewards, "rewards"))
    if mass.ndim != 1 or len(mass) == 0:
        raise ValueError("expected a nonempty vector")
    n, total = len(mass), mass.sum()
    if total == 0:
        return {"gini": 0.0, "entropy": None, "normalized_entropy": None, "mass": 0.0}
    p = mass[mass > 0] / total
    entropy = float(-np.sum(p * np.log(p)))
    gini = float(2 * (np.arange(1, n + 1) @ np.sort(mass)) / (n * total) - (n + 1) / n)
    return {"gini": gini, "entropy": entropy,
            "normalized_entropy": entropy / np.log(n) if n > 1 else 0.0,
            "mass": float(total)}


def reward_rate_normalization(trajectory):
    return {"total_reward": float(trajectory.rewards.sum()),
            "steps": len(trajectory.rewards), "reward_rate": float(trajectory.rewards.mean())}


def termination_timing_distribution(trajectories, bins=10):
    if not trajectories or bins < 1:
        raise ValueError("need trajectories and positive bins")
    fractions, literal, steps = [], [], []
    for t in trajectories:
        where = np.flatnonzero(t.terminated)
        if len(where):
            step = int(where[0]) + 1
            steps.append(step)
            literal.append(float(where[0] / len(t.rewards)))
            if t.horizon is not None:
                fractions.append(step / t.horizon)
    counts, edges = np.histogram(fractions, bins=bins, range=(0, 1))
    return {"episodes": len(trajectories), "terminated_count": len(steps),
            "truncated_only_count": sum(bool(t.truncated[-1] and not t.terminated[-1]) for t in trajectories),
            "unfinished_count": sum(not bool(t.truncated[-1] or t.terminated[-1]) for t in trajectories),
            "termination_steps": steps, "horizon_fractions": fractions,
            "missing_horizon_count": len(steps) - len(fractions),
            "pdf_episode_fractions": literal, "histogram": counts.tolist(), "bin_edges": edges.tolist()}


def action_saturation_entropy(actions, low=-1.0, high=1.0, eps=0.01, bins=20):
    """Marginal histogram differential entropy in normalized [-1,1] coordinates.

    This is a finite-bin estimate, not joint entropy or policy entropy.
    """
    actions = _array(actions, "actions")
    low, high = _array(low, "low"), _array(high, "high")
    if actions.ndim != 2 or not len(actions) or not 0 < eps < 1 or bins < 2 or np.any(high <= low):
        raise ValueError("need nonempty actions, finite ordered bounds, 0<eps<1, bins>=2")
    z = 2 * (actions - low) / (high - low) - 1
    if np.any(np.abs(z) > 1 + 1e-6):
        raise ValueError("actions outside bounds; supply executed actions")
    z = np.clip(z, -1, 1)
    entropy = []
    for column in z.T:
        counts, _ = np.histogram(column, bins=bins, range=(-1, 1))
        p = counts[counts > 0] / len(column)
        entropy.append(float(-np.sum(p * np.log(p))))
    return {"saturation_fraction": float((np.abs(z) > 1 - eps).mean()),
            "saturation_per_dimension": (np.abs(z) > 1 - eps).mean(axis=0).tolist(),
            "histogram_entropy_per_dimension": entropy,
            "differential_entropy_per_dimension": (np.array(entropy) + np.log(2 / bins)).tolist(),
            "bins": bins, "eps": eps}


def return_calibration_gap_variance(gaps_by_episode):
    """Population variances at matched step indices within one checkpoint.

    No padding or interpolation: later indices use only surviving episodes.
    """
    arrays = [_array(g, "gaps") for g in gaps_by_episode]
    if not arrays or any(a.ndim != 1 or not len(a) for a in arrays):
        raise ValueError("need nonempty gap vectors")
    counts, variances = [], []
    for i in range(max(map(len, arrays))):
        samples = [a[i] for a in arrays if i < len(a)]
        counts.append(len(samples))
        variances.append(float(np.var(samples)) if len(samples) >= 2 else None)
    return {"within_episode": [float(a.var()) for a in arrays],
            "episode_mean_variance": float(np.var([a.mean() for a in arrays])),
            "matched_step_variance": variances, "matched_step_count": counts}


def critic_sensitivity(trajectory, critic, sigma=0.01, observation_scale=1.0, samples=8, seed=0):
    if not np.isfinite(sigma) or sigma <= 0 or samples < 1:
        raise ValueError("sigma and samples must be positive")
    scale = _array(observation_scale, "observation_scale")
    if np.any(scale <= 0):
        raise ValueError("observation_scale must be positive")
    obs, actions = trajectory.obs[:-1], trajectory.actions
    base = _values(critic, obs, actions)
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(samples):
        perturbed = obs + rng.normal(size=obs.shape) * sigma * scale
        # Hold actions fixed: this measures critic sensitivity, not policy changes.
        changes.append(np.abs(_values(critic, perturbed, actions) - base))
    changes = np.asarray(changes)
    return {**_summary(changes), "per_step": changes.mean(axis=0).tolist(),
            "sigma": sigma, "samples": samples, "seed": seed}


def cross_agent_policy_distance(shared_obs, policies, gaussian_parameters=None):
    """Evaluate every policy on EXACTLY the same raw observations.

    Optional parameter callables return (mean, std) for diagonal Gaussians in
    the same coordinates. Do not use unsquashed-Gaussian KL for clipped policies.
    """
    states = _array(shared_obs, "shared_obs")
    if states.ndim != 2 or not len(states):
        raise ValueError("need a nonempty shared observation batch")
    actions = {name: _array(policy(states.copy()), name) for name, policy in policies.items()}
    for a in actions.values():
        if a.ndim != 2 or len(a) != len(states):
            raise ValueError("policies must return (N, action_dim)")
    result = {}
    for first, second in combinations(actions, 2):
        a, b = actions[first], actions[second]
        if a.shape != b.shape:
            raise ValueError("policies must share an action space")
        entry = {"mean_l2": float(np.linalg.norm(a - b, axis=1).mean()),
                 "rms_per_component": float(np.sqrt(np.mean((a - b) ** 2)))}
        if gaussian_parameters and first in gaussian_parameters and second in gaussian_parameters:
            params = []
            for name in (first, second):
                mean, std = map(lambda v: _array(v, "Gaussian parameter"), gaussian_parameters[name](states.copy()))
                if mean.shape != a.shape or std.shape != a.shape or np.any(std <= 0):
                    raise ValueError("Gaussian parameters must match actions, with positive std")
                params.append((mean, std))
            (m1, s1), (m2, s2) = params
            def kl(m, s, other_m, other_s):
                return float(np.mean(np.sum(np.log(other_s / s) + (s*s + (m-other_m)**2) / (2*other_s*other_s) - 0.5, axis=1)))
            entry.update(kl_forward=kl(m1, s1, m2, s2), kl_reverse=kl(m2, s2, m1, s1))
        result[f"{first}__{second}"] = entry
    return {"shared_state_count": len(states), "pairs": result}


def evaluate_trajectory(trajectory, critic=None, gamma=0.99, low=-1.0, high=1.0,
                        observation_scale=1.0, seed=0):
    report = {"reward_rate_normalization": reward_rate_normalization(trajectory),
              "reward_concentration": reward_concentration(trajectory.rewards),
              "action_saturation_entropy": action_saturation_entropy(trajectory.actions, low, high)}
    if critic is not None:
        report["td_error_anomaly"] = td_error_anomaly(trajectory, critic, gamma)
        report["return_calibration_gap"] = return_calibration_gap(trajectory, critic, gamma)
        report["critic_sensitivity"] = critic_sensitivity(trajectory, critic, observation_scale=observation_scale, seed=seed)
    return report
