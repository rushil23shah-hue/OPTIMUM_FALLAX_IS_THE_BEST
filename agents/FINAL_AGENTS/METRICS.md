# Reward-hacking diagnostics

`metrics.py` implements all six additional metrics in
`Reward_Hacking_Additional_Metrics.pdf` and the three baseline metrics it names.
The core only depends on NumPy. `metrics_integration.py` connects it to the six
trainers currently registered in `interface.py`: PPO, PPO+ICM, TD3, TD3+RND, SAC,
and model-based Dyna-PPO. The legacy agents under `agents/wait` are not registered
and do not yet have model-specific adapters.

For a standalone demonstration without PyTorch or Gymnasium, run
`python agents/FINAL_AGENTS/metrics_example.py`. It prints every metric using
explicitly labeled synthetic data.

## Run through the interface

From the repository root, with Python 3.10+ and the agents' dependencies installed:

```powershell
python agents/FINAL_AGENTS/interface.py --agents ppo td3 --steps 100000 --episodes 10
```

Omit `--agents` to run all six. Every agent receives the same `--env-id`
(default `BipedalWalker-v3`) and training seed. `--steps` overrides the training
budget; without it, each trainer retains its existing default. PPO+ICM uses
complete rollouts and can round down; Dyna-PPO rounds up. SAC completes its last
episode. These trainer budgets are not identical update counts.

For a short integration check, not a trained detector:

```powershell
python agents/FINAL_AGENTS/interface.py --steps 32 --episodes 2 --eval-max-steps 16
python -m unittest discover -s tests -v
```

Evaluation uses fresh environments, seeds starting at `--seed + 10000`, frozen
normalizers, deterministic actions, and raw extrinsic rewards. A collector cutoff
does not masquerade as an environment termination or truncation. Importing the
interface does not start training; importing SAC no longer parses command-line
arguments. Existing standalone training entry points remain usable.

Each invocation creates a timestamped directory under `runs/` (override with
`--run-dir`). It contains:

- `summary.json`: success/failure and report path per agent; failures do not stop
  remaining agents. The command exits nonzero if any agent failed.
- `<agent>/metrics.json`: per-episode metrics and batch diagnostics at the final
  training checkpoint, including reward basis, critic objective, gamma, and seed.
- `<agent>/trajectory_000.npz`, etc.: raw observations, executed actions,
  rewards, separate terminated/truncated flags, and environment horizon (`-1`
  means unknown).
- `cross_agent_metrics.json`: pairwise distances when at least two agents succeed.
- `shared_observations.npy`: the exact shared raw state batch used for comparison.

## Definitions and interpretation

| Metric | Implementation |
| --- | --- |
| TD-error anomaly | `r + gamma * continuation_value - current_value`; mean, standard deviation, variance, absolute magnitude, skewness, and residual array |
| Return calibration gap | Discounted realized return minus critic value; mean, absolute magnitude, trend, variance, and full gap/return arrays |
| Reward concentration | Gini and entropy of **absolute** rewards, preserving negative penalties as mass; zero total mass has undefined entropy (`null`) |
| Reward-rate normalization | Raw reward sum divided by number of transitions |
| Termination-timing distribution | One-based terminal step divided by environment horizon; histogram excludes nonterminated episodes and episodes with unknown horizon, whose counts are reported separately |
| Action saturation / entropy | Saturation after mapping each finite action bound to `[-1, 1]`; marginal histogram entropy and finite-bin differential-entropy estimate in those coordinates |
| Gap variance | Within-episode variance, variance of episode means, and variance across rollouts at each matched step; unequal lengths are not zero-padded, and single-sample entries are `null` |
| Critic sensitivity | Mean absolute critic change under seeded Gaussian observation perturbations, holding executed actions fixed; default sigma 0.01, eight perturbations |
| Cross-agent policy distance | Mean action-vector L2 and component RMS on exactly the same raw states, each processed through that agent's own frozen normalizer |

The PDF's literal zero-based `argmax(terminated)/episode_length` is also saved as
`pdf_episode_fractions`, for terminated episodes only. It is near one for nearly
every completed episode, so `horizon_fractions` is the useful timing diagnostic.
Termination is an event, not necessarily failure or falling.

The entropy estimate is marginal, bin-dependent, and can be negative. It is not
joint action entropy or stochastic-policy entropy. Cross-agent comparison uses
executed deterministic actions for all six algorithms; it does not infer KL from
unmatched rollouts or confuse clipped Gaussian actions with a Gaussian density.
The standalone function also supports exact directional KL for explicitly
supplied diagonal-Gaussian distributions in common coordinates.

For normalized agents, perturbations are scaled by their frozen training standard
deviation in raw observation space. Other agents use scale 1 in raw coordinates;
callers can provide a domain-appropriate `observation_scale` to the core metric.
Clipping, when used by the agent, is preserved. Observation perturbations are not
guaranteed to be physically realizable states.

True terminal transitions bootstrap to zero. Time-limit truncations and partial
rollouts bootstrap from the final observation, so their returns are estimates,
not complete Monte Carlo returns. Q adapters use the minimum of twin critics,
and evaluate the deterministic policy when no action is supplied.

**These are diagnostics, not an automatic reward-hacking verdict.** A high score
can reflect poor training, reward scaling, distribution shift, legitimate
saturation, or critic error. Similar policies can solve a task correctly. The
framework deliberately does not invent thresholds, labels, or a combined risk
score without a validated baseline.

Critic-objective mismatch matters: PPO+ICM, TD3+RND and Dyna-PPO include intrinsic
rewards; SAC changes large negative rewards and trains soft Q values with an
entropy term. The reports explicitly compare these critics with **raw extrinsic**
returns, so gaps are not exact training Bellman residuals for these agents.
PPO's stochastic training-policy value also differs from deterministic evaluation.
For objective-aligned calibration, supply a critic trained on the evaluation
reward/policy, or reconstruct that objective in a custom trajectory/critic pair.

## Use an existing trained agent without retraining

The trainers now return their model/agent with a copied `metrics_context` containing
the environment, discount and any observation-normalization statistics. PPO still
returns its original network type. After training in your own script:

```python
from metrics_integration import evaluate_agent

adapter, trajectories, report = evaluate_agent(
    "ppo", trained_model, "runs/my_checkpoint/ppo", episodes=10, seed=10000
)
```

If loading a checkpoint, restore its original normalizer and call
`attach_metrics_context` first. Do not recompute it on evaluation states.

For any other agent, use the environment-independent interface directly:

```python
from metrics import Trajectory, evaluate_trajectory, cross_agent_policy_distance

t = Trajectory(obs, actions, rewards, terminated, truncated, horizon=1600)
report = evaluate_trajectory(t, critic, gamma=0.99, low=action_low, high=action_high)
comparison = cross_agent_policy_distance(shared_raw_states, {
    "agent_a": batch_policy_a,
    "agent_b": batch_policy_b,
})
```

`obs` has shape `(T+1, obs_dim)`, actions `(T, action_dim)`, and the three reward/flag
arrays `(T,)`. Split at every reset. `critic(obs_batch, actions=None)` returns
`(N,)` or `(N,1)`; when actions are omitted it must provide continuation values
under the evaluation policy. Policy callables return `(N, action_dim)` in common
environment units. Only continuous, bounded action spaces and flat vector
observations are supported by the supplied adapters. Custom agents can provide
their own preprocessing and critic callables without changing the metric code.

Compute gap variance separately at each checkpoint; do not pool different training
stages into the same matched-step comparison. Late-step statistics only include
episodes surviving to that step, as shown by the reported counts.
