# Recorded evidence and KRISIS investigations

## Using the dashboard

Ask KRISIS to explain a comparison, then select **Guide me through the evidence**.
The guide highlights a matched agent's comparison row, explains the largest raw
sub-score change, opens the second report's agent analysis, and selects a recorded
timeline hotspot. Use Back, Next evidence, Finish, or Close. Missing historical
traces are stated explicitly; Walker replay is a separate fresh evaluation.

KRISIS is a local, rule-based report assistant. Its statements use saved fields
and templates, not an external language model. A diagnostic change does not prove
that a particular reward term caused a behavioral change.

## NASim Reward Crime Scene

Select NASim, a completed report, and Timeline. Select a step, then Investigate
selected step. Playback stays inside that recorded episode. Actions, targets,
outcomes, progress deltas, reward terms, and domain flags come from the report.
The target diagram is not reconstructed network topology. Full before/after
network states are unavailable. Explain this moment sends the selected evidence
to KRISIS.

## Walker simulation replay

Select BipedalWalker, an experiment and an agent on Timeline, then Watch selected
agent. Each recording is one fresh evaluation episode using the selected saved
checkpoint and seed 42. JPEG frames show the state after each action. Frames,
actions, rewards, critic values and timeline diagnostics belong to the same
recorded trajectory. Recording stops at termination or truncation.

These episodes are separate from aggregate report evaluation. Existing metrics,
checkpoints and report scores are not overwritten. PPO's existing evaluation
adapter reconstructs normalization statistics; its displayed reward is explicitly
labelled normalized evaluation reward. Original reward components are not inferred.
Optimized reward terms are recorded when provided by the wrapper. Curiosity
critics may include intrinsic reward while the rollout records extrinsic reward.

Dashboard Walker evaluation jobs also generate these separate replay episodes
after saving the aggregate report. To record without recomputing the report, run
from the engine directory with its Python environment:

```powershell
..\.venv\Scripts\python.exe -m fallax_toolkit.walker_replay --experiment experiments/optimized_v1
```

Use `--experiment .` for legacy checkpoints and `--agents ppo sac` for a subset.
This runs inference only. Recordings live in each experiment's `runs/replays/`,
are ignored by Git, and must be generated locally after cloning. The original
TD3 checkpoint lacks trained critics and cannot produce a diagnostic replay;
the optimized TD3 checkpoint can. Errors are recorded in `status.json`.

## Verification

The initial recording pass produced 11 complete episodes and 5,571 synchronized
frames. Episode boundaries, sequential step indices, and JPEG decoding were
checked. Browser checks exercised frame stepping, playback to the last frame,
KRISIS handoff, guided comparison navigation, metric highlighting and tour exit.
