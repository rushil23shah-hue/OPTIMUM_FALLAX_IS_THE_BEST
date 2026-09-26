# BipedalWalker reward experiment

Training has not been started. From this directory, use the existing project environment:

```powershell
../.venv/Scripts/python.exe interface.py
```

All six agents train from scratch using `optimized_v1`. Checkpoints, CSVs,
plots, the manifest and training summary go into `experiments/optimized_v1`.
Existing checkpoints and `runs/hackability_report.json` stay intact.
The interface refuses a nonempty experiment directory to prevent accidental
resuming or overwriting. For another fresh run, pass `--experiment experiments/optimized_v1_run2`.
Individual agent scripts retain their original reward by default; use the
interface to select the reward and isolate outputs consistently.

After all six agents finish, run:

```powershell
../.venv/Scripts/python.exe run_hackability_report.py
```

The new report is `experiments/optimized_v1/runs/hackability_report.json`.
For a custom experiment directory, pass the same `--experiment` argument.
The report reads the training manifest, checks reward parameters and completion,
loads only that directory's checkpoints, and rejects missing checkpoints.
Scoring weights and thresholds are unchanged. `reward_audit` records original
and new extrinsic returns on the same clean diagnostic rollout, displacement,
and failures for the optimized experiment. These are totals over a fixed step
budget, including partial episodes, not average completed-episode returns.

## Candidate reward

`reward_wrapper.py` applies the following reward before existing normalization
or intrinsic curiosity bonuses:

```text
4.333333 * signed hull displacement
- 0.028 * sum(abs(clipped motor actions))
- 0.10 * min(hull angle squared, 1)
- 0.002 * min(hull angular velocity squared, 25)
- 0.02 * mean((action - previous action) squared)
- 0.01 per step
```

A native failure overrides this sum with exactly -100. Success and time-limit
truncation retain the ordinary per-step reward. The original progress and
effort coefficients are retained; the original change-in-posture term is
replaced with posture and angular-speed costs, plus smoothness and time costs.
There is no positive standing/survival bonus. Backward movement costs reward.
The action-history cost adds dependence on the previous action (not included
in the unchanged 24-dimensional observation), a deliberate tradeoff.
The coefficients are a proposed design, not empirically tuned or proven optimal.

The wrapper is inside PPO normalization/statistics and is shared by all six
training paths and report environments. SAC's legacy -100 to -1 adapter is
bypassed for the optimized profile so it cannot erase the new failure penalty.
The original profile retains SAC's historical behavior for reproducibility.
Native reward semantics were checked against the installed Gymnasium 1.1.1
source and the [upstream implementation](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py).

## Interpreting the comparison

A changed score alone does not prove the toolkit detects reward hacking, nor
does this candidate guarantee lower scores. Compare behavior, native returns,
failure rates, sub-scores and perturbation robustness as well as the aggregate.
Use matched training budgets/seeds and multiple runs for a stronger experiment.

TD3 now saves and restores both trained critics. Its old checkpoint contained
only the actor, so the old report's TD3 critic-based values used a random critic;
do not attribute that portion of an old/new score change to reward design.
For a fresh original-reward control using the corrected checkpoint pipeline:

```powershell
../.venv/Scripts/python.exe interface.py --reward-profile original
../.venv/Scripts/python.exe run_hackability_report.py --experiment experiments/original
```

Existing report limitations remain: PPO normalization statistics are rebuilt
during evaluation; curiosity critics learn extrinsic plus intrinsic targets
but diagnostics use extrinsic rewards; cross-agent normalization differs.
These are recorded in the new report and limit claims of proof.

Validation without training:

```powershell
../.venv/Scripts/python.exe -B -m unittest test_reward_wrapper -v
```

Tests cover reward accounting, clipping, state reset, stationary/backward
behavior, failure versus success/truncation, PPO vector compatibility, isolated
launch configuration, and refusal to overwrite. A paired 100-step real Box2D
check verifies identical observations/termination with different rewards.
