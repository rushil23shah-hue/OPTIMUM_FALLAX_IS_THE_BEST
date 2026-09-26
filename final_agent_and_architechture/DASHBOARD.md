# Fallax local dashboard

The dashboard is a working local control surface for the six BipedalWalker agents.
Its HTML, CSS and JavaScript are bundled inside the Python distribution. Installing
the package brings the interface with it; there is no separate website download,
cloud service, account, Node runtime or frontend build step. Installation does not
automatically launch a server or start training.

## Open this checkout

From `final_agent_and_architechture`:

```powershell
../.venv/Scripts/python.exe -B dashboard.py
```

Or double-click `Open Fallax.cmd` on Windows. Keep the server process running while
using the interface. The default address is `http://127.0.0.1:8765`. Use `--port 8766`
if that port is occupied. `--no-browser` suppresses automatic browser opening.

## Install the package

From the directory containing `pyproject.toml`:

```powershell
python -m pip install .
fallax --root "path/to/final_agent_and_architechture"
```

For a fresh environment with training dependencies, use `python -m pip install ".[training]"`.
Without `--root`, an installed package uses `fallax-data` in the current directory;
a source checkout with existing `runs/` uses its existing data directory.
The package has not been published to PyPI. Distribute a built wheel or the source.

## Screens and controls

- **Overview:** real report scores, suite mean, flagged agents and timestep coverage.
- **Agent analysis:** all seven sub-scores, weights and contributions, context,
  raw metrics, reward audit and adversarial noise sweep.
- **Event timeline:** per-step reward, critic estimate, TD residual, actions,
  saturation and boundaries, with a slider and ranked inspection hotspots.
- **Compare rewards:** shared agents' score deltas and recorded reward configurations.
- **Experiments & jobs:** isolated training runs, agent selection, reward selection,
  default or shared step budgets, completion status, report generation and live logs.
- **Method & setup:** scope, limitations, installed modules and the active interpreter.
- **Export report:** download all report data as JSON, including available traces.

New training runs cannot overwrite a nonempty experiment directory. Selected agents
run sequentially in subprocesses. Job logs and history live under `.fallax/jobs/`.
You can stop only jobs started by the current dashboard process. Stopping training
retains intermediate files and marks it stopped; it does not promise a resumable
optimizer checkpoint. A new training run must use a fresh name.

Training already launched with `interface.py` continues independently. The dashboard
polls its saved manifest/summary and can evaluate agents marked `ok`. It does not
claim a live process state or recover console output for terminal-launched training.
After a server restart, unfinished historical dashboard jobs are marked `detached`;
their process ownership is unknown, so they cannot be stopped from the new server.

Step budgets are requested environment interactions, not an exact common compute
budget. PPO rounds down to vector rollout batches; PPO-ICM and model-based PPO round
down to complete rollouts. SAC may finish its current episode after the requested
budget. Short runs are pipeline checks, not evidence of convergence.

## Evidence and limits

New reports now include evaluation traces. Historical reports do not contain enough
data to reconstruct timestep hotspots; those views deliberately show an empty state.
Timeline indices refer to **evaluation rollout steps**, not training timesteps.
The priority heuristic is `0.75 * (1 - exp(-max(TD residual / residual std, 0) / 2))`
plus `0.25 * action saturation fraction`. It is separate from the aggregate score.
The final nonterminal sample is unscored because its next value is unavailable.
Episode boundaries follow the existing report's zero-bootstrap convention.

Signals are not proof of reward hacking. Baseline TD3 critic checkpoints, PPO
normalization reconstruction and intrinsic/extrinsic target mismatch limit claims.
Use multiple seeds, matched settings and direct behavioral inspection for research.
This release supports BipedalWalker-v3 and the six bundled agents; arbitrary Gym
environments and custom reward-code editing are not yet exposed in the UI.

The server binds only to loopback. Mutations require a per-session token and local
origin/host checks. The UI uses local assets and no remote fonts or analytics.

## Verification

```powershell
../.venv/Scripts/python.exe -B -m unittest test_dashboard test_reward_wrapper -v
```

Tests use temporary workspaces and mocked training subprocesses. No agents are
trained by these tests. HTTP tests validate report/static serving and request
protection; timeline tests validate ranking and episode boundaries.
