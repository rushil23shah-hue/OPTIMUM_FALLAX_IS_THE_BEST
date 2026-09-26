# OPTIMUM_FALLAX_IS_THE_BEST
Reward hacking detection

## Local control dashboard

Open `final_agent_and_architechture/Open Fallax.cmd` on Windows, or run:

```powershell
.venv/Scripts/python.exe -B final_agent_and_architechture/dashboard.py
```

The local interface displays real reports, per-agent diagnostics, future evaluation
timelines, reward comparisons, and controls for isolated training and report jobs.
See [dashboard setup and controls](final_agent_and_architechture/DASHBOARD.md).

The UI ships in the Python package. Install from this directory with
`python -m pip install .`, then launch `fallax`. Use `.[training]` to include the
training dependencies. No package has been published to a public registry yet.
