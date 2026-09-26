"""Rebuild the original baseline hackability report (runs/hackability_report.json) using original rewards."""
import json
import os
from pathlib import Path
import numpy as np
import torch

os.environ["WALKER_REWARD_PROFILE"] = "original"
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from run_hackability_report import (
    AGENTS, N_STEPS, ADV_SIGMAS, ADV_N_STEPS, ADV_N_REPEATS,
    collect_trajectory, td_error_anomaly, return_calibration_gap,
    reward_concentration, reward_rate, termination_timing,
    action_saturation_entropy, critic_sensitivity, adversarial_robustness,
    compute_hackability_score, cross_agent_action_distance, make_walker_env, ENV_ID, WEIGHTS
)
from fallax_toolkit.timeline import build_timeline

np.random.seed(0)
torch.manual_seed(0)

results = {}
act_fns = {}

for name, build_fn in AGENTS.items():
    print(f"=== Baseline {name} (original reward) ===", flush=True)
    try:
        env, act_fn, critic_fn, trained = build_fn()
    except Exception as exc:
        print(f"Skipping {name}: {exc}", flush=True)
        continue

    if not trained:
        env.close()
        print(f"Skipping {name} (untrained)", flush=True)
        continue

    traj = collect_trajectory(env, act_fn, N_STEPS)
    obs_std = np.std(traj.obs, axis=0)
    obs_std[obs_std < 1e-6] = 1e-6

    td_err = td_error_anomaly(traj, critic_fn)
    gap = return_calibration_gap(traj, critic_fn)
    conc = reward_concentration(traj, critic_fn)
    rr = reward_rate(traj)
    tt = termination_timing(traj, max_episode_steps=env.spec.max_episode_steps)
    sat = action_saturation_entropy(traj)
    sens = critic_sensitivity(traj, critic_fn)
    timeline = build_timeline(traj, critic_fn)

    def collect_fn(noisy_act_fn, n_steps):
        return collect_trajectory(env, noisy_act_fn, n_steps)

    adv = adversarial_robustness(
        act_fn, collect_fn, obs_std,
        sigmas=ADV_SIGMAS, n_steps=ADV_N_STEPS, n_repeats=ADV_N_REPEATS,
    )

    report = compute_hackability_score(
        td_error=td_err,
        calibration_gap=gap,
        concentration=conc,
        saturation=sat,
        sensitivity=sens,
        adversarial_robustness_score=adv.robustness_score,
        reward_rate=rr,
        termination_timing=tt,
    )
    env.close()

    results[name] = {
        "name": name,
        "reward_profile": "original",
        "reward_audit": traj.reward_audit,
        "timeline": timeline,
        "trained_checkpoint_found": trained,
        "hackability_score": report.score,
        "verdict": report.verdict,
        "sub_scores": report.sub_scores,
        "context": report.context,
        "adversarial": {
            "mean_returns": adv.mean_returns,
            "retention": adv.retention,
            "worst_case_retention": adv.worst_case_retention,
            "cliff_drop": adv.cliff_drop,
        },
        "raw_metrics": {
            "td_error_anomaly": td_err,
            "return_calibration_gap": gap,
            "reward_concentration": conc,
            "action_saturation_entropy": sat,
            "critic_sensitivity": sens,
        }
    }
    act_fns[name] = act_fn
    print(f"  {name}: score {report.score:.4f} ({report.verdict})", flush=True)

results["_experiment"] = {"profile": "original", "environment": "BipedalWalker-v3"}
results["_limitations"] = [
    "Original baseline uses the un-wrapped native BipedalWalker-v3 environment.",
    "PPO observation/reward normalization statistics are rebuilt during evaluation.",
    "Curiosity critics include intrinsic rewards; diagnostics use extrinsic rewards.",
]

out_path = ROOT / "runs" / "hackability_report.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print(f"Saved {out_path} for original baseline!", flush=True)
