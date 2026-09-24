"""
run_hackability_report.py -- run this to get a hackability score.

End-to-end pipeline: collect a rollout -> compute metrics.py's diagnostics
-> run adversarial.py's noise sweep -> combine into hackability.py's score
-> print a per-agent report and a leaderboard.

Wired for all six agents in the suite. Each has a different save/load
convention and, for the two PPO-family model-based agents, a different
observation-normalization convention (see each build_* function's comment
for the specific caveat) -- all confirmed against an actual checkpoint
produced by training each agent, not guessed from source alone.
"""

import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from metrics import (
    Trajectory,
    td_error_anomaly,
    return_calibration_gap,
    reward_concentration,
    reward_rate,
    termination_timing,
    action_saturation_entropy,
    critic_sensitivity,
)
from adversarial import adversarial_robustness
from hackability import compute_hackability_score, cross_agent_action_distance, WEIGHTS

from ppo_simple import ActorCritic as PPOActorCritic, make_env as ppo_make_env
from td3FINAL import TD3Agent

ENV_ID = "BipedalWalker-v3"
N_STEPS = 2000          # steps for the clean diagnostic rollout
ADV_N_STEPS = 1500      # steps per rollout in the adversarial sweep
ADV_N_REPEATS = 2
ADV_SIGMAS = (0.0, 0.02, 0.05, 0.1, 0.2)
RUN_DIR = "runs"


def collect_trajectory(env, act_fn, n_steps: int) -> Trajectory:
    obs_list, act_list, rew_list, term_list, trunc_list = [], [], [], [], []
    obs, _ = env.reset(seed=0)
    for _ in range(n_steps):
        action = act_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        obs_list.append(obs)
        act_list.append(action)
        rew_list.append(reward)
        term_list.append(terminated)
        trunc_list.append(truncated)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    return Trajectory(
        obs=np.array(obs_list),
        actions=np.array(act_list),
        rewards=np.array(rew_list),
        terminated=np.array(term_list),
        truncated=np.array(trunc_list),
    )


# --- PPO adapter -------------------------------------------------------

def build_ppo():
    env = ppo_make_env(ENV_ID, seed=0, idx=0)()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    net = PPOActorCritic(obs_dim, action_dim, hidden=256)

    ckpt = "ppo_bipedalwalker.pth"
    trained = os.path.exists(ckpt)
    if trained:
        net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    net.eval()

    def act_fn(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = net.get_action_and_value(obs_t)
        return action.squeeze(0).numpy()

    def critic_fn(obs_batch, action_batch=None):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            v = net.get_value(obs_t)
        return v.cpu().numpy()

    return env, act_fn, critic_fn, trained


# --- TD3 adapter --------------------------------------------------------

def build_td3():
    env = gym.make(ENV_ID)
    agent = TD3Agent(env.observation_space.shape[0], env.action_space, device="cpu")

    ckpt = "td3_best.pt"
    trained = os.path.exists(ckpt)
    if trained:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        agent.actor.load_state_dict(state["actor"])

    def act_fn(obs):
        return agent.act(obs, deterministic=True)

    def critic_fn(obs_batch, action_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32)
        with torch.no_grad():
            q = agent.q1(obs_t, act_t)
        return q.squeeze(-1).cpu().numpy()

    return env, act_fn, critic_fn, trained


# --- SAC adapter ---------------------------------------------------------

def build_sac():
    from SAC import SAC_countinuous
    from sac_utilis import Action_adapter

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC_countinuous(
        state_dim=obs_dim, action_dim=action_dim, gamma=0.99, net_width=256,
        a_lr=3e-4, c_lr=3e-4, batch_size=256, alpha=0.12, adaptive_alpha=True, dvc="cpu",
    )

    # sac_main.py saves one checkpoint per `save_interval` steps as
    # model/BWv3_<step-in-thousands>k_actor.pt -- load whichever is furthest
    # along.
    actor_ckpts = sorted(
        Path("model").glob("BWv3_*k_actor.pt"),
        key=lambda p: int(p.stem.split("_")[1].rstrip("k")),
    ) if Path("model").exists() else []
    trained = len(actor_ckpts) > 0
    if trained:
        step = int(actor_ckpts[-1].stem.split("_")[1].rstrip("k"))
        agent.load("BWv3", step)

    def act_fn(obs):
        a = agent.select_action(obs, deterministic=True)  # a in [-1, 1]
        return Action_adapter(a, max_action)

    def critic_fn(obs_batch, action_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32)
        with torch.no_grad():
            q1, q2 = agent.critic(obs_t, act_t)
        return torch.min(q1, q2).squeeze(-1).cpu().numpy()

    return env, act_fn, critic_fn, trained


# --- TD3-RND adapter -------------------------------------------------------

def build_td3_rnd():
    from td3_rndFINAL import TD3RNDAgent

    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    agent = TD3RNDAgent(state_dim=obs_dim, n_actions=n_actions, max_action=max_action, min_action=min_action)

    trained = os.path.exists("tmp/td3_rnd/actor_td3_rnd.pt")
    if trained:
        agent.load_models()

    # NOT agent.choose_action(): it always adds exploration noise, and for
    # the first `warmup` calls (time_step starts at 0 on every fresh
    # TD3RNDAgent -- it isn't part of the saved checkpoint) it ignores the
    # actor network entirely and returns uniform-random actions instead.
    # Call the actor directly for a clean deterministic eval action.
    min_t = torch.as_tensor(min_action, dtype=torch.float32)
    max_t = torch.as_tensor(max_action, dtype=torch.float32)

    def act_fn(obs):
        state = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = agent.actor(state)[0]
        return torch.clamp(action, min_t, max_t).cpu().numpy()

    def critic_fn(obs_batch, action_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32)
        with torch.no_grad():
            q = agent.critic_1(obs_t, act_t)
        return q.squeeze(-1).cpu().numpy()

    return env, act_fn, critic_fn, trained


# --- PPO-ICM / model-based shared adapter -----------------------------
# Both use the same ActorCritic shape and the same outside-the-env
# RunningMeanStd normalization (exploration.normalize_obs) applied manually
# in their training loops, unlike ppo_simple's gym.wrappers.NormalizeObservation.
# That running-stats object is never checkpointed, so this rebuilds it from
# scratch and lets it adapt online during the eval rollout -- same
# cold-start caveat noted for PPO in build_ppo above.

def _build_normalized_actor_critic(actor_critic_cls, checkpoint_path):
    from exploration import RunningMeanStd, normalize_obs

    env = gym.make(ENV_ID)
    env = gym.wrappers.ClipAction(env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    net = actor_critic_cls(obs_dim, action_dim)

    trained = os.path.exists(checkpoint_path)
    if trained:
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    net.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    def act_fn(obs):
        normed = normalize_obs(obs, obs_rms)  # updates obs_rms, matching training
        obs_t = torch.as_tensor(normed, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = net.get_action_and_value(obs_t)
        return action.squeeze(0).numpy()

    def critic_fn(obs_batch, action_batch=None):
        obs_batch = np.atleast_2d(obs_batch)
        normed = (obs_batch - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)  # read-only
        obs_t = torch.as_tensor(normed, dtype=torch.float32)
        with torch.no_grad():
            v = net.get_value(obs_t)
        return v.cpu().numpy()

    return env, act_fn, critic_fn, trained


def build_ppo_icm():
    from ppo_icm import ActorCritic as PPOICMActorCritic
    return _build_normalized_actor_critic(PPOICMActorCritic, "ppo_icm.pth")


def build_model_based():
    from models import ActorCritic as MBActorCritic
    return _build_normalized_actor_critic(MBActorCritic, "model_based_policy.pth")


AGENTS = {
    "ppo": build_ppo,
    "td3": build_td3,
    "sac": build_sac,
    "td3_rnd": build_td3_rnd,
    "ppo_icm": build_ppo_icm,
    "model_based": build_model_based,
}


def run_agent(name: str, build_fn):
    env, act_fn, critic_fn, trained = build_fn()

    traj = collect_trajectory(env, act_fn, N_STEPS)
    obs_std = np.std(traj.obs, axis=0)
    obs_std[obs_std < 1e-6] = 1e-6  # guard against a constant/degenerate obs dim

    td_err = td_error_anomaly(traj, critic_fn)
    gap = return_calibration_gap(traj, critic_fn)
    conc = reward_concentration(traj, critic_fn)
    rr = reward_rate(traj)
    tt = termination_timing(traj, max_episode_steps=env.spec.max_episode_steps)
    sat = action_saturation_entropy(traj)
    sens = critic_sensitivity(traj, critic_fn)

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

    return {
        "name": name,
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
        },
    }, act_fn


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    results = {}
    act_fns = {}

    for name, build_fn in AGENTS.items():
        print(f"\n=== {name} ===")
        result, act_fn = run_agent(name, build_fn)
        results[name] = result
        act_fns[name] = act_fn

        tag = "" if result["trained_checkpoint_found"] else "  [UNTRAINED -- smoke test only, not a real verdict]"
        print(f"hackability_score: {result['hackability_score']:.3f}  ({result['verdict']}){tag}")
        for k, v in result["sub_scores"].items():
            print(f"  {k:20s} {v:.3f}  (weight {WEIGHTS[k]:.2f})")
        print("  context:", result["context"])
        print("  adversarial retention by sigma:", result["adversarial"]["retention"])

    print("\n=== Leaderboard (higher = more suspicious) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["hackability_score"]):
        print(f"  {name:12s} {r['hackability_score']:.3f}  {r['verdict']}")

    # Cross-agent agreement: only meaningful once >=2 agents are ACTUALLY
    # trained (comparing an untrained network to a trained one is noise).
    trained_names = [n for n in results if results[n]["trained_checkpoint_found"]]
    if len(trained_names) >= 2:
        eval_env = gym.make(ENV_ID)
        eval_states = []
        obs, _ = eval_env.reset(seed=123)
        eval_states.append(obs)
        for _ in range(19):
            obs, *_ = eval_env.step(eval_env.action_space.sample())
            eval_states.append(obs)
        eval_env.close()

        distances = cross_agent_action_distance(
            {n: act_fns[n] for n in trained_names}, eval_states
        )
        print("\n=== Cross-agent action distance (trained agents only) ===")
        print(distances)
        results["_cross_agent_action_distance"] = distances
    else:
        print("\nCross-agent action distance skipped: fewer than 2 TRAINED agents on disk.")

    with open(os.path.join(RUN_DIR, "hackability_report.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved runs/hackability_report.json")


if __name__ == "__main__":
    main()
