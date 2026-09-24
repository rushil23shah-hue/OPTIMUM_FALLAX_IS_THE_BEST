"""
metrics_example.py -- proves metrics.py's interface generalizes across two
structurally different agents already in this repo:

  * ppo_simple.ActorCritic  -- on-policy, exposes V(s) via `get_value(obs)`
  * td3FINAL.TD3Agent       -- off-policy, exposes Q(s,a) via `agent.q1(obs, act)`

Neither agent file is modified. This script only wraps each agent's already
public critic method in the small `critic(obs_batch, action_batch=None)`
shape metrics.py expects, collects a short live rollout on Pendulum-v1 (a
lightweight built-in Gymnasium env -- any env works identically), and calls
all 3 metrics the same way regardless of which agent produced the data.

Networks are untrained (freshly initialized) -- this is only a smoke test
of the interface, not a claim about either agent's real behavior.
"""

import numpy as np
import torch
import gymnasium as gym

from metrics import Trajectory, td_error_anomaly, return_calibration_gap, reward_concentration
from ppo_simple import ActorCritic as PPOActorCritic
from td3FINAL import TD3Agent


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


def make_ppo_critic(net: PPOActorCritic):
    """Wraps ActorCritic.get_value (V(s), action-independent) as a CriticFn."""

    def critic(obs_batch, action_batch=None):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        with torch.no_grad():
            v = net.get_value(obs_t)
        return v.cpu().numpy()

    return critic


def make_td3_critic(agent: TD3Agent):
    """Wraps TD3Agent.q1 (Q(s,a), needs the action) as a CriticFn."""

    def critic(obs_batch, action_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32)
        act_t = torch.as_tensor(action_batch, dtype=torch.float32)
        with torch.no_grad():
            q = agent.q1(obs_t, act_t)
        return q.squeeze(-1).cpu().numpy()

    return critic


def run_metrics(name: str, traj: Trajectory, critic):
    print(f"\n=== {name} ===")
    print("td_error_anomaly:      ", td_error_anomaly(traj, critic))
    print("return_calibration_gap:", return_calibration_gap(traj, critic))
    print("reward_concentration:  ", reward_concentration(traj, critic))


def main():
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # --- On-policy, V(s): PPO ---
    ppo_net = PPOActorCritic(obs_dim, action_dim, hidden=64)

    def ppo_act(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = ppo_net.get_action_and_value(obs_t)
        return action.squeeze(0).numpy()

    ppo_traj = collect_trajectory(env, ppo_act, n_steps=200)
    run_metrics("PPO (V-critic)", ppo_traj, make_ppo_critic(ppo_net))

    # --- Off-policy, Q(s,a): TD3 ---
    td3_agent = TD3Agent(obs_dim, env.action_space, device="cpu")

    def td3_act(obs):
        return td3_agent.act(obs, deterministic=False)

    td3_traj = collect_trajectory(env, td3_act, n_steps=200)
    run_metrics("TD3 (Q-critic)", td3_traj, make_td3_critic(td3_agent))

    env.close()


if __name__ == "__main__":
    main()
