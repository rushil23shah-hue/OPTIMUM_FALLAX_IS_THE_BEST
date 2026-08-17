import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance -- lets us
    normalize observations using stats updated one at a time as training
    progresses, no need to store full history."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        delta = x - self.mean
        tot_count = self.count + 1
        new_mean = self.mean + delta / tot_count
        m_a = self.var * self.count
        m2 = m_a + np.square(delta) * self.count / tot_count
        self.mean, self.var, self.count = new_mean, m2 / tot_count, tot_count

    def normalize(self, x, clip=10.0, epsilon=1e-8):
        norm = (x - self.mean) / np.sqrt(self.var + epsilon)
        return np.clip(norm, -clip, clip).astype(np.float32)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


def train_ppo(env_id="Pendulum-v1", total_steps=200_000, n_steps=2048,
              n_epochs=10, batch_size=64, lr=3e-4, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              eval_every=10_000, seed=0, device="cpu"):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    net = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # Logging Trackers
    update_count = 0
    update_history = []
    reward_history = []

    def normalize(o):
        obs_rms.update(o)
        return obs_rms.normalize(o)

    def evaluate(n_episodes=5):
        returns = []
        for _ in range(n_episodes):
            o, _ = env.reset()
            o = obs_rms.normalize(o)
            ep_ret = 0.0
            for _ in range(1000):
                with torch.no_grad():
                    o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    mean = net.actor_mean(o_t)
                    action = torch.clamp(mean, action_low, action_high)
                o, r, term, trunc, _ = env.step(action.squeeze(0).numpy())
                o = obs_rms.normalize(o)
                ep_ret += r
                if term or trunc:
                    break
            returns.append(ep_ret)
        return float(np.mean(returns))

    obs, _ = env.reset(seed=seed)
    obs = normalize(obs)
    total_env_steps = 0
    next_eval = eval_every

    while total_env_steps < total_steps:
        # --- collect one rollout of n_steps ---
        buf_obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        buf_actions = np.zeros((n_steps, action_dim), dtype=np.float32)
        buf_log_probs = np.zeros(n_steps, dtype=np.float32)
        buf_values = np.zeros(n_steps, dtype=np.float32)
        buf_rewards = np.zeros(n_steps, dtype=np.float32)
        buf_terminated = np.zeros(n_steps, dtype=np.float32)
        buf_truncated = np.zeros(n_steps, dtype=np.float32)
        buf_bootstrap_value = np.zeros(n_steps, dtype=np.float32)

        for t in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = net.get_action_and_value(obs_t)
            action_np = action.squeeze(0).numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            next_obs = normalize(next_obs)

            buf_obs[t] = obs
            buf_actions[t] = action_np
            buf_log_probs[t] = log_prob.item()
            buf_values[t] = value.item()
            buf_rewards[t] = reward
            buf_terminated[t] = float(terminated)
            buf_truncated[t] = float(truncated)

            if truncated and not terminated:
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    buf_bootstrap_value[t] = net.get_value(next_obs_t).item()

            total_env_steps += 1

            if terminated or truncated:
                obs, _ = env.reset()
                obs = normalize(obs)
            else:
                obs = next_obs

            if total_env_steps >= next_eval:
                mean_return = evaluate()
                print(f"  step {total_env_steps:>8} | eval_mean_return = {mean_return:8.2f}")
                next_eval += eval_every

            if total_env_steps >= total_steps:
                break

        n_collected = t + 1

        # Track rollout reward for graph logging
        update_count += 1
        update_history.append(update_count)
        reward_history.append(float(buf_rewards[:n_collected].sum()))

        effective_rewards = buf_rewards[:n_collected].copy()
        effective_done = buf_terminated[:n_collected].copy()
        trunc_mask = buf_truncated[:n_collected].astype(bool)
        effective_rewards[trunc_mask] += gamma * buf_bootstrap_value[:n_collected][trunc_mask]
        effective_done[trunc_mask] = 1.0

        # --- GAE ---
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = net.get_value(last_obs_t).item()

        advantages = np.zeros(n_collected, dtype=np.float32)
        last_adv = 0.0
        values_ext = np.append(buf_values[:n_collected], last_value)
        for t in reversed(range(n_collected)):
            mask = 1.0 - effective_done[t]
            delta = effective_rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[t] = last_adv
        returns = advantages + buf_values[:n_collected]

        # --- PPO update ---
        obs_t = torch.as_tensor(buf_obs[:n_collected], dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(buf_actions[:n_collected], dtype=torch.float32, device=device)
        old_log_probs_t = torch.as_tensor(buf_log_probs[:n_collected], dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        indices = np.arange(n_collected)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_collected, batch_size):
                mb = indices[start:start + batch_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=device)

                _, new_log_prob, entropy, value = net.get_action_and_value(obs_t[mb_t], actions_t[mb_t])
                ratio = torch.exp(new_log_prob - old_log_probs_t[mb_t])

                mb_adv = advantages_t[mb_t]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = ((value - returns_t[mb_t]) ** 2).mean()
                loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

    # --- SAVE CSV & PLOT GRAPH ---
    with open("ppo_simple_rewards.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Update", "Reward"])
        for u, r in zip(update_history, reward_history):
            writer.writerow([u, r])

    plt.figure(figsize=(10, 5))
    plt.plot(update_history, reward_history, label="Extrinsic Reward", color="orange", alpha=0.6)
    if len(reward_history) >= 5:
        ma = np.convolve(reward_history, np.ones(5)/5, mode='valid')
        plt.plot(update_history[4:], ma, label="5-Update Moving Avg", color="red", linewidth=2)
    plt.xlabel("Updates")
    plt.ylabel("Rollout Reward")
    plt.title("Standard PPO Training Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("ppo_simple_curve.png", dpi=300)
    plt.close()

    env.close()
    return net


if __name__ == "__main__":
    train_ppo(env_id="BipedalWalker-v3", total_steps=2_000_000, eval_every=20_000)