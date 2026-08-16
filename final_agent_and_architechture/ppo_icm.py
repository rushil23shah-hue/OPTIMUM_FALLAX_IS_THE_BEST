import csv
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. OBSERVATION NORMALIZER

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def normalize_obs(obs, obs_rms):
    obs_rms.update(np.array([obs]))
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)


# 2. ACTOR-CRITIC POLICY NETWORK

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.actor_fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        hidden = self.actor_fc(x)
        mean = self.actor_mean(hidden)
        log_std = self.actor_logstd.expand_as(mean)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(x)
        
        return action, log_prob, entropy, value


# 3. INTRINSIC CURIOSITY MODULE (ICM)

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64):
        super(IntrinsicCuriosityModule, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state, next_state, action):
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        
        pred_action = self.inverse_net(torch.cat([phi_s, phi_s_next], dim=-1))
        pred_phi_s_next = self.forward_net(torch.cat([phi_s, action], dim=-1))
        
        return phi_s_next, pred_phi_s_next, pred_action


# 4. PPO + ICM TRAINING LOOP

def train_ppo_icm(
    env_id="BipedalWalker-v3", 
    total_steps=1_000_000, 
    n_steps=2048,
    n_epochs=10, 
    batch_size=64, 
    lr=3e-4, 
    icm_lr=1e-3,
    gamma=0.99, 
    gae_lambda=0.95,
    clip_range=0.2, 
    ent_coef=0.01, 
    vf_coef=0.5, 
    max_grad_norm=0.5,
    icm_scale=0.01,
    beta=0.2,         
    seed=0, 
    device="cpu"
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    env = gym.wrappers.ClipAction(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    net = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    icm = IntrinsicCuriosityModule(obs_dim, action_dim, feature_dim=64).to(device)
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=icm_lr)

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    raw_obs, _ = env.reset(seed=seed)
    obs = normalize_obs(raw_obs, obs_rms)

    num_updates = total_steps // n_steps
    global_step = 0

    # Logging Trackers
    update_history = []
    reward_history = []

    for update in range(1, num_updates + 1):
        obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        next_obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, action_dim), dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)
        rew_ext_buf = np.zeros(n_steps, dtype=np.float32)
        rew_int_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)

        for step in range(n_steps):
            global_step += 1
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, val = net.get_action_and_value(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            next_raw_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            next_obs = normalize_obs(next_raw_obs, obs_rms)

            obs_buf[step] = obs
            next_obs_buf[step] = next_obs
            act_buf[step] = action_np
            logp_buf[step] = log_prob.item()
            rew_ext_buf[step] = reward
            val_buf[step] = val.item()
            done_buf[step] = float(terminated) 

            s_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            s_next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                phi_s_next, pred_phi_s_next, _ = icm(s_t, s_next_t, a_t)
                intrinsic_reward = 0.5 * F.mse_loss(pred_phi_s_next, phi_s_next, reduction='none').sum(dim=-1)
                rew_int_buf[step] = intrinsic_reward.item()

            obs = next_obs

            if done:
                raw_obs, _ = env.reset()
                obs = normalize_obs(raw_obs, obs_rms)

        rew_int_normalized = rew_int_buf / (rew_int_buf.std() + 1e-8)
        scaled_int_rew = icm_scale * rew_int_normalized
        total_rewards = rew_ext_buf + scaled_int_rew

        # Track rollout reward for graph logging
        update_history.append(update)
        reward_history.append(float(rew_ext_buf.sum()))

        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0
        
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            next_value = net.get_value(last_obs_tensor).item()

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = val_buf[t + 1]

            delta = total_rewards[t] + gamma * next_val * next_non_terminal - val_buf[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + val_buf

        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        b_next_obs = torch.as_tensor(next_obs_buf, dtype=torch.float32, device=device)
        b_act = torch.as_tensor(act_buf, dtype=torch.float32, device=device)
        b_logp = torch.as_tensor(logp_buf, dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(returns, dtype=torch.float32, device=device)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        inds = np.arange(n_steps)
        for epoch in range(n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]

                _, new_logp, entropy, new_val = net.get_action_and_value(b_obs[mb_inds], b_act[mb_inds])
                log_ratio = new_logp - b_logp[mb_inds]
                ratio = log_ratio.exp()

                mb_adv = b_adv[mb_inds]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * F.mse_loss(new_val.squeeze(-1), b_ret[mb_inds])
                entropy_loss = entropy.mean()

                ppo_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                ppo_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

                phi_s_next, pred_phi_s_next, pred_action = icm(
                    b_obs[mb_inds], b_next_obs[mb_inds], b_act[mb_inds]
                )
                
                inverse_loss = F.mse_loss(pred_action, b_act[mb_inds])
                forward_loss = 0.5 * F.mse_loss(pred_phi_s_next, phi_s_next)
                icm_loss = (1 - beta) * inverse_loss + beta * forward_loss

                icm_optimizer.zero_grad()
                icm_loss.backward()
                icm_optimizer.step()

        print(f"Update {update}/{num_updates} | Total Steps: {global_step} | Mean Extrinsic: {rew_ext_buf.sum():.2f} | Scaled Intrinsic Added: {scaled_int_rew.sum():.2f}")

    # --- SAVE CSV & PLOT GRAPH ---
    with open("ppo_icm_rewards.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Update", "Reward"])
        for u, r in zip(update_history, reward_history):
            writer.writerow([u, r])

    plt.figure(figsize=(10, 5))
    plt.plot(update_history, reward_history, label="Extrinsic Reward", color="blue", alpha=0.6)
    if len(reward_history) >= 5:
        ma = np.convolve(reward_history, np.ones(5)/5, mode='valid')
        plt.plot(update_history[4:], ma, label="5-Update Moving Avg", color="red", linewidth=2)
    plt.xlabel("Updates")
    plt.ylabel("Rollout Extrinsic Reward")
    plt.title("PPO + ICM Training Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("ppo_icm_curve.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    train_ppo_icm(
        env_id="BipedalWalker-v3",
        total_steps=1_000_000,
        icm_scale=0.01
    )