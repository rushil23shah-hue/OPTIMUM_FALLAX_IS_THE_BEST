import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
from reward_wrapper import make_walker_env, reward_profile


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_log_std.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


def make_env(env_id, seed, idx):
    def thunk():
        env = make_walker_env(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env.action_space.seed(seed + idx)
        return env

    return thunk


def _extract_final_obs(infos, num_envs, obs_dim):
    """Gymnasium vector envs auto-reset on episode end, so `next_obs` from
    `envs.step()` is already the *reset* observation for any env that just
    finished. The true pre-reset observation (needed to correctly bootstrap
    a truncated -- not terminated -- episode) is stashed in `infos` under
    'final_observation' (older API) or 'final_obs' (newer API), alongside a
    boolean mask under the '_' prefixed key. Returns (mask, obs_array).
    """
    for obs_key, mask_key in (("final_observation", "_final_observation"),
                               ("final_obs", "_final_obs")):
        if obs_key in infos:
            mask = np.asarray(infos[mask_key], dtype=bool)
            obs_arr = infos[obs_key]
            return mask, obs_arr
    return np.zeros(num_envs, dtype=bool), None


def train_ppo(env_id="BipedalWalker-v3", total_steps=2_500_000, num_envs=16, n_steps=2048,
              n_epochs=10, minibatch_size=512, lr=3e-4, gamma=0.99, gae_lambda=0.95,
              clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
              seed=1, device="cpu", resume_checkpoint=True):

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_size = num_envs * n_steps
    num_updates = total_steps // batch_size

    # Create parallel environments
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, i) for i in range(num_envs)])
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    net = ActorCritic(obs_dim, action_dim, hidden=256).to(device)

    # Resume weights if checkpoint exists
    checkpoint_path = "ppo_bipedalwalker.pth"
    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming weights from {checkpoint_path}...", flush=True)
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    # Logging Trackers
    update_history = []
    reward_history = []
    global_step = 0
    start_time = time.time()

    # Rollout Storage Tensors
    obs_b = torch.zeros((n_steps, num_envs, obs_dim), device=device)
    actions_b = torch.zeros((n_steps, num_envs, action_dim), device=device)
    logprobs_b = torch.zeros((n_steps, num_envs), device=device)
    rewards_b = torch.zeros((n_steps, num_envs), device=device)
    values_b = torch.zeros((n_steps, num_envs), device=device)

    # Terminated / truncated are tracked SEPARATELY (not folded into one
    # "done" flag) so a time-limit cutoff never gets treated like a real
    # fall. `bootstrap_value_b` holds gamma-discounted V(true next state)
    # for steps that were truncated-but-not-terminated, added into the
    # reward before GAE so the value estimate isn't corrupted at 1600-step
    # episode boundaries -- exactly where BipedalWalker performance used to
    # plateau/degrade once the policy got good enough to survive that long.
    terminated_b = torch.zeros((n_steps, num_envs), device=device)
    truncated_b = torch.zeros((n_steps, num_envs), device=device)
    bootstrap_value_b = torch.zeros((n_steps, num_envs), device=device)

    # Reset environment
    next_obs_np, _ = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_obs = torch.clamp(next_obs, -10.0, 10.0)
    next_done = torch.zeros(num_envs, device=device)

    recent_ep_returns = []

    print(f"Starting PPO training on {env_id} for {total_steps} timesteps ({num_updates} updates)...", flush=True)

    for update in range(1, num_updates + 1):
        # Linear Learning Rate Annealing
        frac = 1.0 - (update - 1.0) / num_updates
        lr_now = frac * lr
        optimizer.param_groups[0]["lr"] = lr_now

        for step in range(n_steps):
            global_step += num_envs
            obs_b[step] = next_obs
            # dones_b removed: replaced by terminated_b/truncated_b below

            with torch.no_grad():
                action, logprob, _, value = net.get_action_and_value(next_obs)
                values_b[step] = value

            actions_b[step] = action
            logprobs_b[step] = logprob

            # Step environment
            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
            reward_np = np.clip(reward_np, -10.0, 10.0)
            rewards_b[step] = torch.tensor(reward_np, device=device)
            terminated_b[step] = torch.tensor(term_np, dtype=torch.float32, device=device)
            truncated_b[step] = torch.tensor(trunc_np, dtype=torch.float32, device=device)

            # For envs truncated-but-not-terminated this step, fetch the
            # TRUE pre-reset next observation and bootstrap V(s') from it.
            trunc_only = np.logical_and(trunc_np, np.logical_not(term_np))
            if trunc_only.any():
                mask, final_obs_arr = _extract_final_obs(infos, num_envs, obs_dim)
                if final_obs_arr is not None:
                    idxs = np.where(trunc_only & mask)[0]
                    for i in idxs:
                        fo = final_obs_arr[i]
                        if fo is not None:
                            fo_t = torch.tensor(fo, dtype=torch.float32, device=device)
                            fo_t = torch.clamp(fo_t, -10.0, 10.0).unsqueeze(0)
                            with torch.no_grad():
                                bootstrap_value_b[step, i] = net.get_value(fo_t)

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_obs = torch.clamp(next_obs, -10.0, 10.0)
            next_done = torch.tensor(np.logical_or(term_np, trunc_np), dtype=torch.float32, device=device)

            # Record raw episode statistics
            if "episode" in infos:
                for r, is_done in zip(infos["episode"]["r"], infos["_episode"]):
                    if is_done:
                        recent_ep_returns.append(r)

        # Fold the truncation bootstrap into the reward stream, then use a
        # single "chain-cutting" mask (terminated OR truncated) for GAE --
        # cutting the advantage recursion at any episode boundary is correct
        # either way; the fix is that truncated boundaries now carry the
        # true continuation value instead of silently discarding it.
        effective_rewards_b = rewards_b + gamma * bootstrap_value_b
        effective_dones_b = torch.clamp(terminated_b + truncated_b, max=1.0)

        # Value estimation for last state (bootstrapping)
        with torch.no_grad():
            next_value = net.get_value(next_obs)
            advantages = torch.zeros_like(rewards_b, device=device)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - effective_dones_b[t + 1]
                    nextvalues = values_b[t + 1]
                delta = effective_rewards_b[t] + gamma * nextvalues * nextnonterminal - values_b[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns_b = advantages + values_b

        # Flatten batch
        b_obs = obs_b.reshape((-1, obs_dim))
        b_logprobs = logprobs_b.reshape(-1)
        b_actions = actions_b.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns_b.reshape(-1)
        b_values = values_b.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = net.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # Advantage Normalization per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (with clipping)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_range,
                    clip_range,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                vf_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()

        mean_ep_return = float(np.mean(recent_ep_returns[-20:])) if len(recent_ep_returns) > 0 else 0.0
        update_history.append(update)
        reward_history.append(mean_ep_return)

        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update:3d}/{num_updates} | Step {global_step:8d}/{total_steps} | "
              f"Mean Ep Return: {mean_ep_return:7.2f} | SPS: {sps}", flush=True)

        # Periodically checkpoint weights
        if update % 10 == 0 or update == num_updates:
            torch.save(net.state_dict(), checkpoint_path)

        # Reset per-rollout terminated/truncated/bootstrap buffers
        terminated_b.zero_()
        truncated_b.zero_()
        bootstrap_value_b.zero_()

    envs.close()

    # --- SAVE CSV & PLOT GRAPH ---
    with open("ppo_simple_rewards.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Update", "Reward"])
        for u, r in zip(update_history, reward_history):
            writer.writerow([u, r])

    plt.figure(figsize=(10, 5))
    plt.plot(update_history, reward_history, label="Mean Episode Return", color="orange", alpha=0.6)
    if len(reward_history) >= 5:
        ma = np.convolve(reward_history, np.ones(5) / 5, mode="valid")
        plt.plot(update_history[4:], ma, label="5-Update Moving Avg", color="red", linewidth=2)
    plt.axhline(y=300, color="green", linestyle="--", label="Target Reward (300)")
    plt.xlabel("Updates")
    plt.ylabel("Episode Return")
    plt.title("PPO Training Curve on BipedalWalker-v3")
    plt.grid(True)
    plt.legend()
    plt.savefig("ppo_simple_curve.png", dpi=300)
    plt.close()

    print(f"Model saved to {checkpoint_path}", flush=True)
    return net


if __name__ == "__main__":
    train_ppo(env_id="BipedalWalker-v3", total_steps=2_500_000, resume_checkpoint=True)