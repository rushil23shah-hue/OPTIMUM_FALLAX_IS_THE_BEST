import os
import csv
import matplotlib.pyplot as plt
import gymnasium as gym
from reward_wrapper import make_walker_env, reward_profile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ActorCritic, DynamicsModel
from exploration import RunningMeanStd, normalize_obs, IntrinsicCuriosityModule, AdversarialAuditor
from memory import VectorReplayBuffer

# --- PPO UPDATE ENGINE ---
def update_ppo(
    net, optimizer, b_obs, b_act, b_logp, b_adv, b_ret,
    clip_range=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
    n_epochs=10, batch_size=64
):
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
    n_steps = b_obs.shape[0]
    inds = np.arange(n_steps)

    for _ in range(n_epochs):
        np.random.shuffle(inds)
        for start in range(0, n_steps, batch_size):
            end = start + batch_size
            mb_inds = inds[start:end]

            _, new_logp, entropy, new_val = net.get_action_and_value(b_obs[mb_inds], b_act[mb_inds])
            ratio = (new_logp - b_logp[mb_inds]).exp()

            mb_adv = b_adv[mb_inds]
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * F.mse_loss(new_val.squeeze(-1), b_ret[mb_inds])
            entropy_loss = entropy.mean()

            ppo_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()


def train_dynamics(dynamics_model, optimizer, replay_buffer, batch_size=256, n_epochs=5, device="cpu"):
    if replay_buffer.size < batch_size:
        return 0.0

    total_loss = 0.0
    for _ in range(n_epochs):
        b_obs, b_act, b_rew, b_next_obs, b_done = replay_buffer.sample(batch_size, device=device)

        pred_next_obs, pred_rew, pred_done_logit = dynamics_model(b_obs, b_act)
        
        loss_obs = F.mse_loss(pred_next_obs, b_next_obs)
        loss_rew = F.mse_loss(pred_rew, b_rew)
        loss_done = F.binary_cross_entropy_with_logits(pred_done_logit, b_done)

        loss = loss_obs + loss_rew + loss_done

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dynamics_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / n_epochs


def compute_gae(rewards, dones, values, next_value, gamma=0.99, gae_lambda=0.95):
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = next_value
        else:
            next_non_terminal = 1.0 - float(dones[t])
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def generate_dream_rollouts(policy, dynamics, replay_buffer, horizon=10, num_seeds=64, gamma=0.99, gae_lambda=0.95, device="cpu"):
    """Generates short-horizon dream trajectories starting from real states sampled from ReplayBuffer."""
    if replay_buffer.size < num_seeds:
        return None

    init_states = replay_buffer.sample_states(num_seeds, device=device) # (num_seeds, obs_dim)
    curr_state = init_states

    dream_obs = []
    dream_act = []
    dream_logp = []
    dream_rew = []
    dream_val = []
    dream_done = []

    for h in range(horizon):
        with torch.no_grad():
            action, log_prob, _, val = policy.get_action_and_value(curr_state)
            pred_next_state, pred_reward, pred_done_logit = dynamics(curr_state, action)
            pred_done = (torch.sigmoid(pred_done_logit) > 0.5).float()

        dream_obs.append(curr_state)
        dream_act.append(action)
        dream_logp.append(log_prob)
        dream_rew.append(pred_reward.squeeze(-1))
        dream_val.append(val.squeeze(-1))
        dream_done.append(pred_done.squeeze(-1))

        curr_state = pred_next_state

    # Shape: (horizon, num_seeds, ...)
    obs_t = torch.stack(dream_obs, dim=0).reshape(-1, curr_state.shape[-1])
    act_t = torch.stack(dream_act, dim=0).reshape(-1, action.shape[-1])
    logp_t = torch.stack(dream_logp, dim=0).reshape(-1)
    rew_np = torch.stack(dream_rew, dim=0).cpu().numpy() # (horizon, num_seeds)
    val_np = torch.stack(dream_val, dim=0).cpu().numpy() # (horizon, num_seeds)
    done_np = torch.stack(dream_done, dim=0).cpu().numpy() # (horizon, num_seeds)

    with torch.no_grad():
        next_val_np = policy.get_value(curr_state).squeeze(-1).cpu().numpy()

    # Compute GAE across seed batch
    all_adv = []
    all_ret = []
    for s in range(num_seeds):
        adv, ret = compute_gae(rew_np[:, s], done_np[:, s], val_np[:, s], next_val_np[s], gamma, gae_lambda)
        all_adv.append(adv)
        all_ret.append(ret)

    adv_t = torch.as_tensor(np.array(all_adv).T, dtype=torch.float32, device=device).reshape(-1)
    ret_t = torch.as_tensor(np.array(all_ret).T, dtype=torch.float32, device=device).reshape(-1)
    mean_dream_return = float(rew_np.sum(axis=0).mean())

    return obs_t, act_t, logp_t, adv_t, ret_t, mean_dream_return


def main(checkpoint_path="model_based_policy.pth", resume_checkpoint=True, max_updates=1500, n_steps=2048):
    env_id = "BipedalWalker-v3"
    env = make_walker_env(env_id)
    env = gym.wrappers.ClipAction(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = "cpu"

    # Networks
    policy = ActorCritic(obs_dim, action_dim).to(device)
    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming weights from {checkpoint_path}...", flush=True)
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    icm = IntrinsicCuriosityModule(obs_dim, action_dim).to(device)
    icm_opt = torch.optim.Adam(icm.parameters(), lr=1e-3)

    dynamics = DynamicsModel(obs_dim, action_dim).to(device)
    dyn_opt = torch.optim.Adam(dynamics.parameters(), lr=1e-3)

    buffer = VectorReplayBuffer(obs_dim, action_dim, max_size=100_000)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    auditor = AdversarialAuditor(discrepancy_threshold=50.0)

    # Parameters
    icm_scale = 0.01

    raw_obs, _ = env.reset()
    obs = normalize_obs(raw_obs, obs_rms)

    update_history = []
    reward_history = []

    print("Starting Refactored Model-Based (Dyna-PPO) Training Pipeline on BipedalWalker-v3...")

    for update in range(1, max_updates + 1):
        obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        next_obs_buf = np.zeros((n_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((n_steps, action_dim), dtype=np.float32)
        logp_buf = np.zeros(n_steps, dtype=np.float32)
        rew_ext_buf = np.zeros(n_steps, dtype=np.float32)
        rew_int_buf = np.zeros(n_steps, dtype=np.float32)
        val_buf = np.zeros(n_steps, dtype=np.float32)
        done_buf = np.zeros(n_steps, dtype=np.float32)

        # 1. CONTINUOUS REAL ENVIRONMENT ROLLOUT
        for step in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, val = policy.get_action_and_value(obs_t)

            act_np = action.squeeze(0).numpy()
            next_raw_obs, reward, term, trunc, _ = env.step(act_np)
            done = term or trunc
            next_obs = normalize_obs(next_raw_obs, obs_rms)

            # ICM Intrinsic Reward
            s_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            ns_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            a_t = torch.as_tensor(act_np, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                phi_ns, pred_phi_ns, _ = icm(s_t, ns_t, a_t)
                int_reward = 0.5 * F.mse_loss(pred_phi_ns, phi_ns, reduction='none').sum(dim=-1).item()

            obs_buf[step] = obs
            next_obs_buf[step] = next_obs
            act_buf[step] = act_np
            logp_buf[step] = log_prob.item()
            rew_ext_buf[step] = reward
            rew_int_buf[step] = int_reward
            val_buf[step] = val.item()
            done_buf[step] = float(term)

            # Add to Replay Buffer for dynamics training
            buffer.add(obs, act_np, reward, next_obs, float(term))

            obs = next_obs
            if done:
                raw_obs, _ = env.reset()
                obs = normalize_obs(raw_obs, obs_rms)

        # Total Rewards & Intrinsic Normalization
        scaled_int = icm_scale * (rew_int_buf / (rew_int_buf.std() + 1e-8))
        total_rewards = rew_ext_buf + scaled_int

        # Compute GAE for Real Rollout
        with torch.no_grad():
            next_val = policy.get_value(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
        adv, ret = compute_gae(total_rewards, done_buf, val_buf, next_val)

        # Real Tensors
        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32)
        b_act = torch.as_tensor(act_buf, dtype=torch.float32)
        b_next_obs = torch.as_tensor(next_obs_buf, dtype=torch.float32)
        b_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
        b_adv = torch.as_tensor(adv, dtype=torch.float32)
        b_ret = torch.as_tensor(ret, dtype=torch.float32)

        # Update Auditor history
        real_total_return = float(rew_ext_buf.sum())
        auditor.update_real_average(real_total_return)
        update_history.append(update)
        reward_history.append(real_total_return)

        # 2. TRAIN DYNAMICS WORLD MODEL ON REPLAY BUFFER
        dyn_loss = train_dynamics(dynamics, dyn_opt, buffer, batch_size=256, n_epochs=5, device=device)

        # 3. SHORT-HORIZON DREAM ROLLOUTS
        dream_data = generate_dream_rollouts(policy, dynamics, buffer, horizon=5, num_seeds=32, device=device)
        
        if dream_data is not None:
            d_obs, d_act, d_logp, d_adv, d_ret, dream_ret_mean = dream_data
            
            if not auditor.is_hallucination(dream_ret_mean):
                # Concatenate Real + Audited Dream Rollouts for Policy Optimization
                b_obs = torch.cat([b_obs, d_obs], dim=0)
                b_act = torch.cat([b_act, d_act], dim=0)
                b_logp = torch.cat([b_logp, d_logp], dim=0)
                b_adv = torch.cat([b_adv, d_adv], dim=0)
                b_ret = torch.cat([b_ret, d_ret], dim=0)

        # 4. UPDATE PPO POLICY & ICM
        update_ppo(policy, policy_opt, b_obs, b_act, b_logp, b_adv, b_ret)

        phi_ns, pred_phi_ns, pred_act = icm(torch.as_tensor(obs_buf), torch.as_tensor(next_obs_buf), torch.as_tensor(act_buf))
        icm_loss = 0.8 * F.mse_loss(pred_act, torch.as_tensor(act_buf)) + 0.2 * 0.5 * F.mse_loss(pred_phi_ns, phi_ns)
        icm_opt.zero_grad()
        icm_loss.backward()
        icm_opt.step()

        print(f"Update {update}/{max_updates} | Real Rollout Return: {real_total_return:.2f} | Dyn Loss: {dyn_loss:.4f} | Replay Size: {buffer.size}", flush=True)

        if update % 10 == 0 or update == max_updates:
            torch.save(policy.state_dict(), checkpoint_path)

    # Save metrics & plot curve
    with open("ppo_mb_rewards.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Update", "Reward"])
        for u, r in zip(update_history, reward_history):
            writer.writerow([u, r])

    plt.figure(figsize=(10, 5))
    plt.plot(update_history, reward_history, label="Real Extrinsic Reward", color="blue", alpha=0.6)
    if len(reward_history) >= 5:
        ma = np.convolve(reward_history, np.ones(5)/5, mode='valid')
        plt.plot(update_history[4:], ma, label="5-Update Moving Avg", color="red", linewidth=2)
    plt.xlabel("Updates")
    plt.ylabel("Rollout Extrinsic Reward")
    plt.title("Model-Based PPO Training Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("ppo_mb_curve.png", dpi=300)
    plt.close()

    print(f"Training Complete! Saved ppo_mb_rewards.csv, ppo_mb_curve.png, and {checkpoint_path}.")
    return policy


if __name__ == "__main__":
    main()