"""
MBPO (Model-Based Policy Optimization) with SAC for BipedalWalker-v3.

Design notes (why it's structured this way):
- MBPO trains an ensemble of probabilistic dynamics models to predict state
  transitions (delta states) and extrinsic rewards.
- Short synthetic rollouts (k-step) are generated from the ensemble world model
  into a dedicated model replay buffer.
- States are normalized dynamically using RunningMeanStd to prevent feature
  scale disparities in BipedalWalker's 24-dim observation space.
- Synthetic rollouts check for BipedalWalker terminal conditions (falling over)
  to ensure SAC receives proper termination signals and penalties instead of
  exploiting hallucinated states.
- terminated vs truncated are tracked separately to handle bootstrapping correctly:
    * terminated (real failure, e.g. falling over) -> do not bootstrap Q-target.
    * truncated (time-limit cutoff) -> bootstrap Q-target normally.
"""

import math
import os
import random
#import time
from collections import deque
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def plot_learning_curve(x, scores, figure_file: str, window: int = 100) -> None:
    """Plot a running average of episode scores and save to disk."""
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - window) : (i + 1)])

    plt.figure()
    plt.plot(x, scores, alpha=0.3, label="Episode reward")
    plt.plot(x, running_avg, label=f"Running avg ({window})")
    plt.title("MBPO on BipedalWalker-v3")
    plt.xlabel("Episode")
    plt.ylabel("Extrinsic Reward")
    plt.legend()
    os.makedirs(os.path.dirname(figure_file) or ".", exist_ok=True)
    plt.savefig(figure_file)
    plt.close()


# ----------------------------------------------------------------------------
# 1. Running mean/std tracker
# ----------------------------------------------------------------------------


class RunningMeanStd:
    """Welford-style running mean/variance, updated incrementally batch by batch."""

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1 and self.mean.shape == ():
            batch_mean = x.mean()
            batch_var = x.var()
            batch_count = x.shape[0]
        else:
            if x.ndim == 1:
                x = x[None, :]
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


# ----------------------------------------------------------------------------
# 2. Replay Buffer (Environment & Synthetic Model Storage)
# ----------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, n_actions: int):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminated_memory = np.zeros(self.mem_size, dtype=bool)
        self.truncated_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, terminated, truncated) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminated_memory[index] = terminated
        self.truncated_memory[index] = truncated
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int, device: T.device):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        return (
            T.tensor(self.state_memory[batch], dtype=T.float32, device=device),
            T.tensor(self.action_memory[batch], dtype=T.float32, device=device),
            T.tensor(self.reward_memory[batch], dtype=T.float32, device=device).unsqueeze(-1),
            T.tensor(self.next_state_memory[batch], dtype=T.float32, device=device),
            T.tensor(self.terminated_memory[batch], dtype=T.float32, device=device).unsqueeze(-1),
            T.tensor(self.truncated_memory[batch], dtype=T.float32, device=device).unsqueeze(-1),
        )


# ----------------------------------------------------------------------------
# 3. Ensemble Dynamics World Model
# ----------------------------------------------------------------------------


class EnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int = 7):
        super().__init__()
        self.weight = nn.Parameter(T.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(T.empty(ensemble_size, 1, out_features))

        for w in self.weight:
            nn.init.trunc_normal_(w, std=1.0 / (2.0 * math.sqrt(w.shape[1])))
        nn.init.zeros_(self.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return T.bmm(x, self.weight) + self.bias


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        num_models: int = 7,
        hidden_dim: int = 256,
        chkpt_dir: str = "tmp/mbpo",
    ):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "dynamics_ensemble_mbpo.pt")
        os.makedirs(chkpt_dir, exist_ok=True)
        self.num_models = num_models

        self.fc1 = EnsembleLinear(state_dim + n_actions, hidden_dim, num_models)
        self.fc2 = EnsembleLinear(hidden_dim, hidden_dim, num_models)
        self.fc3 = EnsembleLinear(hidden_dim, hidden_dim, num_models)
        self.output_layer = EnsembleLinear(hidden_dim, (state_dim + 1) * 2, num_models)

        self.max_logvar = nn.Parameter(T.ones(1, state_dim + 1) * 0.5)
        self.min_logvar = nn.Parameter(T.ones(1, state_dim + 1) * -10.0)

    def forward(self, state: T.Tensor, action: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        inputs = T.cat([state, action], dim=-1)
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).repeat(self.num_models, 1, 1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.output_layer(x)

        mean, logvar = T.chunk(out, 2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar


class WorldModel:
    def __init__(self, state_dim: int, n_actions: int, lr: float = 1e-3, num_models: int = 7):
        self.num_models = num_models
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.model = EnsembleDynamicsModel(state_dim, n_actions, num_models).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_step(self, env_buffer: ReplayBuffer, obs_rms: RunningMeanStd, batch_size: int = 256) -> float:
        states, actions, rewards, next_states, _, _ = env_buffer.sample_buffer(batch_size, self.device)

        mean = T.as_tensor(obs_rms.mean, dtype=T.float32, device=self.device)
        std = T.as_tensor(obs_rms.std, dtype=T.float32, device=self.device) + 1e-8

        norm_states = (states - mean) / std
        norm_next_states = (next_states - mean) / std
        norm_delta = norm_next_states - norm_states

        targets = T.cat([norm_delta, rewards], dim=-1)
        pred_mean, pred_logvar = self.model(norm_states, actions)

        inv_var = T.exp(-pred_logvar)
        mse_loss = T.mean(T.square(pred_mean - targets) * inv_var, dim=-1)
        var_loss = T.mean(pred_logvar, dim=-1)
        loss = T.mean(mse_loss + var_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, states: T.Tensor, actions: T.Tensor, obs_rms: RunningMeanStd) -> Tuple[T.Tensor, T.Tensor]:
        with T.no_grad():
            mean = T.as_tensor(obs_rms.mean, dtype=T.float32, device=self.device)
            std = T.as_tensor(obs_rms.std, dtype=T.float32, device=self.device) + 1e-8

            norm_states = (states - mean) / std
            pred_mean, pred_logvar = self.model(norm_states, actions)
            pred_std = T.exp(0.5 * pred_logvar)
            samples = pred_mean + T.randn_like(pred_mean) * pred_std

            norm_delta = samples[..., :-1]
            rewards = samples[..., -1:]

            model_idx = random.randint(0, self.num_models - 1)
            next_norm_states = norm_states + norm_delta[model_idx]
            next_states = (next_norm_states * std) + mean
            extrinsic_rewards = rewards[model_idx]

        return next_states, extrinsic_rewards

    def save_checkpoint(self):
        T.save(self.model.state_dict(), self.model.checkpoint_file)

    def load_checkpoint(self):
        self.model.load_state_dict(T.load(self.model.checkpoint_file, map_location=self.device))


# ----------------------------------------------------------------------------
# 4. SAC Actor & Critic Networks
# ----------------------------------------------------------------------------


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha: float,
        state_dim: int,
        n_actions: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        name: str = "actor",
        chkpt_dir: str = "tmp/mbpo",
    ):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_mbpo.pt")
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean_linear = nn.Linear(fc2_dims, n_actions)
        self.log_std_linear = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = T.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = T.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= T.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta: float,
        state_dim: int,
        n_actions: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        name: str = "critic",
        chkpt_dir: str = "tmp/mbpo",
    ):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_mbpo.pt")
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(state_dim + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


# ----------------------------------------------------------------------------
# 5. SAC Agent
# ----------------------------------------------------------------------------


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        alpha: float = 3e-4,
        beta_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(alpha, state_dim, n_actions, name="actor")
        self.critic_1 = CriticNetwork(beta_lr, state_dim, n_actions, name="critic_1")
        self.critic_2 = CriticNetwork(beta_lr, state_dim, n_actions, name="critic_2")
        self.target_critic_1 = CriticNetwork(beta_lr, state_dim, n_actions, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(beta_lr, state_dim, n_actions, name="target_critic_2")

        self.update_network_parameters(tau=1.0)

        self.target_entropy = -float(n_actions)
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha)

    @property
    def entropy_alpha(self) -> T.Tensor:
        return self.log_alpha.exp()

    def choose_action(self, observation: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state = T.tensor(observation[None, :], dtype=T.float32, device=self.device)
        if evaluate:
            mean, _ = self.actor(state)
            action = T.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]

    def learn(
        self,
        env_buffer: ReplayBuffer,
        model_buffer: ReplayBuffer,
        batch_size: int = 256,
        real_ratio: float = 0.2,
    ) -> float:
        env_batch_size = int(batch_size * real_ratio)
        model_batch_size = batch_size - env_batch_size

        s_env, a_env, r_env, ns_env, term_env, _ = env_buffer.sample_buffer(env_batch_size, self.device)
        s_model, a_model, r_model, ns_model, term_model, _ = model_buffer.sample_buffer(
            model_batch_size, self.device
        )

        states = T.cat([s_env, s_model], dim=0)
        actions = T.cat([a_env, a_model], dim=0)
        rewards = T.cat([r_env, r_model], dim=0)
        next_states = T.cat([ns_env, ns_model], dim=0)
        terminateds = T.cat([term_env, term_model], dim=0)

        with T.no_grad():
            next_state_actions, next_state_log_pi = self.actor.sample(next_states)
            q1_next = self.target_critic_1(next_states, next_state_actions)
            q2_next = self.target_critic_2(next_states, next_state_actions)
            min_q_next = T.min(q1_next, q2_next) - self.entropy_alpha * next_state_log_pi
            target_q = rewards + (1.0 - terminateds) * self.gamma * min_q_next

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        pi, log_pi = self.actor.sample(states)
        q1_pi = self.critic_1(states, pi)
        q2_pi = self.critic_2(states, pi)
        min_q_pi = T.min(q1_pi, q2_pi)
        actor_loss = ((self.entropy_alpha * log_pi) - min_q_pi).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_network_parameters()
        return critic_loss.item()

    def update_network_parameters(self, tau: float = None) -> None:
        tau = self.tau if tau is None else tau
        with T.no_grad():
            for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def save_models(self) -> None:
        for net in (self.actor, self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2):
            net.save_checkpoint()

    def load_models(self) -> None:
        for net in (self.actor, self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2):
            net.load_checkpoint()


# ----------------------------------------------------------------------------
# 6. Rollout Generation & Environment Terminal Checker
# ----------------------------------------------------------------------------


def check_bipedal_done(next_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check BipedalWalker-v3 termination conditions:
    - Hull angle exceeding bounds (~0.4 rad or 23 degrees) implies the robot fell.
    Returns (terminated, rewards_adjustment).
    """
    hull_angles = np.abs(next_states[:, 0])
    is_fallen = hull_angles > 0.4
    return is_fallen, is_fallen.astype(np.float32) * -100.0


def generate_model_rollouts(
    env_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: SACAgent,
    obs_rms: RunningMeanStd,
    rollout_length: int = 1,
    batch_size: int = 256,
) -> None:
    states, _, _, _, _, _ = env_buffer.sample_buffer(batch_size, world_model.device)
    cur_states = states

    for _ in range(rollout_length):
        with T.no_grad():
            actions, _ = agent.actor.sample(cur_states)
            next_states, rewards = world_model.predict(cur_states, actions, obs_rms)

        np_states = cur_states.cpu().numpy()
        np_actions = actions.cpu().numpy()
        np_next_states = next_states.cpu().numpy()
        np_rewards = rewards.squeeze(-1).cpu().numpy()

        is_fallen, penalties = check_bipedal_done(np_next_states)
        final_rewards = np_rewards + penalties

        for i in range(batch_size):
            model_buffer.store_transition(
                np_states[i],
                np_actions[i],
                final_rewards[i],
                np_next_states[i],
                terminated=bool(is_fallen[i]),
                truncated=False,
            )

        cur_states = next_states


# ----------------------------------------------------------------------------
# 7. Training Loop with Save Checkpoints
# ----------------------------------------------------------------------------


def train(
    env_id: str = "BipedalWalker-v3",
    n_episodes: int = 3000,
    max_timesteps: int = int(1e6),
    warmup: int = 5000,
    seed: int = 0,
    figure_file: str = "plots/mbpo_bipedalwalker.png",
    plot_every: int = 50,
):
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]  # 24
    n_actions = env.action_space.shape[0]  # 4

    T.manual_seed(seed)
    np.random.seed(seed)

    obs_rms = RunningMeanStd(shape=(state_dim,))
    env_buffer = ReplayBuffer(1_000_000, state_dim, n_actions)
    model_buffer = ReplayBuffer(1_000_000, state_dim, n_actions)

    world_model = WorldModel(state_dim, n_actions)
    agent = SACAgent(state_dim, n_actions)

    ext_reward_history = deque(maxlen=100)
    total_timesteps = 0
    #start_time = time.time()

    recent_critic_losses = deque(maxlen=200)
    recent_world_model_losses = deque(maxlen=200)

    episode_numbers = []
    episode_scores = []
    best_score = -float("inf")

    try:
        for episode in range(1, n_episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            obs_rms.update(state)

            done = False
            ep_ext_reward = 0.0

            while not done:
                if total_timesteps < warmup:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                obs_rms.update(next_state)
                env_buffer.store_transition(state, action, reward, next_state, terminated, truncated)

                # Train ensemble world model and trigger synthetic rollouts
                if total_timesteps >= warmup and total_timesteps % 250 == 0:
                    for _ in range(6):
                        wm_loss = world_model.train_step(env_buffer, obs_rms)
                        recent_world_model_losses.append(wm_loss)
                    generate_model_rollouts(
                        env_buffer, model_buffer, world_model, agent, obs_rms, rollout_length=1
                    )

                # Policy updates
                if total_timesteps >= warmup and model_buffer.mem_cntr > 256:
                    for _ in range(2):
                        critic_loss = agent.learn(env_buffer, model_buffer, real_ratio=0.2)
                        recent_critic_losses.append(critic_loss)

                state = next_state
                ep_ext_reward += reward
                total_timesteps += 1

                if total_timesteps >= max_timesteps:
                    done = True

            ext_reward_history.append(ep_ext_reward)
            episode_numbers.append(episode)
            episode_scores.append(ep_ext_reward)

            # Save periodic checkpoints
            if episode % plot_every == 0:
                plot_learning_curve(episode_numbers, episode_scores, figure_file)
                agent.save_models()
                world_model.save_checkpoint()

            # Save best checkpoint when average score improves
            avg_reward = np.mean(ext_reward_history)
            if avg_reward > best_score and episode >= 50:
                best_score = avg_reward
                print(f"--> Saved best model checkpoint (Avg score: {best_score:.2f})")
                agent.save_models()
                world_model.save_checkpoint()

            if episode % 10 == 0:
                avg_critic_loss = (
                    np.mean(recent_critic_losses) if recent_critic_losses else 0.0
                )
                avg_wm_loss = (
                    np.mean(recent_world_model_losses)
                    if recent_world_model_losses
                    else 0.0
                )
                #elapsed = time.time() - start_time
                print(
                    f"Episode {episode:5d} | Timesteps {total_timesteps:8d} | "
                    f"Extrinsic Reward {ep_ext_reward:8.2f} (avg100 {avg_reward:8.2f}) | "
                    f"World Model Loss {avg_wm_loss:6.4f} | "
                    f"Critic Loss {avg_critic_loss:8.4f} | "
                    #f"Elapsed {elapsed:6.1f}s"
                )

            if total_timesteps >= max_timesteps:
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")

    # Final saving
    plot_learning_curve(episode_numbers, episode_scores, figure_file)
    env.close()
    agent.save_models()
    world_model.save_checkpoint()
    print(f"Saved weights to tmp/mbpo and updated graph in {figure_file}")


if __name__ == "__main__":
    train(
        env_id="BipedalWalker-v3",
        n_episodes=3000,
        max_timesteps=int(1e6),
        warmup=5000,
        seed=0,
        figure_file="plots/mbpo_bipedalwalker.png",
        plot_every=50,
    )