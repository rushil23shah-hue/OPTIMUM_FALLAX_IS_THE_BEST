import os
import time
from collections import deque
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def plot_learning_curve(x, scores, figure_file: str, window: int = 100) -> None:
    """Plot a running average of episode scores and save to disk."""
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - window):(i + 1)])

    plt.figure()
    plt.plot(x, scores, alpha=0.3, label="Episode reward")
    plt.plot(x, running_avg, label=f"Running avg ({window})")
    plt.title("TD3 + RND on BipedalWalker-v3")
    plt.xlabel("Episode")
    plt.ylabel("Extrinsic Reward")
    plt.legend()
    os.makedirs(os.path.dirname(figure_file) or ".", exist_ok=True)
    plt.savefig(figure_file)
    plt.close()

# 1. Running mean/std tracker 

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


# 2. Replay buffer -- intrinsic reward is intentionally NOT stored here.

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
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminated_memory[index] = terminated
        self.truncated_memory[index] = truncated
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        return (
            self.state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.next_state_memory[batch],
            self.terminated_memory[batch],
            self.truncated_memory[batch],
        )


# 3. TD3 Actor / Critic networks
class CriticNetwork(nn.Module):
    def __init__(self, beta: float, state_dim: int, n_actions: int,
                 fc1_dims: int = 256, fc2_dims: int = 256,
                 name: str = "critic", chkpt_dir: str = "tmp/td3_rnd"):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_td3_rnd.pt")
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


class ActorNetwork(nn.Module):
    def __init__(self, alpha: float, state_dim: int, n_actions: int,
                 fc1_dims: int = 256, fc2_dims: int = 256,
                 name: str = "actor", chkpt_dir: str = "tmp/td3_rnd"):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + "_td3_rnd.pt")
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return T.tanh(self.mu(x))  # BipedalWalker action range is [-1, 1]

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))

# 4. RND Target / Predictor networks


def _orthogonal_init(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)


class RNDTargetNetwork(nn.Module):
    """Frozen, randomly-initialized network. Never trained."""

    def __init__(self, state_dim: int, embed_dim: int = 128,
                 hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embed_dim),
        )
        _orthogonal_init(self.net)

        # Freeze: no gradients ever flow into (or out of, via .backward()) this net.
        for p in self.net.parameters():
            p.requires_grad = False

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        self.eval()

    @T.no_grad()
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Trainable network that tries to imitate the frozen target's output."""

    def __init__(self, state_dim: int, embed_dim: int = 128,
                 hidden_dims: Tuple[int, int] = (256, 256), lr: float = 3e-4):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embed_dim), 
        )
        _orthogonal_init(self.net)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# 5. Agent: TD3 core + off-policy RND bonus
# ----------------------------------------------------------------------------

class TD3RNDAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        max_action: np.ndarray,
        min_action: np.ndarray,
        alpha: float = 3e-4,
        beta_lr: float = 3e-4,
        rnd_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        max_size: int = 1_000_000,
        batch_size: int = 256,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_interval: int = 2,
        warmup: int = 1000,
        exploration_noise: float = 0.1,
        intrinsic_beta: float = 0.05,
        rnd_embed_dim: int = 128,
        obs_clip: float = 5.0,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup
        self.exploration_noise = exploration_noise
        self.intrinsic_beta = intrinsic_beta
        self.obs_clip = obs_clip
        self.n_actions = n_actions

        self.max_action = max_action
        self.min_action = min_action

        self.time_step = 0
        self.learn_step_cntr = 0

        self.memory = ReplayBuffer(max_size, state_dim, n_actions)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # --- TD3 core ---
        self.actor = ActorNetwork(alpha, state_dim, n_actions, name="actor")
        self.critic_1 = CriticNetwork(beta_lr, state_dim, n_actions, name="critic_1")
        self.critic_2 = CriticNetwork(beta_lr, state_dim, n_actions, name="critic_2")

        self.target_actor = ActorNetwork(alpha, state_dim, n_actions, name="target_actor")
        self.target_critic_1 = CriticNetwork(beta_lr, state_dim, n_actions, name="target_critic_1")
        self.target_critic_2 = CriticNetwork(beta_lr, state_dim, n_actions, name="target_critic_2")
        self.update_network_parameters(tau=1.0)

        # --- RND ---
        self.rnd_target = RNDTargetNetwork(state_dim, embed_dim=rnd_embed_dim)
        self.rnd_predictor = RNDPredictorNetwork(state_dim, embed_dim=rnd_embed_dim, lr=rnd_lr)

        # Running stats: one for normalizing raw states before they hit the
        # RND nets, one for normalizing the scale of the intrinsic reward.
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.intrinsic_reward_rms = RunningMeanStd(shape=())

    # Environment-facing helpers

    def update_obs_rms(self, raw_state: np.ndarray) -> None:
        """Call once per env step with the raw (unnormalized) state observed."""
        self.obs_rms.update(raw_state[None, :])

    def _normalize_for_rnd(self, states: T.Tensor) -> T.Tensor:
        mean = T.as_tensor(self.obs_rms.mean, dtype=T.float32, device=self.device)
        std = T.as_tensor(self.obs_rms.std, dtype=T.float32, device=self.device)
        normed = (states - mean) / (std + 1e-8)
        return T.clamp(normed, -self.obs_clip, self.obs_clip)

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        if self.time_step < self.warmup:
            mu = np.random.uniform(self.min_action, self.max_action, size=(self.n_actions,))
            mu = T.tensor(mu, dtype=T.float32, device=self.device)
        else:
            state = T.tensor(observation[None, :], dtype=T.float32, device=self.device)
            with T.no_grad():
                mu = self.actor(state)[0]

        noise = T.tensor(
            np.random.normal(scale=self.exploration_noise, size=(self.n_actions,)),
            dtype=T.float32, device=self.device,
        )
        mu_prime = mu + noise

        min_act = T.tensor(self.min_action, dtype=T.float32, device=self.device)
        max_act = T.tensor(self.max_action, dtype=T.float32, device=self.device)
        mu_prime = T.clamp(mu_prime, min_act, max_act)

        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, terminated, truncated) -> None:
        self.memory.store_transition(state, action, reward, next_state, terminated, truncated)

    # Intrinsic reward (computed fresh every learn() call, never stored)


    def _compute_intrinsic_reward(self, next_states: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        """
        Returns:
            intrinsic_detached: (batch,) tensor, detached, safe to add to the
                critic's regression target.
            predictor_loss: (scalar) tensor WITH graph attached, used to train
                the predictor network only.
        """
        normed_next = self._normalize_for_rnd(next_states)

        target_feat = self.rnd_target(normed_next)          # frozen, no_grad already
        pred_feat = self.rnd_predictor(normed_next)          # trainable, has graph

        # Per-sample novelty for the critic bonus (no graph needed).
        with T.no_grad():
            per_sample_error = F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1)

        # Update the running std used to normalize reward *scale* only.
        self.intrinsic_reward_rms.update(per_sample_error.cpu().numpy())
        int_std = float(self.intrinsic_reward_rms.std) + 1e-8
        intrinsic_detached = (per_sample_error / int_std).detach()

        # Predictor loss: full batch MSE, WITH gradient, used only to update
        # the predictor's own weights.
        predictor_loss = F.mse_loss(pred_feat, target_feat.detach())

        return intrinsic_detached, predictor_loss


    # Learning step
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None

        state, action, reward, next_state, terminated, truncated = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float32, device=self.device)
        action = T.tensor(action, dtype=T.float32, device=self.device)
        reward = T.tensor(reward, dtype=T.float32, device=self.device)
        next_state = T.tensor(next_state, dtype=T.float32, device=self.device)
        terminated_t = T.tensor(terminated, dtype=T.float32, device=self.device)  # 1.0 = real failure

        # --- Intrinsic reward, computed on-the-fly from CURRENT RND weights ---
        intrinsic_reward, predictor_loss = self._compute_intrinsic_reward(next_state)

        # Do not award intrinsic bonus on true terminal (failure) transitions;
        # keep it for truncation (time-limit) transitions.
        intrinsic_reward = intrinsic_reward * (1.0 - terminated_t)

        total_reward = reward + self.intrinsic_beta * intrinsic_reward  # already detached

        # --- Update RND predictor (isolated optimizer, isolated graph) ---
        self.rnd_predictor.optimizer.zero_grad()
        predictor_loss.backward()
        self.rnd_predictor.optimizer.step()

        # --- TD3 critic target ---
        with T.no_grad():
            target_actions = self.target_actor(next_state)
            noise = T.clamp(
                T.normal(mean=0.0, std=self.policy_noise, size=target_actions.shape, device=self.device),
                -self.noise_clip, self.noise_clip,
            )
            min_act = T.tensor(self.min_action, dtype=T.float32, device=self.device)
            max_act = T.tensor(self.max_action, dtype=T.float32, device=self.device)
            target_actions = T.clamp(target_actions + noise, min_act, max_act)

            q1_next = self.target_critic_1(next_state, target_actions).squeeze(-1)
            q2_next = self.target_critic_2(next_state, target_actions).squeeze(-1)
            q_next = T.min(q1_next, q2_next)

            # Bootstrap unless the episode truly ended (terminated). A
            # truncation still bootstraps because the trajectory didn't
            # actually end there.
            target_q = total_reward + self.gamma * (1.0 - terminated_t) * q_next
            target_q = target_q.unsqueeze(-1)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        actor_loss_val = None
        if self.learn_step_cntr % self.update_actor_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_loss_val = actor_loss.item()

            self.update_network_parameters()

        return critic_loss.item(), intrinsic_reward.mean().item()

    def update_network_parameters(self, tau: float = None) -> None:
        tau = self.tau if tau is None else tau
        with T.no_grad():
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for tp, p in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
            for tp, p in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def save_models(self) -> None:
        for net in (self.actor, self.target_actor, self.critic_1, self.critic_2,
                    self.target_critic_1, self.target_critic_2):
            net.save_checkpoint()
        T.save(self.rnd_predictor.state_dict(), "tmp/td3_rnd/rnd_predictor.pt")

    def load_models(self) -> None:
        for net in (self.actor, self.target_actor, self.critic_1, self.critic_2,
                    self.target_critic_1, self.target_critic_2):
            net.load_checkpoint()
        self.rnd_predictor.load_state_dict(T.load("tmp/td3_rnd/rnd_predictor.pt", map_location=self.rnd_predictor.device))


# ----------------------------------------------------------------------------
# 6. Training loop
# ----------------------------------------------------------------------------

def train(
    env_id: str = "BipedalWalker-v3",
    n_episodes: int = 3000,
    max_timesteps: int = int(1e6),
    warmup: int = 1000,
    seed: int = 0,
    figure_file: str = "plots/td3_rnd_bipedalwalker.png",
    plot_every: int = 50,
):
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]      # 24
    n_actions = env.action_space.shape[0]            # 4
    max_action = env.action_space.high               # +1.0
    min_action = env.action_space.low                 # -1.0

    T.manual_seed(seed)
    np.random.seed(seed)

    agent = TD3RNDAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        max_action=max_action,
        min_action=min_action,
        alpha=3e-4,
        beta_lr=3e-4,
        rnd_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        max_size=1_000_000,
        batch_size=256,
        policy_noise=0.2,
        noise_clip=0.5,
        update_actor_interval=2,
        warmup=warmup,
        intrinsic_beta=0.05,
    )

    ext_reward_history = deque(maxlen=100)
    total_timesteps = 0
    start_time = time.time()

    # Running-per-episode diagnostics for the print block.
    recent_critic_losses = deque(maxlen=200)
    recent_intrinsic = deque(maxlen=200)

    # Full per-episode history, kept for the learning-curve plot.
    episode_numbers = []
    episode_scores = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        agent.update_obs_rms(state)

        done = False
        ep_ext_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_obs_rms(next_state)
            agent.remember(state, action, reward, next_state, terminated, truncated)

            critic_loss, intrinsic_mean = agent.learn()
            if critic_loss is not None:
                recent_critic_losses.append(critic_loss)
                recent_intrinsic.append(intrinsic_mean)

            state = next_state
            ep_ext_reward += reward
            total_timesteps += 1

            if total_timesteps >= max_timesteps:
                done = True

        ext_reward_history.append(ep_ext_reward)
        episode_numbers.append(episode)
        episode_scores.append(ep_ext_reward)

        if episode % plot_every == 0:
            plot_learning_curve(episode_numbers, episode_scores, figure_file)

        if episode % 10 == 0:
            avg_reward = np.mean(ext_reward_history)
            avg_critic_loss = np.mean(recent_critic_losses) if recent_critic_losses else 0.0
            avg_intrinsic = np.mean(recent_intrinsic) if recent_intrinsic else 0.0
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:5d} | Timesteps {total_timesteps:8d} | "
                f"Extrinsic Reward {ep_ext_reward:8.2f} (avg100 {avg_reward:8.2f}) | "
                f"Intrinsic Reward Mean {avg_intrinsic:6.4f} | "
                f"Critic Loss {avg_critic_loss:8.4f} | "
                f"Elapsed {elapsed:6.1f}s"
            )

        if total_timesteps >= max_timesteps:
            break

    plot_learning_curve(episode_numbers, episode_scores, figure_file)
    env.close()
    agent.save_models()
    print(f"Learning curve saved to {figure_file}")


if __name__ == "__main__":
    train(
        env_id="BipedalWalker-v3",
        n_episodes=3000,
        max_timesteps=int(1e6),
        warmup=1000,
        seed=0,
        figure_file="plots/td3_rnd_bipedalwalker.png",
        plot_every=50,
    )