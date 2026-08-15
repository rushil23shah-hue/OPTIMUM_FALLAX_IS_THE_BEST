import copy
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.device = device
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.terminated[i] = terminated

        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)

        arrays = (
            self.states,
            self.actions,
            self.rewards,
            self.next_states,
            self.terminated,
        )

        return tuple(
            torch.as_tensor(array[indices], dtype=torch.float32, device=self.device)
            for array in arrays
        )


def make_network(input_dim, output_dim, hidden_dim=256):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim, action_low, action_high):
        super().__init__()

        action_dim = len(action_high)
        self.net = make_network(state_dim, action_dim * 2)

        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32),
        )

    def distribution(self, state):
        mean, log_std = self.net(state).chunk(2, dim=-1)

        # Prevent extremely small/large policy standard deviations.
        log_std = torch.clamp(log_std, -5, 2)
        return Normal(mean, log_std.exp())

    def sample(self, state):
        distribution = self.distribution(state)

        # Reparameterized action sample.
        latent_action = distribution.rsample()
        squashed_action = torch.tanh(latent_action)

        action = squashed_action * self.action_scale + self.action_bias

        # Tanh change-of-variables correction.
        log_prob = distribution.log_prob(latent_action)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    @torch.no_grad()
    def deterministic_action(self, state):
        mean = self.distribution(state).mean
        return torch.tanh(mean) * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = make_network(state_dim + action_dim, 1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


class SACAgent:
    def __init__(
        self,
        state_dim,
        action_space,
        device,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.1,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = float(alpha)
        self.action_shape = action_space.shape

        action_dim = action_space.shape[0]

        self.actor = Actor(
            state_dim, action_space.low, action_space.high
        ).to(device)

        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q2 = Critic(state_dim, action_dim).to(device)

        self.target_q1 = copy.deepcopy(self.q1).to(device)
        self.target_q2 = copy.deepcopy(self.q2).to(device)

        for target in (self.target_q1, self.target_q2):
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr,
        )

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if deterministic:
            action = self.actor.deterministic_action(state)
        else:
            action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0].reshape(self.action_shape)

    def update(self, replay, batch_size):
        states, actions, rewards, next_states, terminated = replay.sample(
            batch_size
        )

        # Twin target-critic SAC backup.
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)

            next_q = torch.minimum(
                self.target_q1(next_states, next_actions),
                self.target_q2(next_states, next_actions),
            )

            q_target = rewards + self.gamma * (1 - terminated) * (
                next_q - self.alpha * next_log_prob
            )

        q1_loss = F.mse_loss(self.q1(states, actions), q_target)
        q2_loss = F.mse_loss(self.q2(states, actions), q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Do not calculate unused critic parameter gradients during actor update.
        for critic in (self.q1, self.q2):
            for parameter in critic.parameters():
                parameter.requires_grad_(False)

        sampled_actions, log_prob = self.actor.sample(states)

        q_pi = torch.minimum(
            self.q1(states, sampled_actions),
            self.q2(states, sampled_actions),
        )

        # Fixed alpha: no automatic entropy tuning.
        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for critic in (self.q1, self.q2):
            for parameter in critic.parameters():
                parameter.requires_grad_(True)

        # Soft-update target critics.
        with torch.no_grad():
            for target, source in zip(
                self.target_q1.parameters(), self.q1.parameters()
            ):
                target.lerp_(source, self.tau)

            for target, source in zip(
                self.target_q2.parameters(), self.q2.parameters()
            ):
                target.lerp_(source, self.tau)

        return critic_loss.item(), actor_loss.item()


@torch.no_grad()
def evaluate(agent, env, episodes=5):
    scores = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0.0

        while not done:
            action = agent.act(state, deterministic=True)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        scores.append(score)

    return float(np.mean(scores))


def train_sac(
    env_id="BipedalWalker-v3",
    total_steps=1_000_000,
    batch_size=256,
    buffer_size=1_000_000,
    start_steps=10_000,
    update_after=1_000,
    updates_per_step=1,
    alpha=0.1,
    seed=1,
    device=None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    state, _ = env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)

    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(
        state_dim,
        env.action_space,
        device,
        alpha=alpha,
    )

    replay = ReplayBuffer(
        state_dim,
        action_dim,
        buffer_size,
        device,
    )

    output_dir = Path(__file__).resolve().parent
    scores = []
    rolling_scores = deque(maxlen=100)

    episode = 0
    episode_score = 0.0
    best_eval = -np.inf

    for global_step in range(1, total_steps + 1):
        if global_step <= start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # `truncated` ends the episode but still allows critic bootstrapping.
        done = terminated or truncated

        replay.add(
            state,
            action,
            reward,
            next_state,
            terminated,
        )

        state = next_state
        episode_score += reward

        if global_step >= update_after and replay.size >= batch_size:
            for _ in range(updates_per_step):
                agent.update(replay, batch_size)

        if done:
            episode += 1
            scores.append(episode_score)
            rolling_scores.append(episode_score)

            if episode % 20 == 0:
                evaluation = evaluate(agent, eval_env)

                if evaluation > best_eval:
                    best_eval = evaluation
                    torch.save(
                        {"actor": agent.actor.state_dict()},
                        output_dir / "sac_best.pt",
                    )

                print(
                    f"Episode {episode:4d} | "
                    f"step {global_step:7d} | "
                    f"avg-100 {np.mean(rolling_scores):7.1f} | "
                    f"eval {evaluation:7.1f} | "
                    f"alpha {alpha:.3f}"
                )
            else:
                print(
                    f"Episode {episode:4d} | "
                    f"step {global_step:7d} | "
                    f"score {episode_score:7.1f} | "
                    f"avg-100 {np.mean(rolling_scores):7.1f}"
                )

            state, _ = env.reset()
            episode_score = 0.0

    running_average = [
        np.mean(scores[max(0, i - 99): i + 1])
        for i in range(len(scores))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(scores) + 1), running_average)
    plt.xlabel("Episode")
    plt.ylabel("Training return (100-episode average)")
    plt.title("SAC-v2 on BipedalWalker-v3")
    plt.tight_layout()
    plt.savefig(output_dir / "sac_learning_curve.png", dpi=150)
    plt.close()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_sac(
        env_id="BipedalWalker-v3",
        total_steps=1_000_000,
        alpha=0.1,
    )