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


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity, self.device, self.position, self.size = capacity, device, 0, 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated):
        i = self.position
        self.states[i], self.actions[i] = state, action
        self.rewards[i], self.next_states[i], self.terminated[i] = reward, next_state, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        arrays = (self.states, self.actions, self.rewards, self.next_states, self.terminated)
        return tuple(torch.as_tensor(array[indices], dtype=torch.float32, device=self.device) for array in arrays)


def make_network(input_dim, output_dim, hidden_1=400, hidden_2=300):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_1), nn.ReLU(),
        nn.Linear(hidden_1, hidden_2), nn.ReLU(),
        nn.Linear(hidden_2, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim, action_low, action_high):
        super().__init__()
        self.net = make_network(state_dim, len(action_high))
        self.register_buffer("scale", torch.as_tensor((action_high - action_low) / 2, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor((action_high + action_low) / 2, dtype=torch.float32))

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.scale + self.bias


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = make_network(state_dim + action_dim, 1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


class TD3Agent:
    def __init__(self, state_dim, action_space, device, gamma=0.99, tau=0.005, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, exploration_noise=0.1):
        self.device, self.gamma, self.tau = device, gamma, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_delay, self.exploration_noise, self.update_count = policy_delay, exploration_noise, 0
        action_dim = action_space.shape[0]
        self.action_shape = action_space.shape
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2

        self.actor = Actor(state_dim, action_space.low, action_space.high).to(device)
        self.q1, self.q2 = Critic(state_dim, action_dim).to(device), Critic(state_dim, action_dim).to(device)
        self.target_actor, self.target_q1, self.target_q2 = copy.deepcopy(self.actor), copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for target in (self.target_actor, self.target_q1, self.target_q2):
            for parameter in target.parameters(): parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state)
        if not deterministic:
            action += torch.randn_like(action) * self.exploration_noise * self.action_scale
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low).cpu().numpy()[0].reshape(self.action_shape)

    def update(self, replay, batch_size):
        states, actions, rewards, next_states, terminated = replay.sample(batch_size)
        with torch.no_grad():
            # Target-policy smoothing reduces exploitation of narrow Q-function errors.
            target_noise = torch.randn_like(actions) * self.policy_noise * self.action_scale
            target_noise = target_noise.clamp(-self.noise_clip * self.action_scale, self.noise_clip * self.action_scale)
            next_actions = self.target_actor(next_states) + target_noise
            next_actions = torch.maximum(torch.minimum(next_actions, self.action_high), self.action_low)
            next_q = torch.minimum(self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions))
            q_target = rewards + self.gamma * (1 - terminated) * next_q

        critic_loss = F.mse_loss(self.q1(states, actions), q_target) + F.mse_loss(self.q2(states, actions), q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.update_count += 1

        actor_loss = None
        if self.update_count % self.policy_delay == 0:
            # The Q parameters remain fixed during the policy loss but dQ/da is retained.
            for critic in (self.q1, self.q2):
                for parameter in critic.parameters(): parameter.requires_grad_(False)
            actor_loss = -self.q1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for critic in (self.q1, self.q2):
                for parameter in critic.parameters(): parameter.requires_grad_(True)

            with torch.no_grad():
                for target, source in zip(self.target_actor.parameters(), self.actor.parameters()): target.lerp_(source, self.tau)
                for target, source in zip(self.target_q1.parameters(), self.q1.parameters()): target.lerp_(source, self.tau)
                for target, source in zip(self.target_q2.parameters(), self.q2.parameters()): target.lerp_(source, self.tau)
        return critic_loss.item(), None if actor_loss is None else actor_loss.item()


@torch.no_grad()
def evaluate(agent, env, episodes=5):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        done, score = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state, deterministic=True))
            done, score = terminated or truncated, score + reward
        scores.append(score)
    return float(np.mean(scores))


def train_td3(env_id="BipedalWalker-v3", total_steps=1_000_000, batch_size=100, buffer_size=1_000_000,
              start_steps=10_000, update_after=1_000, updates_per_step=1, seed=1, device=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    env, eval_env = gym.make(env_id), gym.make(env_id)
    state, _ = env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)
    env.action_space.seed(seed)
    agent = TD3Agent(env.observation_space.shape[0], env.action_space, device)
    replay = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], buffer_size, device)
    output_dir = Path(__file__).resolve().parent
    scores, rolling, score = [], deque(maxlen=100), 0.0
    episode, best_eval = 0, -np.inf

    for global_step in range(1, total_steps + 1):
        # Uniformly random actions during warm-up, rather than small Gaussian noise near zero.
        action = env.action_space.sample() if global_step <= start_steps else agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # A TimeLimit truncation ends an episode but is not an MDP terminal for the critic target.
        replay.add(state, action, reward, next_state, terminated)
        state, score = next_state, score + reward

        if global_step >= update_after and replay.size >= batch_size:
            for _ in range(updates_per_step): agent.update(replay, batch_size)

        if done:
            episode += 1
            scores.append(score); rolling.append(score)
            if episode % 20 == 0:
                evaluation = evaluate(agent, eval_env)
                if evaluation > best_eval:
                    best_eval = evaluation
                    torch.save({"actor": agent.actor.state_dict()}, output_dir / "td3_best.pt")
                print(f"Episode {episode:4d} | step {global_step:7d} | avg-100 {np.mean(rolling):7.1f} | eval {evaluation:7.1f}")
            else:
                print(f"Episode {episode:4d} | step {global_step:7d} | score {score:7.1f} | avg-100 {np.mean(rolling):7.1f}")
            state, _ = env.reset()
            score = 0.0

    running_average = [np.mean(scores[max(0, i - 99): i + 1]) for i in range(len(scores))]
    plt.plot(range(1, len(scores) + 1), running_average)
    plt.xlabel("Episode"); plt.ylabel("Training return (100-episode average)")
    plt.title("TD3 on BipedalWalker-v3")
    plt.tight_layout(); plt.savefig(output_dir / "td3_learning_curve.png", dpi=150); plt.close()
    env.close(); eval_env.close()


if __name__ == "__main__":
    train_td3(env_id="BipedalWalker-v3", total_steps=1_000_000)
