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


class RunningMeanStd:
    """Streaming mean/variance used to scale the non-stationary RND error."""
    def __init__(self, epsilon=1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, epsilon

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        batch_mean, batch_var, batch_count = values.mean(), values.var(), values.size
        delta, total = batch_mean - self.mean, self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total


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


class RandomNetworkDistillation(nn.Module):
    """A fixed random target and a trainable predictor; novelty is prediction error."""
    def __init__(self, state_dim, feature_dim=128):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, feature_dim))
        self.predictor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, feature_dim))
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.error_rms = RunningMeanStd()

    @torch.no_grad()
    def intrinsic_reward(self, next_states):
        error = (self.predictor(next_states) - self.target(next_states)).pow(2).mean(dim=-1, keepdim=True)
        self.error_rms.update(error.squeeze(-1).cpu().numpy())
        # The error stays non-negative; divide by its running scale and cap early spikes.
        return (error / np.sqrt(self.error_rms.var + 1e-8)).clamp(max=5.0)

    def update(self, next_states):
        target_features = self.target(next_states)
        predicted_features = self.predictor(next_states)
        loss = F.mse_loss(predicted_features, target_features)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class TD3RNDAgent:
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
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1).to(device), copy.deepcopy(self.q2).to(device)
        for target in (self.target_actor, self.target_q1, self.target_q2):
            for parameter in target.parameters(): parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.rnd = RandomNetworkDistillation(state_dim).to(device)

    @torch.no_grad()
    def act(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state)
        if not deterministic:
            action += torch.randn_like(action) * self.exploration_noise * self.action_scale
        action = torch.maximum(torch.minimum(action, self.action_high), self.action_low)
        return action.cpu().numpy()[0].reshape(self.action_shape)

    def update(self, replay, batch_size, rnd_scale=0.01):
        states, actions, rewards, next_states, terminated = replay.sample(batch_size)
        with torch.no_grad():
            intrinsic_reward = self.rnd.intrinsic_reward(next_states)
            target_noise = torch.randn_like(actions) * self.policy_noise * self.action_scale
            target_noise = target_noise.clamp(-self.noise_clip * self.action_scale, self.noise_clip * self.action_scale)
            next_actions = self.target_actor(next_states) + target_noise
            next_actions = torch.maximum(torch.minimum(next_actions, self.action_high), self.action_low)
            next_q = torch.minimum(self.target_q1(next_states, next_actions), self.target_q2(next_states, next_actions))
            # RND affects learning only; reported/evaluation score remains extrinsic environment reward.
            q_target = rewards + rnd_scale * intrinsic_reward + self.gamma * (1 - terminated) * next_q

        critic_loss = F.mse_loss(self.q1(states, actions), q_target) + F.mse_loss(self.q2(states, actions), q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        rnd_loss = self.rnd.update(next_states)
        self.update_count += 1

        actor_loss = None
        if self.update_count % self.policy_delay == 0:
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

        return critic_loss.item(), rnd_loss, intrinsic_reward.mean().item(), None if actor_loss is None else actor_loss.item()


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


def train_td3_rnd(env_id="BipedalWalker-v3", total_steps=1_000_000, batch_size=100, buffer_size=1_000_000,
                  start_steps=10_000, update_after=1_000, updates_per_step=1, rnd_scale=0.01, seed=1, device=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    env, eval_env = gym.make(env_id), gym.make(env_id)
    state, _ = env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)
    env.action_space.seed(seed)
    agent = TD3RNDAgent(env.observation_space.shape[0], env.action_space, device)
    replay = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], buffer_size, device)
    output_dir = Path(__file__).resolve().parent
    scores, rolling, score = [], deque(maxlen=100), 0.0
    episode, best_eval, intrinsic_mean = 0, -np.inf, 0.0

    for global_step in range(1, total_steps + 1):
        action = env.action_space.sample() if global_step <= start_steps else agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Truncation ends collection, but only true termination blocks Bellman bootstrapping.
        replay.add(state, action, reward, next_state, terminated)
        state, score = next_state, score + reward

        if global_step >= update_after and replay.size >= batch_size:
            for _ in range(updates_per_step):
                _, _, intrinsic_mean, _ = agent.update(replay, batch_size, rnd_scale)

        if done:
            episode += 1
            scores.append(score); rolling.append(score)
            if episode % 20 == 0:
                evaluation = evaluate(agent, eval_env)
                if evaluation > best_eval:
                    best_eval = evaluation
                    torch.save({"actor": agent.actor.state_dict(), "rnd_predictor": agent.rnd.predictor.state_dict()}, output_dir / "td3_rnd_best.pt")
                print(f"Episode {episode:4d} | step {global_step:7d} | avg-100 {np.mean(rolling):7.1f} | eval {evaluation:7.1f} | RND {intrinsic_mean:.3f}")
            else:
                print(f"Episode {episode:4d} | step {global_step:7d} | score {score:7.1f} | avg-100 {np.mean(rolling):7.1f}")
            state, _ = env.reset()
            score = 0.0

    running_average = [np.mean(scores[max(0, i - 99):i + 1]) for i in range(len(scores))]
    plt.plot(range(1, len(scores) + 1), running_average)
    plt.xlabel("Episode"); plt.ylabel("Extrinsic return (100-episode average)")
    plt.title("TD3 + RND on BipedalWalker-v3")
    plt.tight_layout(); plt.savefig(output_dir / "td3_rnd_learning_curve.png", dpi=150); plt.close()
    env.close(); eval_env.close()


if __name__ == "__main__":
    train_td3_rnd(env_id="BipedalWalker-v3", total_steps=1_000_000, rnd_scale=0.01)
