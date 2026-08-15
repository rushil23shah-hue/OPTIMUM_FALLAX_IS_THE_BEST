
import math
import os
import random
from collections import deque
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def plot_learning_curve(
    x, scores, figure_file: str, window: int = 100
) -> None:
    running_avg = np.zeros(len(scores), dtype=np.float32)
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - window + 1) : i + 1])

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


class RunningMeanStd:
    """Running statistics used only until the normalization is frozen."""

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

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 1e-12))


# ----------------------------------------------------------------------------
# Replay buffers
# ----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, n_actions: int):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros(
            (max_size, state_dim), dtype=np.float32
        )
        self.next_state_memory = np.zeros(
            (max_size, state_dim), dtype=np.float32
        )
        self.action_memory = np.zeros(
            (max_size, n_actions), dtype=np.float32
        )
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminated_memory = np.zeros(max_size, dtype=np.bool_)
        self.truncated_memory = np.zeros(max_size, dtype=np.bool_)

    def __len__(self) -> int:
        return min(self.mem_cntr, self.mem_size)

    def clear(self) -> None:
        self.mem_cntr = 0

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
    ) -> None:
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminated_memory[index] = bool(terminated)
        self.truncated_memory[index] = bool(truncated)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int, device: T.device):
        max_mem = len(self)
        if max_mem < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with "
                f"{max_mem} transitions."
            )

        batch = np.random.choice(max_mem, batch_size, replace=False)

        return (
            T.as_tensor(
                self.state_memory[batch], dtype=T.float32, device=device
            ),
            T.as_tensor(
                self.action_memory[batch], dtype=T.float32, device=device
            ),
            T.as_tensor(
                self.reward_memory[batch], dtype=T.float32, device=device
            ).unsqueeze(-1),
            T.as_tensor(
                self.next_state_memory[batch],
                dtype=T.float32,
                device=device,
            ),
            T.as_tensor(
                self.terminated_memory[batch],
                dtype=T.float32,
                device=device,
            ).unsqueeze(-1),
            T.as_tensor(
                self.truncated_memory[batch],
                dtype=T.float32,
                device=device,
            ).unsqueeze(-1),
        )


# ----------------------------------------------------------------------------
# Ensemble dynamics model
# ----------------------------------------------------------------------------

class EnsembleLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 7,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            T.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(
            T.zeros(ensemble_size, 1, out_features)
        )

        for w in self.weight:
            nn.init.trunc_normal_(
                w,
                std=1.0 / (2.0 * math.sqrt(w.shape[1])),
            )

    def forward(self, x: T.Tensor) -> T.Tensor:
        # x: [ensemble, batch, features]
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

        self.checkpoint_file = os.path.join(
            chkpt_dir, "dynamics_ensemble_mbpo.pt"
        )
        os.makedirs(chkpt_dir, exist_ok=True)

        self.num_models = num_models

        self.fc1 = EnsembleLinear(
            state_dim + n_actions, hidden_dim, num_models
        )
        self.fc2 = EnsembleLinear(hidden_dim, hidden_dim, num_models)
        self.fc3 = EnsembleLinear(hidden_dim, hidden_dim, num_models)

        # Predict normalized delta-state + reward.
        self.output_layer = EnsembleLinear(
            hidden_dim,
            (state_dim + 1) * 2,
            num_models,
        )

        self.max_logvar = nn.Parameter(
            T.ones(1, state_dim + 1) * 0.5
        )
        self.min_logvar = nn.Parameter(
            T.ones(1, state_dim + 1) * -10.0
        )

    def forward(
        self,
        state: T.Tensor,
        action: T.Tensor,
    ) -> Tuple[T.Tensor, T.Tensor]:

        if state.ndim == 2:
            state = state.unsqueeze(0).repeat(
                self.num_models, 1, 1
            )
            action = action.unsqueeze(0).repeat(
                self.num_models, 1, 1
            )

        x = T.cat([state, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        out = self.output_layer(x)
        mean, logvar = T.chunk(out, 2, dim=-1)

        logvar = (
            self.max_logvar
            - F.softplus(self.max_logvar - logvar)
        )
        logvar = (
            self.min_logvar
            + F.softplus(logvar - self.min_logvar)
        )

        return mean, logvar


class WorldModel:
    """
    Probabilistic bootstrap ensemble.

    The ensemble members are trained on independently sampled bootstrap
    batches, which gives the ensemble meaningful disagreement.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        num_models: int = 7,
    ):
        self.num_models = num_models
        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )

        self.model = EnsembleDynamicsModel(
            state_dim,
            n_actions,
            num_models,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        self.state_mean = None
        self.state_std = None

    def set_normalization(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> None:
        self.state_mean = np.asarray(
            state_mean, dtype=np.float32
        ).copy()

        self.state_std = np.maximum(
            np.asarray(state_std, dtype=np.float32),
            1e-3,
        ).copy()

    def _normalization_tensors(self):
        if self.state_mean is None or self.state_std is None:
            raise RuntimeError(
                "World-model normalization has not been set."
            )

        mean = T.as_tensor(
            self.state_mean,
            dtype=T.float32,
            device=self.device,
        )
        std = T.as_tensor(
            self.state_std,
            dtype=T.float32,
            device=self.device,
        )
        return mean, std

    def train_step(
        self,
        env_buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> float:

        if len(env_buffer) < batch_size:
            return 0.0

        mean, std = self._normalization_tensors()

        # Independent bootstrap batch for every ensemble member.
        max_mem = len(env_buffer)
        indices = np.stack(
            [
                np.random.choice(
                    max_mem,
                    batch_size,
                    replace=False,
                )
                for _ in range(self.num_models)
            ],
            axis=0,
        )

        states = T.as_tensor(
            env_buffer.state_memory[indices],
            dtype=T.float32,
            device=self.device,
        )
        actions = T.as_tensor(
            env_buffer.action_memory[indices],
            dtype=T.float32,
            device=self.device,
        )
        rewards = T.as_tensor(
            env_buffer.reward_memory[indices],
            dtype=T.float32,
            device=self.device,
        ).unsqueeze(-1)

        next_states = T.as_tensor(
            env_buffer.next_state_memory[indices],
            dtype=T.float32,
            device=self.device,
        )

        norm_states = (states - mean) / std
        norm_next_states = (next_states - mean) / std
        norm_delta = norm_next_states - norm_states

        targets = T.cat([norm_delta, rewards], dim=-1)

        pred_mean, pred_logvar = self.model(
            norm_states,
            actions,
        )

        # Gaussian NLL:
        # 0.5 * ((target - mean)^2 / variance + log variance)
        nll = 0.5 * (
            (targets - pred_mean).pow(2)
            * T.exp(-pred_logvar)
            + pred_logvar
        )

        loss = nll.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        T.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0,
        )
        self.optimizer.step()

        return float(loss.item())

    @T.no_grad()
    def predict(
        self,
        states: T.Tensor,
        actions: T.Tensor,
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:

        mean, logvar = self._normalization_tensors()

        norm_states = (states - mean) / std

        pred_mean, pred_logvar = self.model(
            norm_states,
            actions,
        )

        # Sample aleatoric uncertainty from each ensemble member.
        pred_std = T.exp(0.5 * pred_logvar)
        samples = (
            pred_mean
            + T.randn_like(pred_mean) * pred_std
        )

        norm_delta = samples[..., :-1]
        rewards = samples[..., -1:]

        # Randomized model assignment per transition.
        batch_size = states.shape[0]
        model_indices = T.randint(
            low=0,
            high=self.num_models,
            size=(batch_size,),
            device=self.device,
        )

        batch_indices = T.arange(
            batch_size,
            device=self.device,
        )

        selected_delta = norm_delta[
            model_indices,
            batch_indices,
        ]
        selected_rewards = rewards[
            model_indices,
            batch_indices,
        ]

        next_norm_states = norm_states + selected_delta
        next_states = next_norm_states * std + mean

        # Epistemic uncertainty: disagreement between ensemble means.
        mean_delta = pred_mean[..., :-1]
        disagreement = mean_delta.var(
            dim=0,
            unbiased=False,
        ).mean(dim=-1, keepdim=True).sqrt()

        return (
            next_states,
            selected_rewards,
            disagreement,
        )

    def save_checkpoint(self):
        T.save(
            {
                "model": self.model.state_dict(),
                "state_mean": self.state_mean,
                "state_std": self.state_std,
            },
            self.model.checkpoint_file,
        )

    def load_checkpoint(self):
        checkpoint = T.load(
            self.model.checkpoint_file,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.state_mean = checkpoint["state_mean"]
        self.state_std = checkpoint["state_std"]


# ----------------------------------------------------------------------------
# SAC
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

        self.checkpoint_file = os.path.join(
            chkpt_dir,
            name + "_mbpo.pt",
        )
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean_linear = nn.Linear(fc2_dims, n_actions)
        self.log_std_linear = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=alpha,
        )

        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)

        std = log_std.exp()
        normal = T.distributions.Normal(mean, std)

        x_t = normal.rsample()
        action = T.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= T.log(
            1.0 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(
            dim=1,
            keepdim=True,
        )

        return action, log_prob

    def save_checkpoint(self):
        T.save(
            self.state_dict(),
            self.checkpoint_file,
        )

    def load_checkpoint(self):
        self.load_state_dict(
            T.load(
                self.checkpoint_file,
                map_location=self.device,
            )
        )


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

        self.checkpoint_file = os.path.join(
            chkpt_dir,
            name + "_mbpo.pt",
        )
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(
            state_dim + n_actions,
            fc1_dims,
        )
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=beta,
        )

        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        T.save(
            self.state_dict(),
            self.checkpoint_file,
        )

    def load_checkpoint(self):
        self.load_state_dict(
            T.load(
                self.checkpoint_file,
                map_location=self.device,
            )
        )


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

        self.device = T.device(
            "cuda:0" if T.cuda.is_available() else "cpu"
        )

        self.actor = ActorNetwork(
            alpha,
            state_dim,
            n_actions,
            name="actor",
        )

        self.critic_1 = CriticNetwork(
            beta_lr,
            state_dim,
            n_actions,
            name="critic_1",
        )
        self.critic_2 = CriticNetwork(
            beta_lr,
            state_dim,
            n_actions,
            name="critic_2",
        )

        self.target_critic_1 = CriticNetwork(
            beta_lr,
            state_dim,
            n_actions,
            name="target_critic_1",
        )
        self.target_critic_2 = CriticNetwork(
            beta_lr,
            state_dim,
            n_actions,
            name="target_critic_2",
        )

        self.update_network_parameters(tau=1.0)

        self.target_entropy = -float(n_actions)

        self.log_alpha = T.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=alpha,
        )

    @property
    def entropy_alpha(self):
        return self.log_alpha.exp()

    def choose_action(
        self,
        observation: np.ndarray,
        evaluate: bool = False,
    ) -> np.ndarray:

        state = T.as_tensor(
            observation[None, :],
            dtype=T.float32,
            device=self.device,
        )

        with T.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = T.tanh(mean)
            else:
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def learn(
        self,
        env_buffer: ReplayBuffer,
        model_buffer: ReplayBuffer,
        batch_size: int = 256,
        real_ratio: float = 0.5,
    ) -> dict:

        real_batch = int(batch_size * real_ratio)
        model_batch = batch_size - real_batch

        if len(env_buffer) < real_batch:
            return {}

        if model_batch > 0 and len(model_buffer) < model_batch:
            return {}

        (
            s_env,
            a_env,
            r_env,
            ns_env,
            term_env,
            _,
        ) = env_buffer.sample_buffer(
            real_batch,
            self.device,
        )

        if model_batch > 0:
            (
                s_model,
                a_model,
                r_model,
                ns_model,
                term_model,
                _,
            ) = model_buffer.sample_buffer(
                model_batch,
                self.device,
            )

            states = T.cat([s_env, s_model], dim=0)
            actions = T.cat([a_env, a_model], dim=0)
            rewards = T.cat([r_env, r_model], dim=0)
            next_states = T.cat([ns_env, ns_model], dim=0)
            terminateds = T.cat(
                [term_env, term_model],
                dim=0,
            )
        else:
            states = s_env
            actions = a_env
            rewards = r_env
            next_states = ns_env
            terminateds = term_env

        with T.no_grad():
            next_actions, next_log_pi = self.actor.sample(
                next_states
            )

            q1_next = self.target_critic_1(
                next_states,
                next_actions,
            )
            q2_next = self.target_critic_2(
                next_states,
                next_actions,
            )

            min_q_next = (
                T.min(q1_next, q2_next)
                - self.entropy_alpha * next_log_pi
            )

            target_q = (
                rewards
                + (1.0 - terminateds)
                * self.gamma
                * min_q_next
            )

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_loss = (
            F.mse_loss(q1, target_q)
            + F.mse_loss(q2, target_q)
        )

        self.critic_1.optimizer.zero_grad(
            set_to_none=True
        )
        self.critic_2.optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        pi, log_pi = self.actor.sample(states)

        q1_pi = self.critic_1(states, pi)
        q2_pi = self.critic_2(states, pi)

        min_q_pi = T.min(q1_pi, q2_pi)

        actor_loss = (
            self.entropy_alpha.detach() * log_pi
            - min_q_pi
        ).mean()

        self.actor.optimizer.zero_grad(
            set_to_none=True
        )
        actor_loss.backward()
        self.actor.optimizer.step()

        alpha_loss = -(
            self.log_alpha
            * (
                log_pi
                + self.target_entropy
            ).detach()
        ).mean()

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_network_parameters()

        return {
            "critic_loss": float(
                critic_loss.item()
            ),
            "actor_loss": float(
                actor_loss.item()
            ),
            "alpha": float(
                self.entropy_alpha.item()
            ),
            "q_mean": float(
                min_q_pi.mean().item()
            ),
        }

    def update_network_parameters(
        self,
        tau: float = None,
    ):
        tau = self.tau if tau is None else tau

        with T.no_grad():
            for target, source in zip(
                self.target_critic_1.parameters(),
                self.critic_1.parameters(),
            ):
                target.data.mul_(1.0 - tau)
                target.data.add_(tau * source.data)

            for target, source in zip(
                self.target_critic_2.parameters(),
                self.critic_2.parameters(),
            ):
                target.data.mul_(1.0 - tau)
                target.data.add_(tau * source.data)

    def save_models(self):
        for net in (
            self.actor,
            self.critic_1,
            self.critic_2,
            self.target_critic_1,
            self.target_critic_2,
        ):
            net.save_checkpoint()

    def load_models(self):
        for net in (
            self.actor,
            self.critic_1,
            self.critic_2,
            self.target_critic_1,
            self.target_critic_2,
        ):
            net.load_checkpoint()


# ----------------------------------------------------------------------------
# BipedalWalker termination helper
# ----------------------------------------------------------------------------

def check_bipedal_done(
    next_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    # BipedalWalker observation index 0 is hull angle.
    hull_angles = np.abs(next_states[:, 0])

    fallen = hull_angles > 0.4
    penalties = fallen.astype(np.float32) * -100.0

    return fallen, penalties


# ----------------------------------------------------------------------------
# MBPO model rollouts
# ----------------------------------------------------------------------------

def generate_model_rollouts(
    env_buffer: ReplayBuffer,
    model_buffer: ReplayBuffer,
    world_model: WorldModel,
    agent: SACAgent,
    rollout_length: int = 1,
    batch_size: int = 256,
    uncertainty_threshold: float = None,
) -> Tuple[float, float]:

    if len(env_buffer) < batch_size:
        return 0.0, 0.0

    states, _, _, _, _, _ = env_buffer.sample_buffer(
        batch_size,
        world_model.device,
    )

    cur_states = states

    all_uncertainties = []
    generated = 0

    for _ in range(rollout_length):
        with T.no_grad():
            actions, _ = agent.actor.sample(cur_states)

            (
                next_states,
                rewards,
                disagreement,
            ) = world_model.predict(
                cur_states,
                actions,
            )

        np_states = cur_states.cpu().numpy()
        np_actions = actions.cpu().numpy()
        np_next_states = next_states.cpu().numpy()
        np_rewards = rewards.squeeze(-1).cpu().numpy()

        disagreement_np = (
            disagreement.squeeze(-1).cpu().numpy()
        )

        all_uncertainties.extend(
            disagreement_np.tolist()
        )

        fallen, penalties = check_bipedal_done(
            np_next_states
        )

        final_rewards = np_rewards + penalties

        keep = np.ones(
            batch_size,
            dtype=np.bool_,
        )

        if uncertainty_threshold is not None:
            keep &= (
                disagreement_np
                <= uncertainty_threshold
            )

        for i in range(batch_size):
            if not keep[i]:
                continue

            model_buffer.store_transition(
                np_states[i],
                np_actions[i],
                final_rewards[i],
                np_next_states[i],
                terminated=bool(fallen[i]),
                truncated=False,
            )
            generated += 1

        # Don't propagate states that have already terminated.
        alive = (
            (~fallen)
            & keep
        )

        if not np.any(alive):
            break

        alive_tensor = T.as_tensor(
            alive,
            dtype=T.bool,
            device=world_model.device,
        )

        cur_states = next_states[alive_tensor]

        # Batch size can shrink after uncertainty/termination filtering.
        batch_size = int(cur_states.shape[0])

    mean_uncertainty = (
        float(np.mean(all_uncertainties))
        if all_uncertainties
        else 0.0
    )

    return float(generated), mean_uncertainty


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train(
    env_id: str = "BipedalWalker-v3",
    n_episodes: int = 3000,
    max_timesteps: int = int(1e6),
    warmup: int = 5000,
    seed: int = 0,
    figure_file: str = "plots/mbpo_bipedalwalker.png",
    plot_every: int = 50,

    # MBPO settings
    model_train_interval: int = 250,
    model_train_steps: int = 10,
    model_rollout_batches: int = 4,
    model_rollout_length: int = 1,
    sac_updates_per_env_step: int = 1,
    real_ratio: float = 0.5,

    # Set to None initially. Once diagnostics look good, try ~0.5-1.0.
    uncertainty_threshold: float = None,
):
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Large real replay buffer.
    env_buffer = ReplayBuffer(
        1_000_000,
        state_dim,
        n_actions,
    )

    # Deliberately small/fresh synthetic buffer.
    model_buffer = ReplayBuffer(
        20_000,
        state_dim,
        n_actions,
    )

    obs_rms = RunningMeanStd(
        shape=(state_dim,)
    )

    world_model = WorldModel(
        state_dim,
        n_actions,
    )

    agent = SACAgent(
        state_dim,
        n_actions,
    )

    normalization_frozen = False

    ext_reward_history = deque(maxlen=100)

    recent_critic_losses = deque(maxlen=200)
    recent_actor_losses = deque(maxlen=200)
    recent_world_model_losses = deque(maxlen=200)
    recent_alpha = deque(maxlen=200)

    episode_numbers = []
    episode_scores = []

    total_timesteps = 0
    best_score = -float("inf")

    try:
        for episode in range(1, n_episodes + 1):
            state, _ = env.reset(
                seed=seed + episode
            )

            if not normalization_frozen:
                obs_rms.update(state)

            done = False
            ep_ext_reward = 0.0

            while not done:
                # ------------------------------------------------------------
                # 1. Real environment interaction
                # ------------------------------------------------------------
                if total_timesteps < warmup:
                    action = env.action_space.sample()
                else:
                    action = agent.choose_action(state)

                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )

                done = terminated or truncated

                if not normalization_frozen:
                    obs_rms.update(next_state)

                env_buffer.store_transition(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    truncated,
                )

                # ------------------------------------------------------------
                # 2. Freeze normalization once warmup is complete
                # ------------------------------------------------------------
                if (
                    not normalization_frozen
                    and total_timesteps + 1 >= warmup
                    and len(env_buffer) >= warmup
                ):
                    world_model.set_normalization(
                        obs_rms.mean,
                        obs_rms.std,
                    )
                    normalization_frozen = True

                    print(
                        "World-model normalization frozen "
                        f"after {len(env_buffer)} real transitions."
                    )

                # ------------------------------------------------------------
                # 3. Periodically retrain dynamics and refresh model buffer
                # ------------------------------------------------------------
                if (
                    normalization_frozen
                    and len(env_buffer) >= 256
                    and total_timesteps >= warmup
                    and total_timesteps % model_train_interval == 0
                ):
                    for _ in range(model_train_steps):
                        wm_loss = world_model.train_step(
                            env_buffer,
                            batch_size=256,
                        )
                        recent_world_model_losses.append(
                            wm_loss
                        )

                    # IMPORTANT:
                    # Synthetic transitions are tied to the current model.
                    # Throw away old-model transitions before generating new ones.
                    model_buffer.clear()

                    total_generated = 0
                    uncertainty_values = []

                    for _ in range(model_rollout_batches):
                        generated, uncertainty = (
                            generate_model_rollouts(
                                env_buffer,
                                model_buffer,
                                world_model,
                                agent,
                                rollout_length=model_rollout_length,
                                batch_size=256,
                                uncertainty_threshold=uncertainty_threshold,
                            )
                        )

                        total_generated += generated
                        uncertainty_values.append(
                            uncertainty
                        )

                    mean_uncertainty = (
                        float(np.mean(uncertainty_values))
                        if uncertainty_values
                        else 0.0
                    )

                    print(
                        f"[MBPO refresh] real={len(env_buffer)} "
                        f"synthetic={len(model_buffer)} "
                        f"generated={total_generated} "
                        f"ensemble_std={mean_uncertainty:.4f}"
                    )

                # ------------------------------------------------------------
                # 4. SAC updates
                # ------------------------------------------------------------
                if (
                    normalization_frozen
                    and len(env_buffer) >= 256
                    and len(model_buffer) >= 128
                ):
                    for _ in range(
                        sac_updates_per_env_step
                    ):
                        stats = agent.learn(
                            env_buffer,
                            model_buffer,
                            batch_size=256,
                            real_ratio=real_ratio,
                        )

                        if stats:
                            recent_critic_losses.append(
                                stats["critic_loss"]
                            )
                            recent_actor_losses.append(
                                stats["actor_loss"]
                            )
                            recent_alpha.append(
                                stats["alpha"]
                            )

                state = next_state
                ep_ext_reward += reward
                total_timesteps += 1

                if total_timesteps >= max_timesteps:
                    done = True

            # ------------------------------------------------------------
            # Episode logging
            # ------------------------------------------------------------
            ext_reward_history.append(
                ep_ext_reward
            )

            episode_numbers.append(episode)
            episode_scores.append(
                ep_ext_reward
            )

            avg_reward = np.mean(
                ext_reward_history
            )

            if (
                episode >= 50
                and avg_reward > best_score
            ):
                best_score = avg_reward

                print(
                    f"--> Saved best model "
                    f"(avg100={best_score:.2f})"
                )

                agent.save_models()
                world_model.save_checkpoint()

            if episode % plot_every == 0:
                plot_learning_curve(
                    episode_numbers,
                    episode_scores,
                    figure_file,
                )

                agent.save_models()
                world_model.save_checkpoint()

            if episode % 10 == 0:
                avg_wm_loss = (
                    np.mean(
                        recent_world_model_losses
                    )
                    if recent_world_model_losses
                    else 0.0
                )

                avg_critic_loss = (
                    np.mean(
                        recent_critic_losses
                    )
                    if recent_critic_losses
                    else 0.0
                )

                avg_actor_loss = (
                    np.mean(
                        recent_actor_losses
                    )
                    if recent_actor_losses
                    else 0.0
                )

                avg_alpha = (
                    np.mean(recent_alpha)
                    if recent_alpha
                    else 0.0
                )

                print(
                    f"Episode {episode:5d} | "
                    f"Steps {total_timesteps:8d} | "
                    f"Reward {ep_ext_reward:8.2f} | "
                    f"Avg100 {avg_reward:8.2f} | "
                    f"WM {avg_wm_loss:8.4f} | "
                    f"Critic {avg_critic_loss:9.4f} | "
                    f"Actor {avg_actor_loss:9.4f} | "
                    f"Alpha {avg_alpha:7.4f}"
                )

            if total_timesteps >= max_timesteps:
                break

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. "
            "Saving current model state..."
        )

    plot_learning_curve(
        episode_numbers,
        episode_scores,
        figure_file,
    )

    env.close()

    agent.save_models()
    world_model.save_checkpoint()

    print(
        f"Saved weights to tmp/mbpo and graph to "
        f"{figure_file}"
    )


if __name__ == "__main__":
    train(
        env_id="BipedalWalker-v3",
        n_episodes=3000,
        max_timesteps=int(1e6),
        warmup=5000,
        seed=0,
        figure_file="plots/mbpo_bipedalwalker.png",
        plot_every=50,

        # ------------------------------------------------------------
        # Conservative MBPO settings.
        # ------------------------------------------------------------
        model_train_interval=250,
        model_train_steps=10,

        # Generate 4 x 256 = 1024 fresh synthetic transitions
        # after each model refresh.
        model_rollout_batches=4,

        # Keep one-step rollouts until the dynamics model is reliable.
        model_rollout_length=1,

        # One SAC update per real environment step.
        sac_updates_per_env_step=1,

        # 50% real / 50% model-generated transitions.
        real_ratio=0.5,

        # Start without filtering. After inspecting ensemble_std,
        # try a threshold such as 0.5-1.0.
        uncertainty_threshold=None,
    )