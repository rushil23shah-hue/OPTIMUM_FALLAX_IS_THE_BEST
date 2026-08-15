from utilis import Actor, Double_Q_Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class LegacySACContinuous:
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

		if self.adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
			# We learn log_alpha instead of alpha to ensure alpha>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

	def select_action(self, state, deterministic):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			a, _ = self.actor(state, deterministic, with_logprob=False)
		return a.cpu().numpy()[0]

	def train(self,):
		s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
			target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = r + (~dw) * self.gamma * (target_Q - self.alpha * log_pi_a_next) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze critic so you don't waste computational effort computing gradients for them when update actor
		for params in self.q_critic.parameters(): params.requires_grad = False

		a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
		current_Q1, current_Q2 = self.q_critic(s, a)
		Q = torch.min(current_Q1, current_Q2)

		a_loss = (self.alpha * log_pi_a - Q).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		for params in self.q_critic.parameters(): params.requires_grad = True

		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# We learn log_alpha instead of alpha to ensure alpha>0
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class LegacyReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size, dvc):
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
		self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		#每次只放入一个时刻的数据
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = torch.from_numpy(a).to(self.dvc) # Note that a is numpy.array
		self.r[self.ptr] = r
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size #存满了又重头开始存
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
"""Soft Actor-Critic implementation used by main.py."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[indices], device=device),
            torch.as_tensor(self.action[indices], device=device),
            torch.as_tensor(self.reward[indices], device=device),
            torch.as_tensor(self.next_state[indices], device=device),
            torch.as_tensor(self.done[indices], device=device),
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)

    def forward(self, state, deterministic=False, with_logprob=True):
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-20, 2)
        distribution = Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = None
        if with_logprob:
            # Change-of-variables correction for the tanh squashing function.
            log_prob = distribution.log_prob(pre_tanh).sum(dim=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(dim=-1, keepdim=True)
        return action, log_prob


class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, width):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = self._make_q(input_dim, width)
        self.q2 = self._make_q(input_dim, width)

    @staticmethod
    def _make_q(input_dim, width):
        return nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        return self.q1(state_action), self.q2(state_action)


class SAC_countinuous:
    def __init__(self, state_dim, action_dim, gamma, net_width, a_lr, c_lr,
                 batch_size, alpha, adaptive_alpha, dvc, **_):
        self.device = dvc
        self.gamma = gamma
        self.batch_size = batch_size
        self.adaptive_alpha = adaptive_alpha
        self.actor = Actor(state_dim, action_dim, net_width).to(self.device)
        self.critic = DoubleCritic(state_dim, action_dim, net_width).to(self.device)
        self.critic_target = DoubleCritic(state_dim, action_dim, net_width).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.tau = 0.005

        if adaptive_alpha:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(alpha), device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=c_lr)
        else:
            self.alpha = alpha

    def select_action(self, state, deterministic):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state, deterministic, with_logprob=False)
        return action.cpu().numpy()[0]

    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
        alpha = self.log_alpha.exp().detach() if self.adaptive_alpha else self.alpha

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state, deterministic=False, with_logprob=True)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target = reward + self.gamma * (1 - done) * target_q

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor(state, deterministic=False, with_logprob=True)
        q1_new, q2_new = self.critic(state, new_action)
        actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_parameter.mul_(1 - self.tau).add_(self.tau * parameter)

    def save(self, env_name, step):
        os.makedirs('model', exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join('model', f'{env_name}_{step}k_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join('model', f'{env_name}_{step}k_critic.pt'))

    def load(self, env_name, step):
        actor_path = os.path.join('model', f'{env_name}_{step}k_actor.pt')
        critic_path = os.path.join('model', f'{env_name}_{step}k_critic.pt')
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
