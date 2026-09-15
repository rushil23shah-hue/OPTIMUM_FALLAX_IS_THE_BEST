import torch
import torch.nn as nn
import torch.nn.functional as F

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


class DynamicsModel(nn.Module):
    """World Model predicting residual state transitions, reward, and termination logits."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        input_dim = state_dim + action_dim
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.delta_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        feat = self.fc(x)
        delta_state = self.delta_state_head(feat)
        pred_reward = self.reward_head(feat)
        pred_done_logit = self.done_head(feat)
        pred_next_state = state + delta_state
        return pred_next_state, pred_reward, pred_done_logit