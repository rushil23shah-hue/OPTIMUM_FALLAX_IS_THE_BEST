import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def normalize_obs(obs, obs_rms):
    obs_rms.update(np.array([obs]))
    return (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=64):
        super(IntrinsicCuriosityModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state, next_state, action):
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        pred_action = self.inverse_net(torch.cat([phi_s, phi_s_next], dim=-1))
        pred_phi_s_next = self.forward_net(torch.cat([phi_s, action], dim=-1))
        return phi_s_next, pred_phi_s_next, pred_action


class AdversarialAuditor:
    def __init__(self, discrepancy_threshold=15.0):
        self.discrepancy_threshold = discrepancy_threshold
        self.historical_real_returns = []
        self.average_real_return = 0.0

    def update_real_average(self, real_return):
        self.historical_real_returns.append(real_return)
        if len(self.historical_real_returns) > 100:
            self.historical_real_returns.pop(0)
        self.average_real_return = float(np.mean(self.historical_real_returns))

    def is_hallucination(self, dream_return, dream_obs=None, replay_buffer=None):
        if len(self.historical_real_returns) < 5:
            return False

        discrepancy = abs(dream_return - self.average_real_return)
        if discrepancy > self.discrepancy_threshold:
            print(f"[AUDITOR ALARM] Dream predicted return {dream_return:.2f}, reality average is {self.average_real_return:.2f}!", flush=True)
            return True

        # Additional safety: if any dream state is completely out-of-bounds (extreme values)
        if dream_obs is not None and len(dream_obs) > 0:
            if torch.max(torch.abs(dream_obs)) > 25.0: # BipedalWalker safe observation threshold
                print(f"[AUDITOR ALARM] Dream state out of bounds detected!", flush=True)
                return True

        return False