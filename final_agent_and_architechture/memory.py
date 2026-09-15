import torch
import numpy as np

class VectorReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device="cpu"):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]).to(device),
            torch.FloatTensor(self.actions[ind]).to(device),
            torch.FloatTensor(self.rewards[ind]).to(device),
            torch.FloatTensor(self.next_states[ind]).to(device),
            torch.FloatTensor(self.dones[ind]).to(device)
        )

    def sample_states(self, batch_size, device="cpu"):
        ind = np.random.randint(0, self.size, size=batch_size)
        return torch.FloatTensor(self.states[ind]).to(device)