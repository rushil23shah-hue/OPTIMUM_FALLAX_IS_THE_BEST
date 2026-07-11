import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import gymnasium as gym # type: ignore
import random 
from collections import deque

class ReplayBuffer :
    def __init__(self,buffer_size,batch_size) :
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self,state,action,reward,next_state,done) :
        self.memory.append((state,action,reward,next_state,done))
    
    def sample(self) :
        experiences = random.sample(self.memory,k = self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.FloatTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences])
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.FloatTensor([e[4] for e in experiences])
        
        return states,actions,rewards,next_states,dones
    
class SoftQNetwork(nn.Module) :
    def __init__(self,state_dim,action_dim,hidden_dim=256) :
        super(SoftQNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim = 1)
        
    def forward(self,state,action):
        x = torch.cat([state,action],1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class PolicyNetwork(nn.module) : 
    def __init__ (self,state_dim,action_dim,hidden_dim=256,log_std_min=20,log_std_max=2):
        super(PolicyNetwork,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim,action_dim)
        self.log_std_linear = nn.Linear(hidden_dim,action_dim)
        
        #constraining log std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self,state) :
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)
        
        return mean,log_std
    
    def sample(self,state) : 
        mean,log_std = self.forward(state)
        std = log_std.exp()
        
        #Reparametrization 
        noise = torch.randn_link(mean)
        action = mean + std*noise 
        
        #Compute log probability 
        
        log_prob = -0.5 *((noise**2)+2*log_std + np.log(2*np.pi))
        log_prob = log_prob.sum(dim=1,keepdim=True)
        
        #Tanh squashing 
        action = torch.tanh(action)
        log_prob -= torch.log(1-action.pow(2)+ 1e-6).sum(dim=1,keepdim=True) 
        
        return action,log_prob
    
class SACagent : 
    def __init__(self,state_dim,action_dim,
                 lr = 3e-4,
                 gamma=0.99,
                 tau=0.005,
                 target_entropy = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #policy network 
        self.policy_net = PolicyNetwork(state_dim,action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(),lr=lr)
        
        # Q networks(two for reduced overestimation bias)
        
        self.q_net1 = SoftQNetwork(state_dim,action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(state_dim,action_dim).to(self.device)
        self.policy_optimizer1 = optim.Adam(self.q_net1.parameters(),lr=lr)
        self.policy_optimizer2 = optim.Adam(self.q_net2.parameters(),lr=lr)
        
        #Target Q networks 
        self.target_q_net1 = SoftQNetwork(state_dim,action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(state_dim,action_dim).to(self.device)
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        self.gamma = gamma 
        self.tau = tau 
        
        #Temperature Parameter 
        if target_entropy is None : 
            self.target_entropy = -action_dim 
        else : 
            self.target_entropy = target_entropy
            
        self.log_alpha = torch.zeros(1,requires_grad=True,device = self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha],lr=lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size = 100000,batch_size = 256)
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad