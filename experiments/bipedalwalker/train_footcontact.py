import gymnasium as gym 
from stable_baselines3 import SAC


class FootAlternationRewardWrapper(gym.Wrapper):
    def __init__(self,env,switch_factor=0.5):
        super().__init__(env)
        self.switch_factor = switch_factor
        self.prev_single_foot = None
        self.switch_count = 0 
        self.step_count = 0
        
        
    def get_foot_contacts(self):
        left_contact = 1 if self.env.unwrapped.legs[1].ground_contact else 0
        right_contact = 1 if self.env.unwrapped.legs[3].ground_contact else 0
        return left_contact,right_contact
        
        
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.prev_single_foot = None 
        self.switch_count = 0
        self.step_count = 0 
        return obs,info 
    
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        
        left_contact,right_contact = self.get_foot_contacts()
        
        current_single_foot = None 
        
        if left_contact and not right_contact :
            current_single_foot = "left"
            
        elif right_contact and not left_contact :
            current_single_foot = "right"
        
        alternation = 0 
        if current_single_foot is not None : 
            if (self.prev_single_foot is not None and current_single_foot != self.prev_single_foot):
                alternation = 1
            self.prev_single_foot = current_single_foot
            
        proxy_reward = reward+self.switch_factor*alternation
        
        self.switch_count += alternation
        self.step_count +=1
        
        info["switch_count"]= self.switch_count
        info["avg_switch_rate"]= self.switch_count/self.step_count
        
        return obs,proxy_reward,terminated,truncated,info
    
env = gym.make("BipedalWalker-v3")
env = FootAlternationRewardWrapper(env,switch_factor = 0.5)

model = SAC.load("../models/Baseline/sac_500000",env=env,)

model.learn(total_timesteps=100000,progress_bar=True,)

model.save("models/Experiments/foot_alteration_reward0.5")

env.close()


        
        
        
        