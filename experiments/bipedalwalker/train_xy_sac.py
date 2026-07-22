import gymnasium as gym 
from stable_baselines3 import SAC
import os 

class RewardWrapper(gym.Wrapper):
    def __init__(self,env,x_factor = 6 ,y_factor = 0.3 ,time_penalty=0.08) : 
        super().__init__(env)
        self.prev_x = 0.0
        self.start_y = 0.0
        self.x_factor = x_factor
        self.y_factor = y_factor 
        self.time_penalty = time_penalty
    
    def get_x(self):
        return self.env.unwrapped.hull.position.x
    
    def get_y(self):
        return self.env.unwrapped.hull.position.y
    
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.prev_x = self.get_x()
        self.start_y = self.get_y()
        return obs,info 
    
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        
        current_x = self.get_x()
        current_y = self.get_y()
        
        dx = (current_x-self.prev_x)
        self.prev_x = current_x
        
        height_error = abs(current_y - self.start_y)
        
        proxy_reward = reward + (self.x_factor*dx )- (self.y_factor*height_error) - self.time_penalty
        
        info["x_progress"] = dx
        info["height_error"] = height_error
        info["proxy_reward"] = proxy_reward
        
        return obs,proxy_reward,terminated,truncated,info 
    
env = gym.make("BipedalWalker-v3")
env = RewardWrapper(env,x_factor=6,y_factor=0.3,time_penalty=0.08)

model = SAC.load("models/Baseline/sac_500000",env=env)

model.learn(total_timesteps= 100000,progress_bar=True)
model.save("models/Experiments/xy_reward01")

env.close()
