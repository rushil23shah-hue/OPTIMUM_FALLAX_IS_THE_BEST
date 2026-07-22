import gymnasium as gym
from stable_baselines3 import SAC

class AngleRewardWrapper(gym.Wrapper):
    def __init__(self,env,angle_factor=3):
        super().__init__(env)
        self.angle_factor = angle_factor 
        
    def get_angle(self):
        return self.env.unwrapped.hull.angle
    
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.angle_sum = 0
        self.step_count = 0
        return obs,info 
        
        
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        angle = abs(self.get_angle())
        self.angle_sum += angle
        self.step_count += 1
        proxy_reward = reward - (self.angle_factor* angle)
        info["avg_angle"] = self.angle_sum/ self.step_count
        return obs,proxy_reward,terminated,truncated,info 
    
env = gym.make("BipedalWalker-v3")
env = AngleRewardWrapper(env,angle_factor =3)

model = SAC.load("models/Baseline/sac_500000",env=env)

model.learn(total_timesteps=100000,progress_bar=True)
model.save("models/Experiments/angle_penalty01_factor=3")
env.close()