import gymnasium as gym 
from stable_baselines3 import SAC

class LinearVelocityRewardWrapper(gym.Wrapper):
    def __init__(self,env,speed_factor=10):
        super().__init__(env)
        self.speed_factor = speed_factor
        
    def get_velocity(self):
        return self.env.unwrapped.hull.linearVelocity.x
    
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        return obs,info 
    
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        speed_x = self.get_velocity()
        proxy_reward = reward + self.speed_factor*speed_x
        info["forward_velocity"] = speed_x
        return obs,proxy_reward,terminated,truncated,info 
    
env = gym.make("BipedalWalker-v3")
env = LinearVelocityRewardWrapper(env,speed_factor = 10)
model = SAC.load("models/Baseline/sac_500000",env = env)

model.learn(total_timesteps=100000,progress_bar=True)
model.save("models/Experiments/LinearVelocity_reward01_factor=10")

env.close()

        