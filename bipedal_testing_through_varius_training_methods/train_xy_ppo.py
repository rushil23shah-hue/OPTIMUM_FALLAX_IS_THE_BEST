import os
import gymnasium as gym
from stable_baselines3 import PPO

class XYRewardWrapper(gym.Wrapper):
    def __init__(self,env,x_factor=6,y_factor=0.3,time_penalty=0.08):
        super().__init__(env)
        self.x_factor=x_factor
        self.y_factor=y_factor
        self.time_penalty=time_penalty
        self.prev_x=0
        self.start_y=0

    def get_x(self):
        return self.env.unwrapped.hull.position.x

    def get_y(self):
        return self.env.unwrapped.hull.position.y

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.prev_x=self.get_x()
        self.start_y=self.get_y()
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)

        current_x=self.get_x()
        current_y=self.get_y()

        x_progress=current_x-self.prev_x
        self.prev_x=current_x

        height_error=abs(current_y-self.start_y)

        proxy_reward=reward+(x_progress*self.x_factor)-(height_error*self.y_factor)-self.time_penalty

        return obs,proxy_reward,terminated,truncated,info

os.makedirs("models",exist_ok=True)

env=gym.make("BipedalWalker-v3")
env=XYRewardWrapper(env,x_factor=6,y_factor=0.3,time_penalty=0.08)

model=PPO.load("models/xy_ppo_bipedalwalker",env=env)

model.learn(total_timesteps=500000)

model.save("models/xy_ppo_bipedalwalker_refined")

env.close()