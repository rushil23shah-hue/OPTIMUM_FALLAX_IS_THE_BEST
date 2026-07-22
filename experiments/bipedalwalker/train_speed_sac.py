import os
import gymnasium as gym
from stable_baselines3 import SAC

class SpeedRewardWrapper(gym.Wrapper):
    def __init__(self,env,speed_factor=17):
        super().__init__(env)
        self.speed_factor=speed_factor
        self.prev_x=0

    def get_x(self):
        return self.env.unwrapped.hull.position.x

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.prev_x=self.get_x()
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)

        current_x=self.get_x()
        velocity=current_x-self.prev_x
        self.prev_x=current_x

        proxy_reward=reward+(velocity*self.speed_factor)

        return obs,proxy_reward,terminated,truncated,info

env=gym.make("BipedalWalker-v3")
env=SpeedRewardWrapper(env,speed_factor=17)

model=SAC.load("models/Baseline/sac_500000",env=env)

model.learn(total_timesteps=100000)

model.save("models/Experiments/fwdspeed_reward01")

env.close()