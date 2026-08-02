import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class AngleXYRewardWrapper(gym.Wrapper):
    def __init__(self,env,x_factor=6,y_factor=0.3,angle_factor=3,angular_velocity_factor=0.3,time_penalty=0.05):
        super().__init__(env)
        self.x_factor=x_factor
        self.y_factor=y_factor
        self.angle_factor=angle_factor
        self.angular_velocity_factor=angular_velocity_factor
        self.time_penalty=time_penalty
        self.prev_x=0
        self.start_y=0

    def get_x(self):
        return self.env.unwrapped.hull.position.x

    def get_y(self):
        return self.env.unwrapped.hull.position.y

    def get_angle(self):
        return self.env.unwrapped.hull.angle

    def get_angular_velocity(self):
        return self.env.unwrapped.hull.angularVelocity

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.prev_x=self.get_x()
        self.start_y=self.get_y()
        self.angle_sum=0
        self.angular_velocity_sum=0
        self.height_error_sum=0
        self.x_progress_sum=0
        self.step_count=0
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)

        current_x=self.get_x()
        current_y=self.get_y()

        x_progress=current_x-self.prev_x
        self.prev_x=current_x

        height_error=abs(current_y-self.start_y)
        angle_error=abs(self.get_angle())
        angular_velocity_error=abs(self.get_angular_velocity())

        self.angle_sum+=angle_error
        self.angular_velocity_sum+=angular_velocity_error
        self.height_error_sum+=height_error
        self.x_progress_sum+=x_progress
        self.step_count+=1

        proxy_reward=reward+(x_progress*self.x_factor)-(height_error*self.y_factor)-(angle_error*self.angle_factor)-(angular_velocity_error*self.angular_velocity_factor)-self.time_penalty

        info["avg_angle"]=round(self.angle_sum/self.step_count,3)
        info["avg_angular_velocity"]=round(self.angular_velocity_sum/self.step_count,3)
        info["avg_height_error"]=round(self.height_error_sum/self.step_count,3)
        info["total_x_progress"]=round(self.x_progress_sum,2)
        info["episode_steps"]=self.step_count

        return obs,proxy_reward,terminated,truncated,info

os.makedirs("models",exist_ok=True)
os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3")

env=AngleXYRewardWrapper(
    env,
    x_factor=6,
    y_factor=0.3,
    angle_factor=3,
    angular_velocity_factor=0.3,
    time_penalty=0.05
)

env=Monitor(
    env,
    "logs/angle_xy_monitor.csv",
    info_keywords=("avg_angle","avg_angular_velocity","avg_height_error","total_x_progress","episode_steps")
)

model=PPO.load("models/xy_ppo_bipedalwalker_refined",env=env)

model.learn(total_timesteps=300000)

model.save("models/angle_xy_ppo_bipedalwalker")

env.close()