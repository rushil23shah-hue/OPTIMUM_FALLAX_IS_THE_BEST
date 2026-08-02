import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class ComJitterRewardWrapper(gym.Wrapper):
    def __init__(self,env,x_factor=6,y_factor=0.3,angle_factor=3,angular_velocity_factor=0.3,time_penalty=0.05,jitter_factor=6):
        super().__init__(env)
        self.x_factor=x_factor
        self.y_factor=y_factor
        self.angle_factor=angle_factor
        self.angular_velocity_factor=angular_velocity_factor
        self.time_penalty=time_penalty
        self.jitter_factor=jitter_factor
        self.prev_x=0
        self.prev_com_x=0
        self.prev_com_y=0
        self.prev_x_velocity=0
        self.start_y=0

    def get_x(self):
        return self.env.unwrapped.hull.position.x

    def get_y(self):
        return self.env.unwrapped.hull.position.y

    def get_angle(self):
        return self.env.unwrapped.hull.angle

    def get_angular_velocity(self):
        return self.env.unwrapped.hull.angularVelocity

    def get_center_of_mass(self):
        bodies=[self.env.unwrapped.hull]+list(self.env.unwrapped.legs)
        total_mass=0
        com_x=0
        com_y=0

        for body in bodies:
            mass=getattr(body,"mass",1)
            com_x+=body.position.x*mass
            com_y+=body.position.y*mass
            total_mass+=mass

        return com_x/total_mass,com_y/total_mass

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)

        self.prev_x=self.get_x()
        self.start_y=self.get_y()

        self.prev_com_x,self.prev_com_y=self.get_center_of_mass()
        self.prev_x_velocity=0

        self.com_jitter_sum=0
        self.vertical_jitter_sum=0
        self.speed_jitter_sum=0
        self.x_progress_sum=0
        self.height_error_sum=0
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

        com_x,com_y=self.get_center_of_mass()

        x_velocity=com_x-self.prev_com_x
        vertical_jitter=abs(com_y-self.prev_com_y)
        speed_jitter=abs(x_velocity-self.prev_x_velocity)

        self.prev_com_x=com_x
        self.prev_com_y=com_y
        self.prev_x_velocity=x_velocity

        com_jitter=vertical_jitter+(speed_jitter*0.5)

        stable_reward=reward+(x_progress*self.x_factor)-(height_error*self.y_factor)-(angle_error*self.angle_factor)-(angular_velocity_error*self.angular_velocity_factor)-self.time_penalty

        proxy_reward=stable_reward-(com_jitter*self.jitter_factor)

        self.com_jitter_sum+=com_jitter
        self.vertical_jitter_sum+=vertical_jitter
        self.speed_jitter_sum+=speed_jitter
        self.x_progress_sum+=x_progress
        self.height_error_sum+=height_error
        self.step_count+=1

        info["avg_com_jitter"]=round(self.com_jitter_sum/self.step_count,4)
        info["avg_vertical_jitter"]=round(self.vertical_jitter_sum/self.step_count,4)
        info["avg_speed_jitter"]=round(self.speed_jitter_sum/self.step_count,4)
        info["total_x_progress"]=round(self.x_progress_sum,2)
        info["avg_height_error"]=round(self.height_error_sum/self.step_count,3)
        info["episode_steps"]=self.step_count

        return obs,proxy_reward,terminated,truncated,info

os.makedirs("models",exist_ok=True)
os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3")

env=ComJitterRewardWrapper(
    env,
    x_factor=6,
    y_factor=0.3,
    angle_factor=3,
    angular_velocity_factor=0.3,
    time_penalty=0.05,
    jitter_factor=6
)

env=Monitor(
    env,
    "logs/com_jitter_monitor.csv",
    info_keywords=("avg_com_jitter","avg_vertical_jitter","avg_speed_jitter","total_x_progress","avg_height_error","episode_steps")
)

model=PPO.load("models/com_jitter_extra_ppo_bipedalwalker",env=env)

model.learn(total_timesteps=200000)

model.save("models/com_jitter_extra_ppo_bipedalwalker")

env.close()