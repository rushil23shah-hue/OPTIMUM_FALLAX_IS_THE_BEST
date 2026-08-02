import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class FootHeroRewardWrapper(gym.Wrapper):
    def __init__(self,env,contact_factor=0.1,switch_factor=0.4,both_penalty=0.1,air_penalty=0.05):
        super().__init__(env)
        self.contact_factor=contact_factor
        self.switch_factor=switch_factor
        self.both_penalty=both_penalty
        self.air_penalty=air_penalty
        self.prev_single_foot=None

    def get_foot_contacts(self):
        left_contact=1 if self.env.unwrapped.legs[1].ground_contact else 0
        right_contact=1 if self.env.unwrapped.legs[3].ground_contact else 0
        return left_contact,right_contact

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.prev_single_foot=None
        self.left_contact_count=0
        self.right_contact_count=0
        self.single_contact_count=0
        self.both_contact_count=0
        self.no_contact_count=0
        self.switch_count=0
        self.step_count=0
        return obs,info

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)

        left_contact,right_contact=self.get_foot_contacts()
        total_contact=left_contact+right_contact

        single_contact=1 if total_contact==1 else 0
        both_contact=1 if total_contact==2 else 0
        no_contact=1 if total_contact==0 else 0

        current_single_foot=None
        if left_contact==1 and right_contact==0:
            current_single_foot="left"
        elif right_contact==1 and left_contact==0:
            current_single_foot="right"

        alternation=0
        if current_single_foot is not None:
            if self.prev_single_foot is not None and current_single_foot!=self.prev_single_foot:
                alternation=1
            self.prev_single_foot=current_single_foot

        foot_gait_reward=(single_contact*self.contact_factor)+(alternation*self.switch_factor)-(both_contact*self.both_penalty)-(no_contact*self.air_penalty)

        proxy_reward=reward+foot_gait_reward

        self.left_contact_count+=left_contact
        self.right_contact_count+=right_contact
        self.single_contact_count+=single_contact
        self.both_contact_count+=both_contact
        self.no_contact_count+=no_contact
        self.switch_count+=alternation
        self.step_count+=1

        info["left_contact_count"]=self.left_contact_count
        info["right_contact_count"]=self.right_contact_count
        info["single_contact_count"]=self.single_contact_count
        info["both_contact_count"]=self.both_contact_count
        info["no_contact_count"]=self.no_contact_count
        info["switch_count"]=self.switch_count
        info["episode_steps"]=self.step_count

        return obs,proxy_reward,terminated,truncated,info

os.makedirs("models",exist_ok=True)
os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3")

env=FootHeroRewardWrapper(
    env,
    contact_factor=0.1,
    switch_factor=0.4,
    both_penalty=0.1,
    air_penalty=0.05
)

env=Monitor(
    env,
    "logs/foot_hero_monitor.csv",
    info_keywords=("left_contact_count","right_contact_count","single_contact_count","both_contact_count","no_contact_count","switch_count","episode_steps")
)

model=PPO.load("models/angle_xy_ppo_bipedalwalker",env=env)

model.learn(total_timesteps=300000)

model.save("models/foot_hero_ppo_bipedalwalker")

env.close()