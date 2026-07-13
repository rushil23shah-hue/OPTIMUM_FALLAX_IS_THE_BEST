import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO

contact_factor=0.08
switch_factor=0.3
both_penalty=0.08
air_penalty=0.04

os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3",render_mode="human")
model=PPO.load("models/foot_hero_ppo_bipedalwalker")

def get_x():
    return env.unwrapped.hull.position.x

def get_angle():
    return env.unwrapped.hull.angle

def get_angular_velocity():
    return env.unwrapped.hull.angularVelocity

def get_foot_contacts():
    left_contact=1 if env.unwrapped.legs[1].ground_contact else 0
    right_contact=1 if env.unwrapped.legs[3].ground_contact else 0
    return left_contact,right_contact

obs,info=env.reset()

start_x=get_x()
prev_x=start_x
max_x=start_x
prev_single_foot=None

done=False
steps=0
true_reward=0
proxy_reward=0

left_contact_count=0
right_contact_count=0
single_contact_count=0
both_contact_count=0
no_contact_count=0
switch_count=0

rows=[]

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    current_x=get_x()
    speed=current_x-prev_x
    prev_x=current_x
    max_x=max(max_x,current_x)

    angle=abs(get_angle())
    angular_velocity=abs(get_angular_velocity())

    left_contact,right_contact=get_foot_contacts()
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
        if prev_single_foot is not None and current_single_foot!=prev_single_foot:
            alternation=1
        prev_single_foot=current_single_foot

    foot_gait_reward=(single_contact*contact_factor)+(alternation*switch_factor)-(both_contact*both_penalty)-(no_contact*air_penalty)

    current_proxy_reward=reward+foot_gait_reward

    true_reward+=reward
    proxy_reward+=current_proxy_reward

    left_contact_count+=left_contact
    right_contact_count+=right_contact
    single_contact_count+=single_contact
    both_contact_count+=both_contact
    no_contact_count+=no_contact
    switch_count+=alternation

    steps+=1
    done=terminated or truncated

    rows.append([
        steps,
        round(max_x-start_x,3),
        round(speed,3),
        round(angle,3),
        round(angular_velocity,3),
        left_contact,
        right_contact,
        single_contact,
        both_contact,
        no_contact,
        alternation,
        round(true_reward,3),
        round(proxy_reward,3)
    ])

final_x=get_x()
distance=max_x-start_x

env.close()

with open("logs/foot_hero_test_values.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["step","distance","speed","angle","angular_velocity","left_contact","right_contact","single_contact","both_contact","no_contact","alternation","true_reward","proxy_reward"])
    writer.writerows(rows)

print("true_reward:",round(true_reward,2))
print("proxy_reward:",round(proxy_reward,2))
print("distance:",round(distance,2))
print("final_x:",round(final_x,2))
print("steps:",steps)
print("left_contact_count:",left_contact_count)
print("right_contact_count:",right_contact_count)
print("single_contact_count:",single_contact_count)
print("both_contact_count:",both_contact_count)
print("no_contact_count:",no_contact_count)
print("switch_count:",switch_count)
print("csv saved at logs/foot_hero_test_values.csv")