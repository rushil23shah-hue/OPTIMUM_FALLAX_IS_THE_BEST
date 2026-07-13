import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO

model_path="models/speed_ppo_bipedalwalker"

speed_factor=10
distance_factor=6
time_penalty=0.2

true_x_factor=6
true_y_factor=0.3
true_angle_factor=3
true_angular_velocity_factor=0.3
true_time_penalty=0.05

os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3",render_mode="human")
model=PPO.load(model_path)

def get_x():
    return env.unwrapped.hull.position.x

def get_y():
    return env.unwrapped.hull.position.y

def get_angle():
    return env.unwrapped.hull.angle

def get_angular_velocity():
    return env.unwrapped.hull.angularVelocity

obs,info=env.reset()

start_x=get_x()
prev_x=start_x
max_x=start_x
start_y=get_y()

done=False
steps=0
true_reward=0
proxy_reward=0
rows=[]

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    current_x=get_x()
    current_y=get_y()

    x_progress=current_x-prev_x
    speed=x_progress
    prev_x=current_x
    max_x=max(max_x,current_x)

    distance=max_x-start_x
    height_error=abs(current_y-start_y)
    angle_error=abs(get_angle())
    angular_velocity_error=abs(get_angular_velocity())

    current_true_reward=reward+(x_progress*true_x_factor)-(height_error*true_y_factor)-(angle_error*true_angle_factor)-(angular_velocity_error*true_angular_velocity_factor)-true_time_penalty

    current_proxy_reward=reward+(speed*speed_factor)+(x_progress*distance_factor)-time_penalty

    true_reward+=current_true_reward
    proxy_reward+=current_proxy_reward

    steps+=1
    done=terminated or truncated

    rows.append([
        steps,
        round(distance,3),
        round(speed,3),
        round(time_penalty*steps,3),
        round(true_reward,3),
        round(proxy_reward,3)
    ])

env.close()

with open("logs/speed_distance_time_values.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["step","distance","speed","cumulative_time_penalty","true_reward","proxy_reward"])
    writer.writerows(rows)

print("steps:",steps)
print("distance:",round(distance,2))
print("true_reward:",round(true_reward,2))
print("proxy_reward:",round(proxy_reward,2))
print("csv saved at logs/speed_distance_time_values.csv")