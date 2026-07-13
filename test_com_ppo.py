import os
import csv
import gymnasium as gym
from stable_baselines3 import PPO

x_factor=6
y_factor=0.3
angle_factor=3
angular_velocity_factor=0.3
time_penalty=0.05
jitter_factor=2

os.makedirs("logs",exist_ok=True)

env=gym.make("BipedalWalker-v3",render_mode="human")
model=PPO.load("models/com_jitter_ppo_bipedalwalker")

def get_x():
    return env.unwrapped.hull.position.x

def get_y():
    return env.unwrapped.hull.position.y

def get_angle():
    return env.unwrapped.hull.angle

def get_angular_velocity():
    return env.unwrapped.hull.angularVelocity

def get_center_of_mass():
    bodies=[env.unwrapped.hull]+list(env.unwrapped.legs)
    total_mass=0
    com_x=0
    com_y=0

    for body in bodies:
        mass=getattr(body,"mass",1)
        com_x+=body.position.x*mass
        com_y+=body.position.y*mass
        total_mass+=mass

    return com_x/total_mass,com_y/total_mass

obs,info=env.reset()

start_x=get_x()
prev_x=start_x
max_x=start_x
start_y=get_y()

prev_com_x,prev_com_y=get_center_of_mass()
prev_x_velocity=0

done=False
steps=0
true_reward=0
proxy_reward=0

com_jitter_sum=0
vertical_jitter_sum=0
speed_jitter_sum=0

rows=[]

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    current_x=get_x()
    current_y=get_y()

    x_progress=current_x-prev_x
    prev_x=current_x
    max_x=max(max_x,current_x)

    height_error=abs(current_y-start_y)
    angle_error=abs(get_angle())
    angular_velocity_error=abs(get_angular_velocity())

    com_x,com_y=get_center_of_mass()

    x_velocity=com_x-prev_com_x
    vertical_jitter=abs(com_y-prev_com_y)
    speed_jitter=abs(x_velocity-prev_x_velocity)

    prev_com_x=com_x
    prev_com_y=com_y
    prev_x_velocity=x_velocity

    com_jitter=vertical_jitter+(speed_jitter*0.5)

    current_true_reward=reward+(x_progress*x_factor)-(height_error*y_factor)-(angle_error*angle_factor)-(angular_velocity_error*angular_velocity_factor)-time_penalty
    current_proxy_reward=current_true_reward-(com_jitter*jitter_factor)

    true_reward+=current_true_reward
    proxy_reward+=current_proxy_reward

    com_jitter_sum+=com_jitter
    vertical_jitter_sum+=vertical_jitter
    speed_jitter_sum+=speed_jitter

    steps+=1
    done=terminated or truncated

    rows.append([
        steps,
        round(max_x-start_x,3),
        round(x_progress,4),
        round(com_x,4),
        round(com_y,4),
        round(vertical_jitter,5),
        round(speed_jitter,5),
        round(com_jitter,5),
        round(height_error,4),
        round(angle_error,4),
        round(angular_velocity_error,4),
        round(true_reward,3),
        round(proxy_reward,3)
    ])

env.close()

distance=max_x-start_x

with open("logs/com_jitter_extra_test_values.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow([
        "step",
        "distance",
        "x_progress",
        "com_x",
        "com_y",
        "vertical_jitter",
        "speed_jitter",
        "com_jitter",
        "height_error",
        "angle_error",
        "angular_velocity_error",
        "true_reward",
        "proxy_reward"
    ])
    writer.writerows(rows)

print("reward factors used:")
print("x_factor:",x_factor)
print("y_factor:",y_factor)
print("angle_factor:",angle_factor)
print("angular_velocity_factor:",angular_velocity_factor)
print("time_penalty:",time_penalty)
print("jitter_factor:",jitter_factor)

print("steps:",steps)
print("distance:",round(distance,2))
print("true_reward:",round(true_reward,2))
print("proxy_reward:",round(proxy_reward,2))
print("avg_com_jitter:",round(com_jitter_sum/steps,5))
print("avg_vertical_jitter:",round(vertical_jitter_sum/steps,5))
print("avg_speed_jitter:",round(speed_jitter_sum/steps,5))
print("csv saved at logs/com_jitter_extra_test_values.csv")