import gymnasium as gym
from stable_baselines3 import PPO

x_factor=6
y_factor=0.3
angle_factor=3
angular_velocity_factor=0.3
time_penalty=0.05

env=gym.make("BipedalWalker-v3",render_mode="human")
model=PPO.load("models/angle_xy_ppo_bipedalwalker")

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
true_reward=0
proxy_reward=0
steps=0

angle_sum=0
angular_velocity_sum=0
height_error_sum=0
x_progress_sum=0

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    current_x=get_x()
    current_y=get_y()

    x_progress=current_x-prev_x
    prev_x=current_x

    height_error=abs(current_y-start_y)
    angle_error=abs(get_angle())
    angular_velocity_error=abs(get_angular_velocity())

    true_reward+=reward
    proxy_reward+=reward+(x_progress*x_factor)-(height_error*y_factor)-(angle_error*angle_factor)-(angular_velocity_error*angular_velocity_factor)-time_penalty

    angle_sum+=angle_error
    angular_velocity_sum+=angular_velocity_error
    height_error_sum+=height_error
    x_progress_sum+=x_progress

    max_x=max(max_x,current_x)
    steps+=1
    done=terminated or truncated

final_x=get_x()
distance=max_x-start_x

env.close()

print("reward factors used:")
print("x_factor:",x_factor)
print("y_factor:",y_factor)
print("angle_factor:",angle_factor)
print("angular_velocity_factor:",angular_velocity_factor)
print("time_penalty:",time_penalty)

print("true_reward:",round(true_reward,2))
print("proxy_reward:",round(proxy_reward,2))
print("distance:",round(distance,2))
print("final_x:",round(final_x,2))
print("steps:",steps)

print("avg_angle:",round(angle_sum/steps,3))
print("avg_angular_velocity:",round(angular_velocity_sum/steps,3))
print("avg_height_error:",round(height_error_sum/steps,3))
print("total_x_progress:",round(x_progress_sum,2))