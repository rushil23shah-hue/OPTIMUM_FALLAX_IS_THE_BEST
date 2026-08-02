import gymnasium as gym
from stable_baselines3 import PPO

x_factor=6
y_factor=0.3
time_penalty=0.05

env=gym.make("BipedalWalker-v3",render_mode="human")
model=PPO.load("models/xy_ppo_bipedalwalker_refined")

def get_x():
    return env.unwrapped.hull.position.x

def get_y():
    return env.unwrapped.hull.position.y

obs,info=env.reset()

start_x=get_x()
prev_x=start_x
max_x=start_x
start_y=get_y()

done=False
true_reward=0
proxy_reward=0
steps=0

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    current_x=get_x()
    current_y=get_y()

    x_progress=current_x-prev_x
    prev_x=current_x

    height_error=abs(current_y-start_y)

    true_reward+=reward
    proxy_reward+=reward+(x_progress*x_factor)-(height_error*y_factor)-time_penalty

    max_x=max(max_x,current_x)
    steps+=1
    done=terminated or truncated

env.close()

distance=max_x-start_x
final_x=get_x()

print("true_reward:",round(true_reward,2))
print("proxy_reward:",round(proxy_reward,2))
print("distance:",round(distance,2))
print("final_x:",round(final_x,2))
print("steps:",steps)