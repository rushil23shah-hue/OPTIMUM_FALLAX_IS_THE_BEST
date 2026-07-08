import gymnasium as gym
from stable_baselines3 import PPO

env=gym.make("BipedalWalker-v3",render_mode="human")

model=PPO.load("models/ppo_bipedalwalker-1000000")

obs,info=env.reset()
done=False
total_reward=0

while not done:
    action,_=model.predict(obs,deterministic=True)
    obs,reward,terminated,truncated,info=env.step(action)

    total_reward+=reward
    done=terminated or truncated

env.close()

print("reward:",round(total_reward,2))