import gymnasium as gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from envs.E00_reward_sep import BipedalWalker

env = BipedalWalker(render_mode="human")
#env = gym.make("BipedalWalker-v3",render_mode ="human")
#model = PPO.load("models/E00/model_ppo")
model = PPO.load("models/Baseline/ppo_1000000")
observation,info = env.reset()
terminated = False
truncated = False 
total_rewards = 0 

while not (terminated or truncated) : 
    action,_ = model.predict(observation,deterministic = True )
    observation,reward,terminated,truncated,info = env.step(action)
    total_rewards += reward
    
print(total_rewards)
env.close()