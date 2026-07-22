import gymnasium as gym  # type: ignore
from stable_baselines3 import SAC  # type: ignore
from envs.E00_reward_sep import BipedalWalker

#env = BipedalWalker(render_mode="human")
env = gym.make("BipedalWalker-v3", render_mode = "human")
#model = SAC.load("models/E00/model_sac")
model = SAC.load("models/Baseline/sac_500000")

observation, info = env.reset()

terminated = False
truncated = False 
total_reward = 0 

while not (terminated or truncated) : 
    action,_ = model.predict(observation,deterministic = True )
    observation,reward,terminated,truncated,info = env.step(action)
    total_reward += reward
    
print(total_reward)
env.close()