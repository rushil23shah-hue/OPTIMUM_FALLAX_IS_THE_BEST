import gymnasium as gym # type: ignore
import numpy as np
from stable_baselines3 import PPO # type: ignore


env = gym.make("BipedalWalker-v3",render_mode="human")

#model = PPO("MlpPolicy",env,verbose=1)

observation,info = env.reset()


terminated = False
truncated = False
total_reward=0
timesteps_survived=0


while not (terminated or truncated) : 
    action = env.action_space.sample()
    #action,_ = model.predict(observation)
    #action = np.array([0.5,-0.5,-0.5,0.5])
    observation,reward,terminated,truncated,info = env.step(action)
    total_reward += reward
    timesteps_survived +=1
    
print(total_reward)
print(timesteps_survived)
env.close()


