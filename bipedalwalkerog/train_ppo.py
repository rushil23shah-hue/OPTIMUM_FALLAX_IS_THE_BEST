import os
import gymnasium as gym
from stable_baselines3 import PPO

os.makedirs("models",exist_ok=True)
# Create the environment
env = gym.make("BipedalWalker-v3")

# Create the PPO model
model = PPO("MlpPolicy",
             env,
             verbose=1,
             tensorboard_log="./ppo_bipedalwalker_tensorboard/")

# Train the model
model.learn(total_timesteps=1000000)
model.save("models/ppo_bipedalwalker-1000000")
env.close()