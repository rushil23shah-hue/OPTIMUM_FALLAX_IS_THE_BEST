import gymnasium as gym  # type: ignore
from stable_baselines3 import SAC # type: ignore

env = gym.make("BipedalWalker-v3")

model = SAC("MlpPolicy",
            env,
            verbose = 1,
            tensorboard_log = "./logs/")

model.learn(total_timesteps = 500000)
model.save("models/sac_500000")

env.close()