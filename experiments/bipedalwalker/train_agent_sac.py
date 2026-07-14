import gymnasium as gym  # type: ignore
from stable_baselines3 import SAC # type: ignore
from envs.E00_reward_sep import BipedalWalker()

env = BipedalWalker()

model = SAC("MlpPolicy",
            env,
            verbose = 1,
            tensorboard_log = "./logs/E00/")

model.learn(total_timesteps = 500000,progress_bar=True)
model.save("models/E00/model_sac")

env.close()