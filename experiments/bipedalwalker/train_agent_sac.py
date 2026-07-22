import gymnasium as gym  # type: ignore
from stable_baselines3 import SAC # type: ignore
from envs.E00_reward_sep import BipedalWalker
from stable_baselines3.common.monitor import Monitor

env = Monitor(BipedalWalker())

model = SAC("MlpPolicy",
            env,
            verbose = 1,
            tensorboard_log = "./logs/E00/onlyprogress")

model.learn(total_timesteps = 500000,progress_bar=True,tb_log_name="SAC_onlyprogress")
model.save("models/E00/SAC/onlyprogress")

env.close()

