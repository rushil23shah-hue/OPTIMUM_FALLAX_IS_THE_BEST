import gymnasium as gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from envs.E00_reward_sep import BipedalWalker
from stable_baselines3.common.monitor import Monitor

env = Monitor(BipedalWalker())

model = PPO("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/E00/")


model.learn(total_timesteps=1000000,progress_bar = True,tb_log_name="PPO_E00_1M")
model.save("models/E00/model_ppo")
env.close()