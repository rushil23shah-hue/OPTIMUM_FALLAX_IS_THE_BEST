import gymnasium as gym  # type: ignore
from stable_baselines3 import PPO  # type: ignore
from envs.E00_reward_sep import BipedalWalker

env = BipedalWalker()

model = PPO("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/E00/")


model.learn(total_timesteps=1000000)
model.save("models/E00/model_ppo")
env.close()