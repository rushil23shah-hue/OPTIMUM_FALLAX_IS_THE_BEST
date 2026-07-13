import gymnasium as gym
import numpy as np

env = BipedalWalker()

obs, info = env.reset()

total_reward = 0

for step in range(20):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"Step {step+1:02d} | Reward = {reward:.4f}"
    )

    if terminated or truncated:
        print("Episode ended.")
        break

env.close()

print(f"\nTotal Reward = {total_reward:.3f}")