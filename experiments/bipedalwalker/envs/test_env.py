from envs.E00_reward_sep import BipedalWalker

env = BipedalWalker(render_mode="human")

obs, info = env.reset(seed=42)

terminated = False
truncated = False

episode_reward = 0
max_steps=2000
steps=0

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    steps+=1
    if steps>=max_steps:
        print("timeout")
        break

print(f"Episode reward: {episode_reward:.2f}")

env.close()