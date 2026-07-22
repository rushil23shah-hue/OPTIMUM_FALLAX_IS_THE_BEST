import gymnasium as gym
from stable_baselines3 import SAC


class COMJitterRewardWrapper(gym.Wrapper):

    def __init__(self, env, jitter_factor=5):
        super().__init__(env)
        self.jitter_factor = jitter_factor

    def get_center_of_mass(self):
        bodies = [self.env.unwrapped.hull] + list(self.env.unwrapped.legs)
        total_mass = 0
        com_x = 0
        com_y = 0

        for body in bodies:
            mass = body.mass
            com_x += body.position.x * mass
            com_y += body.position.y * mass
            total_mass += mass

        return (com_x / total_mass,com_y / total_mass,)

    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)

        self.prev_com_x, self.prev_com_y = self.get_center_of_mass()

        self.prev_x_velocity = 0

        self.com_jitter_sum = 0
        self.vertical_jitter_sum = 0
        self.speed_jitter_sum = 0
        self.step_count = 0

        return obs, info

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        com_x, com_y = self.get_center_of_mass()

        x_velocity = com_x - self.prev_com_x

        vertical_jitter = abs(com_y - self.prev_com_y)

        speed_jitter = abs(x_velocity - self.prev_x_velocity)

        self.prev_com_x = com_x
        self.prev_com_y = com_y
        self.prev_x_velocity = x_velocity

        com_jitter = vertical_jitter + (0.5 * speed_jitter)

        proxy_reward = reward - (self.jitter_factor * com_jitter)

        self.com_jitter_sum += com_jitter
        self.vertical_jitter_sum += vertical_jitter
        self.speed_jitter_sum += speed_jitter
        self.step_count += 1

        info["avg_com_jitter"] = self.com_jitter_sum / self.step_count
        info["avg_vertical_jitter"] = self.vertical_jitter_sum / self.step_count
        info["avg_speed_jitter"] = self.speed_jitter_sum / self.step_count

        return obs, proxy_reward, terminated, truncated, info


env = gym.make("BipedalWalker-v3")

env = COMJitterRewardWrapper(env,jitter_factor=5,)
model = SAC.load("models/Baseline/sac_500000",env=env,)

model.learn(total_timesteps=100000,progress_bar=True,)

model.save("models/Experiments/com_jitter_reward5")

env.close()