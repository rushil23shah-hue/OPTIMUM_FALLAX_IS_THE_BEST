import os
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC


MODEL_PATH = "models/Experiments/rushil_perfectwalker"
MODEL_NAME = "Rushil perfect"


SAVE_DIR = f"plots/{MODEL_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

env = gym.make("BipedalWalker-v3")
model = SAC.load(MODEL_PATH)

obs, info = env.reset()

done = False



steps = []

reward_history = []

velocity_history = []

angle_history = []

x_history = []

y_history = []

distance_history = []

vertical_jitter_history = []

speed_jitter_history = []

com_jitter_history = []

# foot_history = []


bodies = [env.unwrapped.hull] + list(env.unwrapped.legs)

total_mass = 0
com_x = 0
com_y = 0

for body in bodies:
    m = getattr(body, "mass", 1)
    total_mass += m
    com_x += body.position.x * m
    com_y += body.position.y * m

prev_com_x = com_x / total_mass
prev_com_y = com_y / total_mass
prev_velocity = 0

initial_x = env.unwrapped.hull.position.x

step = 0

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    step += 1

    steps.append(step)

    reward_history.append(reward)

    x = env.unwrapped.hull.position.x
    y = env.unwrapped.hull.position.y

    x_history.append(x)
    y_history.append(y)

    distance_history.append(x - initial_x)

    velocity = env.unwrapped.hull.linearVelocity.x
    velocity_history.append(velocity)

    angle = abs(env.unwrapped.hull.angle)
    angle_history.append(angle)


    bodies = [env.unwrapped.hull] + list(env.unwrapped.legs)

    total_mass = 0
    com_x = 0
    com_y = 0

    for body in bodies:
        m = getattr(body, "mass", 1)
        total_mass += m
        com_x += body.position.x * m
        com_y += body.position.y * m

    com_x /= total_mass
    com_y /= total_mass

    x_vel = com_x - prev_com_x

    vertical = abs(com_y - prev_com_y)

    speed_jitter = abs(x_vel - prev_velocity)

    com_jitter = vertical + 0.5 * speed_jitter

    vertical_jitter_history.append(vertical)
    speed_jitter_history.append(speed_jitter)
    com_jitter_history.append(com_jitter)

    prev_com_x = com_x
    prev_com_y = com_y
    prev_velocity = x_vel


'''
    left = env.unwrapped.legs[1].ground_contact
    right = env.unwrapped.legs[3].ground_contact

    if left and not right:
        foot_history.append(1)

    elif right and not left:
        foot_history.append(-1)

    else:
        foot_history.append(0)
'''
env.close()

def save_plot(y, ylabel, title, filename):

    plt.figure(figsize=(9,4))

    plt.plot(steps, y)

    plt.xlabel("Step")
    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)

    plt.close()


save_plot(reward_history,
          "Reward",
          "Reward per Step",
          "reward.png")

save_plot(velocity_history,
          "Velocity",
          "Forward Velocity",
          "velocity.png")

save_plot(angle_history,
          "Angle (rad)",
          "Hull Angle",
          "angle.png")

save_plot(distance_history,
          "Distance",
          "Distance Travelled",
          "distance.png")

save_plot(x_history,
          "Hull X",
          "Hull X Position",
          "x_position.png")

save_plot(y_history,
          "Hull Y",
          "Hull Height",
          "height.png")

save_plot(vertical_jitter_history,
          "Vertical Jitter",
          "Vertical COM Jitter",
          "vertical_jitter.png")

save_plot(speed_jitter_history,
          "Speed Jitter",
          "Speed Jitter",
          "speed_jitter.png")

save_plot(com_jitter_history,
          "COM Jitter",
          "COM Jitter",
          "com_jitter.png")

# Foot Contact Plot
'''
plt.figure(figsize=(10,2))

plt.plot(steps, foot_history)

plt.yticks([-1,0,1], ["Right","Both/Air","Left"])

plt.xlabel("Step")

plt.title("Foot Contact Pattern")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "foot_contact.png"), dpi=300)

plt.close()
'''

print(f"\nPlots saved to: {SAVE_DIR}")