import os
import csv
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC

MODELS = {
    "Baseline": "models/Baseline/sac_500000",
    "XY Reward": "models/Experiments/xy_reward01",
    "Forward Speed": "models/Experiments/fwdspeed_reward01",
    "Linear Velocity": "models/Experiments/LinearVelocity_reward01_factor=10",
    "Angle": "models/Experiments/angle_penalty01_factor=3",
    #"Foot Contact": "models/Experiments/foot_alteration_reward0.5",
    "COM":"models/Experiments/com_jitter_reward5",
    "Rushil": "models/Experiments/rushil_perfectwalker"
}

N_EPISODES = 20
env = gym.make("BipedalWalker-v3")
os.makedirs("results", exist_ok=True)

csv_file = open("results/results.csv","w",newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "Model",
    "Reward",
    "Distance",
    "Velocity",
    "Angle",
    "COM Jitter",
    #"Foot Alternations",
    "Episode Length"
])
for model_name, model_path in MODELS.items():
    print("=" * 50)
    print("Loading:", model_name)
    print("Path:", model_path)

    try:
        model = SAC.load(model_path)
        print("Loaded successfully!")
    except Exception as e:
        print(f"Failed to load {model_name}")
        print(e)
        break

    

    rewards = []
    distances = []
    velocities = []
    angles = []
    com_jitters = []
    #foot_switches = []
    episode_lengths = []

    for episode in range(N_EPISODES):

        obs, info = env.reset()

        done = False

        total_reward = 0
        step_count = 0

        initial_x = env.unwrapped.hull.position.x

        prev_com_x = 0
        prev_com_y = 0
        prev_velocity = 0

        angle_sum = 0
        velocity_sum = 0
        com_jitter_sum = 0

        prev_single_foot = None
        switch_count = 0

        # Initialize COM
        bodies = [env.unwrapped.hull] + list(env.unwrapped.legs)

        total_mass = 0
        com_x = 0
        com_y = 0

        for body in bodies:
            mass = getattr(body, "mass", 1)
            total_mass += mass
            com_x += body.position.x * mass
            com_y += body.position.y * mass

        prev_com_x = com_x / total_mass
        prev_com_y = com_y / total_mass

        while not done:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Velocity
            velocity = env.unwrapped.hull.linearVelocity.x
            velocity_sum += velocity

            # Angle
            angle = abs(env.unwrapped.hull.angle)
            angle_sum += angle

            # COM
            bodies = [env.unwrapped.hull] + list(env.unwrapped.legs)

            total_mass = 0
            com_x = 0
            com_y = 0

            for body in bodies:
                mass = getattr(body, "mass", 1)
                total_mass += mass
                com_x += body.position.x * mass
                com_y += body.position.y * mass

            com_x /= total_mass
            com_y /= total_mass

            x_velocity = com_x - prev_com_x

            vertical_jitter = abs(com_y - prev_com_y)
            speed_jitter = abs(x_velocity - prev_velocity)

            com_jitter = vertical_jitter + 0.5 * speed_jitter

            com_jitter_sum += com_jitter

            prev_com_x = com_x
            prev_com_y = com_y
            prev_velocity = x_velocity
            '''
            # Foot alternation
            left = env.unwrapped.legs[1].ground_contact
            right = env.unwrapped.legs[3].ground_contact

            current_single = None

            if left and not right:
                current_single = "left"

            elif right and not left:
                current_single = "right"

            if current_single is not None:

                if (
                    prev_single_foot is not None
                    and current_single != prev_single_foot
                ):
                    switch_count += 1

                prev_single_foot = current_single
                '''
                
        # Episode finished
        final_x = env.unwrapped.hull.position.x
        distance = final_x - initial_x

        if step_count > 0:
            avg_velocity = velocity_sum / step_count
            avg_angle = angle_sum / step_count
            avg_com_jitter = com_jitter_sum / step_count
        else:
            avg_velocity = 0
            avg_angle = 0
            avg_com_jitter = 0

        rewards.append(total_reward)
        distances.append(distance)
        velocities.append(avg_velocity)
        angles.append(avg_angle)
        com_jitters.append(avg_com_jitter)
        #foot_switches.append(switch_count)
        episode_lengths.append(step_count)

    writer.writerow([
        model_name,
        round(np.mean(rewards), 2),
        round(np.mean(distances), 2),
        round(np.mean(velocities), 3),
        round(np.mean(angles), 3),
        round(np.mean(com_jitters), 4),
        #round(np.mean(foot_switches), 2),
        round(np.mean(episode_lengths), 2)
    ])

    print(f"{model_name} Done")
csv_file.close()
env.close()
print("Evaluation Complete!")