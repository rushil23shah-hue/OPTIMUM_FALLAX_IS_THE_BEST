import os
import gymnasium as gym
import numpy as np
from td3 import Agent
from plot import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')

    # Ensure output directory exists for plot saving
    os.makedirs('plots', exist_ok=True)

    agent = Agent(
        alpha=0.001,
        beta=0.001,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_actions=env.action_space.shape[0]
    )

    n_games = 1000
    filename = os.path.join('plots', f'BipedalWalker_{n_games}_games.png')

    best_score = float('-inf')
    score_history = []

    # Safely attempt to load pre-trained models if they exist
    try:
        agent.load_models()
        print("Pre-trained weights loaded successfully.")
    except Exception as e:
        print("No saved checkpoint found. Starting training from scratch.")

    for i in range(n_games):
        # Gymnasium reset returns (observation, info)
        observation, info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            
            # Gymnasium step returns 5 values: obs, reward, terminated, truncated, info
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'Episode {i} | Score: {score:.1f} | Avg Score (last 100): {avg_score:.1f}')

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)