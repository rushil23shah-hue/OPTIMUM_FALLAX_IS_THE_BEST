import os
import numpy as np
import gymnasium as gym  # Modern replacement for legacy gym
from sac import Agent
from plot import plot_learning_curve

if __name__ == '__main__':
    # 1. Create directory for saving plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # 2. Initialize Environment (Gymnasium API)
    env = gym.make('BipedalWalker-v3')

    # Optional: Video Recording (Modern Gymnasium Replacement for gym.wrappers.Monitor)
    # env = gym.wrappers.RecordVideo(
    #     env, 
    #     video_folder='tmp/video', 
    #     episode_trigger=lambda ep_id: True
    # )

    agent = Agent(
        input_dims=env.observation_space.shape, 
        env=env,
        n_actions=env.action_space.shape[0]
    )

    n_games = 1000
    filename = 'bipedal_walker_sac.png'
    figure_file = os.path.join('plots', filename)

    # 3. Robust initialization for best_score (avoids reward_range AttributeError)
    best_score = float('-inf')
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # For Gymnasium, rendering mode is passed to gym.make('BipedalWalker-v3', render_mode='human')

    for i in range(n_games):
        # 4. Gymnasium returns (observation, info) on reset
        observation, info = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            # 5. Gymnasium returns 5 values on step
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            score += reward
            agent.remember(observation, action, reward, observation_, done)
            
            if not load_checkpoint:
                agent.learn()
                
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f'Episode {i} | Score: {score:.1f} | Avg Score (100 eps): {avg_score:.1f}')

    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)