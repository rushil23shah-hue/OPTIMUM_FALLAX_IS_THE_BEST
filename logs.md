Optimum Fallax Progress Record

Till now, I worked on implementing and comparing two different methods on the BipedalWalker-v3 environment: PPO and NEAT. The main goal was to train agents, observe their behavior, compare how they learn, and understand whether the reward objective actually produces the intended walking behavior.

1. Environment Studied
I used the Gymnasium environment:
BipedalWalker-v3
This environment has continuous observations and continuous actions. The agent controls the walker using 4 motor actions, so algorithms that can handle continuous control are more suitable.

2. PPO Implementation
For PPO, I used Stable-Baselines3 because PPO is a strong policy-gradient method for continuous-control tasks.
The simple PPO training code does the following:
1. Creates the BipedalWalker environment
2. Creates a PPO model using MlpPolicy
3. Trains it for 1,000,000 timesteps
4. Saves the trained model
5. Tests the saved model separately

Important PPO settings used:
Algorithm: PPO
Policy: MlpPolicy
Environment: BipedalWalker-v3
Training timesteps: 1,000,000
Logging: TensorBoard logging was added
Testing: Deterministic actions were used during testing
PPO solved the reward objective, but the learned movement was not natural walking.

3. NEAT Implementation
For NEAT, I used neat-python. NEAT works differently from PPO. Instead of using backpropagation, it evolves a population of neural networks over generations.
The NEAT implementation does the following:
1. Creates a population of neural networks
2. Tests each genome in BipedalWalker
3. Assigns a fitness score to each genome
4. Evolves better genomes using selection, mutation, crossover, and speciation
5. Saves the best genome
6. Tests the saved genome separately

The network structure used:
Inputs: 24 observations from BipedalWalker
Outputs: 4 motor actions
Output activation: tanh
Actions clipped between -1 and 1
4. NEAT Hyperparameter Tuning
Initially, I started with a smaller NEAT setup:
population size = 50
generations = 30
fitness = total_reward
This was not enough because the agent struggled to move far. Then I increased the training effort and tuned the setup.
Changes made:
population size increased from 50 to 150
generations increased from 30 to around 100-200
survival_threshold changed from 0.2 to 0.15
checkpoints added after fixed generation intervals
fitness_threshold increased to 999999
no_fitness_termination set to True

The reason for changing fitness_threshold and no_fitness_termination was that the new custom fitness score could become large quickly,
which was stopping training early even though the walker had not properly solved the environment.
5. NEAT Fitness Function Tuning
The first fitness function was simple:
fitness = total_reward
But this did not guide NEAT strongly enough.

Then I added distance-based shaping:
fitness = total_reward + (distance * 5)
This helped the agent move farther.
Later I added a timestep penalty because the agent was moving too slowly and surviving for a long time without crossing quickly.
Final improved fitness idea:
fitness = total_reward + (distance * 8) - (steps * 0.1) + (crossed * 700)
