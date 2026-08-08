# Proximal Policy Optimization (PPO) & Intrinsic Curiosity Module (ICM) on BipedalWalker-v3

A PyTorch implementation of Proximal Policy Optimization (PPO) and the Intrinsic Curiosity Module (ICM) for continuous control environments using Gymnasium.


## Key Features

* Continuous Control Policy: Parameterizes multi-joint torque control using Gaussian action distributions.
* Online Observation Normalization: Tracks running mean and variance online using Welford's algorithm (RunningMeanStd).
* Correct Cutoff Handling: Explicitly distinguishes between true episode termination (done = 1.0) and time-limit truncation bootstrapping using V(s_next).
* Generalized Advantage Estimation (GAE): Implements backward k-step advantage estimation to balance variance and temporal credit assignment.
* Clipped Surrogate Objective: Clamps probability ratios to [1-eps, 1+eps] to prevent destructive policy updates.
* Intrinsic Curiosity Dynamics (ICM): Self-supervised exploration engine featuring a State Feature Encoder, Inverse Dynamics Model (noise filter), and Forward Dynamics Model (curiosity predictor).


## Mathematical Formulation

1. Observation Normalization:
   Normalized_Obs = (Obs - Mean) / sqrt(Var + 1e-8)

2. Temporal Difference Error (delta_t) & GAE (Advantage_t):
   delta_t = (r_ext_t + icm_scale * r_int_t) + gamma * V(s_next) * (1 - done_t) - V(s_t)
   Advantage_t = delta_t + (gamma * gae_lambda) * (1 - done_t) * Advantage_{t+1}

3. PPO Probability Ratio & Clipped Actor Loss:
   Ratio_t = exp(new_log_prob - old_log_prob)
   Surrogate_1 = Ratio_t * Advantage_t
   Surrogate_2 = clamp(Ratio_t, 1 - clip_range, 1 + clip_range) * Advantage_t
   Policy_Loss = -mean(min(Surrogate_1, Surrogate_2))

4. Unified PPO Engine Loss:
   PPO_Loss = Policy_Loss + (vf_coef * Value_Loss) - (ent_coef * Entropy_Loss)

5. Intrinsic Curiosity Module (ICM) Loss:
   Inverse_Loss = MSE(Predicted_Action, Actual_Action)
   Forward_Loss = 0.5 * MSE(Predicted_Next_Feature, Actual_Next_Feature)
   ICM_Loss = (1 - beta) * Inverse_Loss + (beta * Forward_Loss)


## Hyperparameter Configuration

| Hyperparameter | Value | Description |
| env_id | BipedalWalker-v3 | Continuous Gymnasium environment |
| n_steps | 2048 | Rollout step horizon per update |
| n_epochs | 10 | Optimization passes per rollout batch |
| batch_size | 64 | Minibatch size |
| lr | 3e-4 | Learning rate for Adam optimizer |
| gamma | 0.99 | Reward discount factor |
| gae_lambda | 0.95 | GAE bias-variance decay parameter |
| clip_range | 0.2 | PPO surrogate clipping threshold |
| vf_coef | 0.5 | Value function loss weight |
| ent_coef | 0.01 | Entropy regularization weight |
| icm_scale | 0.01 | Intrinsic curiosity reward scaling factor |
| beta | 0.2 | Forward vs Inverse loss weighting |


## Project Structure

* ppo_simple.py : Standard PPO implementation
* ppo_icm.py : PPO + Intrinsic Curiosity Module (ICM) implementation
* ppo_simple_rewards.csv : Logged training returns (Standard PPO)
* ppo_icm_rewards.csv : Logged training returns (PPO + ICM)
* ppo_simple_curve.png : Standard PPO training graph
* ppo_icm_curve.png : PPO + ICM training graph
* README.md : Project documentation

## Execution

To run Standard PPO training:
python ppo_simple.py

To run PPO + ICM training:
python ppo_icm.py


## Benchmark Graphs

* Standard PPO Curve: Refer to ppo_simple_curve.png
* PPO + ICM Curve: Refer to ppo_icm_curve.png
