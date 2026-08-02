# How we selected our 5 axes and 9 agents

## 1. What is an "axis" here?

An axis is just one core design decision that every reinforcement learning algorithm has to make. Think of it like a set of switches — every algorithm has these switches set to one position or another, and the combination of switch positions is basically what makes one algorithm different from another.

We picked 5 switches (axes) that we believe matter most for how an algorithm might interact with — and potentially exploit — a reward function:

1. **On-policy vs off-policy** — can the agent learn only from data it just collected under its current behavior, or can it reuse older experience too?
2. **Value-based vs policy-based vs actor-critic** — does the agent judge actions through a value function, learn a policy directly, or use both together?
3. **Stochastic vs deterministic** — does the agent output a probability distribution over actions, or commit to one exact action?
4. **Exploration strategy** — how does the agent decide to try something new: built-in randomness, curiosity about the unfamiliar, or systematic novelty-seeking?
5. **Model-free vs model-based** — does the agent learn purely from real interaction, or does it build an internal model of the environment to simulate ahead?

## 2. Why these axes, specifically?

Our project isn't just "train an agent and see what happens" — it's about testing whether a reward function itself is exploitable, in a way that isn't just a quirk of one particular algorithm.

If we only used, say, actor-critic methods, and all of them found the same shortcut, we couldn't tell whether that shortcut is a real weakness in the reward, or something all actor-critic methods happen to be prone to. But if agents that are fundamentally different across these 5 axes all independently discover the same exploit, that's strong evidence the flaw lives in the environment's reward design — not in any single algorithm's assumptions.

In short: the 5 axes are our experimental control. They make sure our agents are genuinely different "attackers" probing the same reward function from different angles, so any pattern we find is trustworthy and not a coincidence of algorithm choice.

## 3. The 9 finalized agents

| # | Agent | On/Off-policy | Value / Policy / Actor-Critic | Stochastic / Deterministic | Exploration | Model-free / based |
|---|---|---|---|---|---|---|
| 1 | PPO | On-policy | Actor-critic | Stochastic | Entropy bonus | Free |
| 2 | SAC | Off-policy | Actor-critic | Stochastic | Max-entropy | Free |
| 3 | TD3 / DDPG | Off-policy | Actor-critic | Deterministic | Gaussian noise | Free |
| 4 | TD3 + RND | Off-policy | Actor-critic | Deterministic | RND (novelty-seeking) | Free |
| 5 | PPO + ICM | On-policy | Actor-critic | Stochastic | ICM (curiosity) | Free |
| 6 | REINFORCE | On-policy | Pure policy | Stochastic | Entropy (weak) | Free |
| 7 | MBPO | Off-policy | Actor-critic | Stochastic | Max-entropy | Based |
| 8 | Dreamer | Off-policy | Actor-critic (world-model) | Stochastic | Model-driven imagination | Based |
| 9 | NAF | Off-policy | Value-based | Deterministic | Gaussian noise | Free |

## 4. Why this specific combination covers the design space well

- **On-policy vs off-policy** is well covered — PPO, PPO+ICM, and REINFORCE are on-policy; the remaining six are off-policy.
- **Value/policy/actor-critic** is fully represented — REINFORCE is pure policy, NAF is pure value-based, and the rest are actor-critic.
- **Stochastic vs deterministic** is balanced — TD3, TD3+RND, and NAF are deterministic; the other six are stochastic.
- **Exploration strategies** span five genuinely different mechanisms: entropy bonus, max-entropy, Gaussian noise, curiosity (ICM), and novelty-seeking (RND).
- **Model-free vs model-based** — seven agents are model-free, and MBPO and Dreamer represent two structurally different model-based approaches (short-horizon learned rollouts vs full latent-space world-model imagination).

Two honest limitations worth stating openly rather than hiding:
- **e-greedy exploration** and **pure discrete value-based methods** don't have a clean, standard analogue in continuous action spaces, so those specific values are not represented — this is a property of continuous control itself, not an oversight in our selection.
- **MBPO's policy-learning core is SAC**, so it should be read as "SAC + a learned model," not as a fully independent data point on the actor-critic axis — useful specifically because it isolates the model-free vs model-based comparison while holding the underlying policy algorithm fixed.
