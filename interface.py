"""
interface.py
=============
Single entry point connecting all 8 agents to the environment:
PPO, SAC, PPO+ICM, TD3, Dreamer, TD3+RND, NAF, MBPO.

This file:
  1. Imports BaseAgent -- the contract every agent implements (agents/base_agent.py).
  2. Registers all 8 agent classes under a string name, importing each
     safely: an agent module that's missing, unfinished, or has a bug is
     skipped into AGENT_IMPORT_ERRORS instead of crashing every other
     agent's ability to run.
  3. Provides make_agent(...) to construct any agent, sized correctly
     for a given environment.
  4. Provides make_env(...) to build any continuous Gymnasium environment,
     training on its ORIGINAL, UNMODIFIED reward -- no reward stripping or
     shaping happens here. Hackability is probed by comparing how the 8
     structurally different agents behave under the SAME real reward,
     not by removing or altering the reward signal.
  5. Provides run_episode(...), a single per-step training loop that
     works identically for every agent/environment pair.

Individual agent implementations live in agents/<name>_agent.py and are
imported ONLY here. Training/experiment code (run_comparison.py) should
never import an agent class directly -- always go through AGENT_REGISTRY.
"""

from __future__ import annotations
from typing import Dict, Optional
import importlib
import numpy as np
import gymnasium as gym

from agents.base_agent import BaseAgent


_AGENT_IMPORT_SPECS = [
    ("ppo",      "agents.ppo_agent",      "PPOAgent"),
    ("sac",      "agents.sac_agent",      "SACAgent"),
    ("ppo_icm",  "agents.ppo_icm_agent",  "PPOICMAgent"),
    ("td3",      "agents.td3_agent",      "TD3Agent"),
    ("dreamer",  "agents.dreamer_agent",  "DreamerAgent"),
    ("td3_rnd",  "agents.td3_rnd_agent",  "TD3RNDAgent"),
    ("naf",      "agents.naf_agent",      "NAFAgent"),
    ("mbpo",     "agents.mbpo_agent",     "MBPOAgent"),
]

AGENT_REGISTRY: Dict[str, type] = {}
AGENT_IMPORT_ERRORS: Dict[str, str] = {}

for _key, _module_name, _class_name in _AGENT_IMPORT_SPECS:
    try:
        _module = importlib.import_module(_module_name)
        AGENT_REGISTRY[_key] = getattr(_module, _class_name)
    except Exception as _e:  
        AGENT_IMPORT_ERRORS[_key] = str(_e)



def make_agent(name: str, env: gym.Env, **kwargs) -> BaseAgent:
    """
    Construct an agent by name, sized correctly for the given environment.

    Args:
        name: key into AGENT_REGISTRY, e.g. "ppo", "td3", "naf"
        env:  a Gymnasium env with continuous (Box) observation and action spaces
        kwargs: any agent-specific hyperparameters (seed, buffer_size, etc.)
    """
    name = name.lower()
    if name not in AGENT_REGISTRY:
        available = list(AGENT_REGISTRY.keys())
        failed = list(AGENT_IMPORT_ERRORS.keys())
        raise ValueError(
            f"Unknown or unavailable agent '{name}'. "
            f"Available: {available}. Failed to import: {failed} "
            f"(see AGENT_IMPORT_ERRORS for details)."
        )

    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("interface.py currently supports continuous (Box) observation spaces only.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("interface.py currently supports continuous (Box) action spaces only.")

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low
    action_high = env.action_space.high

    agent_cls = AGENT_REGISTRY[name]
    return agent_cls(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        **kwargs,
    )



def make_env(env_id: str, seed: Optional[int] = None) -> gym.Env:
    """
    Central place to build any environment for the study. Every agent
    trains on the environment's real, unmodified reward -- hackability is
    detected later by comparing behavior ACROSS agents, not by altering
    the reward.
    """
    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
    return env


def run_episode(agent: BaseAgent, env: gym.Env, max_steps: int = 1000,
                 train: bool = True, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Run one episode. If train=True, calls store_transition() + update()
    after every step (per-step online training -- standard for the
    off-policy agents here; on-policy agents self-gate inside update()).
    Returns {"episode_return": float, "episode_length": int}.
    """
    obs, _ = env.reset(seed=seed)
    agent.reset()
    episode_return = 0.0
    episode_length = 0

    for _ in range(max_steps):
        action = agent.act(obs, deterministic=not train)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

        episode_return += reward
        episode_length += 1
        obs = next_obs
        if done:
            break

    return {"episode_return": episode_return, "episode_length": episode_length}


if __name__ == "__main__":
    ENV_ID = "BipedalWalker-v3"
    AGENT_NAME = "ppo"

    print("Registered agents:", list(AGENT_REGISTRY.keys()))
    if AGENT_IMPORT_ERRORS:
        print("Not yet available:", list(AGENT_IMPORT_ERRORS.keys()))

    env = make_env(ENV_ID, seed=0)
    agent = make_agent(AGENT_NAME, env, seed=0)

    for episode in range(10):
        result = run_episode(agent, env, max_steps=300, train=True)
        print(f"[{AGENT_NAME}] episode {episode}: return = {result['episode_return']:.2f}")

    env.close()