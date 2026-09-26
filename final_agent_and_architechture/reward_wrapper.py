"""Shared extrinsic reward; direct agent scripts default to original rewards."""
import os
from dataclasses import asdict, dataclass
import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    version: str = "optimized_v1"
    progress: float = 130.0 / 30.0
    effort: float = 0.028
    tilt: float = 0.10
    angular_speed: float = 0.002
    action_change: float = 0.02
    time_cost: float = 0.01
    failure: float = -100.0


CONFIG = RewardConfig()


def reward_profile():
    profile = os.environ.get("WALKER_REWARD_PROFILE", "original")
    if profile not in ("original", "optimized_v1"):
        raise ValueError(f"Unknown reward profile: {profile}")
    return profile


def reward_metadata(profile):
    return {"profile": profile, "parameters": asdict(CONFIG) if profile == "optimized_v1" else {}}


class OptimizedWalkerReward(gym.Wrapper):
    """Signed progress minus effort, posture, smoothness and time costs.

    No survival/upright bonus can be collected by standing still. Native
    failures remain -100; finish and time-limit truncation are not failures.
    Physics, observations and episode boundaries are unchanged. Smoothness
    uses the previous applied action, initialized to zero on each reset.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_x = float(self.unwrapped.hull.position.x)
        self.previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action):
        applied = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        obs, native_reward, terminated, truncated, info = self.env.step(applied)
        hull = self.unwrapped.hull
        x = float(hull.position.x)
        terms = {
            "progress": CONFIG.progress * (x - self.previous_x),
            "effort": -CONFIG.effort * float(np.abs(applied).sum()),
            "tilt": -CONFIG.tilt * min(float(hull.angle) ** 2, 1.0),
            "angular_speed": -CONFIG.angular_speed * min(float(hull.angularVelocity) ** 2, 25.0),
            "action_change": -CONFIG.action_change * float(np.square(applied - self.previous_action).mean()),
            "time_cost": -CONFIG.time_cost,
        }
        failed = bool(terminated and (self.unwrapped.game_over or x < 0))
        reward = CONFIG.failure if failed else sum(terms.values())
        terms["failure_adjustment"] = reward - sum(terms.values())
        info = dict(info)
        info.update(native_reward=float(native_reward), optimized_reward=float(reward),
                    reward_terms=terms, forward_displacement=x - self.previous_x,
                    walker_failure=failed)
        self.previous_x, self.previous_action = x, applied.copy()
        return obs, float(reward), terminated, truncated, info


def make_walker_env(env_id, **kwargs):
    profile = reward_profile()
    if profile != "original" and env_id != "BipedalWalker-v3":
        raise ValueError("optimized_v1 is defined only for BipedalWalker-v3")
    env = gym.make(env_id, **kwargs)
    return OptimizedWalkerReward(env) if profile == "optimized_v1" else env
