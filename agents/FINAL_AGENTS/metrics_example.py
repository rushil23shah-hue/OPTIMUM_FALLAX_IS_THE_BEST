"""NumPy-only example of every metric. Synthetic data, not a trained-agent result.

Run: python agents/FINAL_AGENTS/metrics_example.py
For actual agents use interface.py or metrics_integration.evaluate_agent.
"""
import json
import numpy as np

try:
    from .metrics import (Trajectory, evaluate_trajectory, termination_timing_distribution,
                          return_calibration_gap_variance, cross_agent_policy_distance)
except ImportError:
    from metrics import (Trajectory, evaluate_trajectory, termination_timing_distribution,
                         return_calibration_gap_variance, cross_agent_policy_distance)


def main():
    episodes = [
        Trajectory([[0], [1], [2], [3]], [[0], [0.5], [1]], [1, 1, 5],
                   [False, False, True], [False, False, False], horizon=10),
        Trajectory([[0], [1], [2]], [[0], [-1]], [1, -2],
                   [False, True], [False, False], horizon=10),
    ]

    # V(s) adapter: ignores actions. A Q adapter would use them and supply
    # evaluation-policy continuation values when actions=None.
    def critic(obs_batch, action_batch=None):
        return 0.5 * obs_batch[:, 0]

    reports = [evaluate_trajectory(t, critic) for t in episodes]
    shared_obs = np.array([[0.0], [1.0], [2.0]])
    output = {
        "data_source": "synthetic example, not evidence of reward hacking",
        "episodes": reports,
        "termination_timing_distribution": termination_timing_distribution(episodes),
        "return_calibration_gap_variance": return_calibration_gap_variance(
            [r["return_calibration_gap"]["gaps"] for r in reports]),
        "cross_agent_policy_distance": cross_agent_policy_distance(shared_obs, {
            "policy_a": lambda obs: np.tanh(obs),
            "policy_b": lambda obs: np.tanh(0.5 * obs),
        }),
    }
    print(json.dumps(output, indent=2, allow_nan=False))
    return output


if __name__ == "__main__":
    main()
