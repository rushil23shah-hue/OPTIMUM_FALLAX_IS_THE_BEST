"""Run with: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents" / "FINAL_AGENTS"))
import metrics as m
from metrics_integration import collect_trajectories, attach_metrics_context
import interface


def trajectory(term=True, trunc=False, horizon=10):
    return m.Trajectory([[0], [1], [2]], [[0], [1]], [1, 2], [False, term], [False, trunc], horizon)


def critic(obs, actions=None):
    return np.full(len(obs), 10.0)


class MetricsTests(unittest.TestCase):
    def test_td_terminal_and_truncated_bootstrap(self):
        np.testing.assert_allclose(m.td_error_anomaly(trajectory(), critic, 0.5)["residuals"], [-4, -8])
        np.testing.assert_allclose(m.td_error_anomaly(trajectory(False, True), critic, 0.5)["residuals"], [-4, -3])

    def test_return_terminal_and_truncated_bootstrap(self):
        np.testing.assert_allclose(m.return_calibration_gap(trajectory(), critic, 0.5)["returns"], [2, 2])
        result = m.return_calibration_gap(trajectory(False, True), critic, 0.5)
        np.testing.assert_allclose(result["returns"], [4.5, 7])
        self.assertAlmostEqual(result["variance"], 1.5625)
        self.assertAlmostEqual(result["trend_per_step"], 2.5)

    def test_terminal_state_is_not_queried(self):
        def limited(obs, actions=None):
            self.assertFalse(np.any(obs == 2))
            return np.zeros(len(obs))
        m.td_error_anomaly(trajectory(), limited)
        m.return_calibration_gap(trajectory(), limited)

    def test_concentration_signed_zero_uniform_and_spike(self):
        self.assertAlmostEqual(m.reward_concentration([-1, 1])["gini"], 0)
        self.assertAlmostEqual(m.reward_concentration([1, 0, 0, 0])["gini"], 0.75)
        self.assertIsNone(m.reward_concentration([0, 0])["entropy"])
        self.assertAlmostEqual(m.reward_concentration([1, 1])["normalized_entropy"], 1)

    def test_reward_rate(self):
        self.assertEqual(m.reward_rate_normalization(trajectory())["reward_rate"], 1.5)

    def test_termination_distinguishes_nonterminal_and_horizon(self):
        result = m.termination_timing_distribution([trajectory(), trajectory(False, True), trajectory(False, False)])
        self.assertEqual(result["horizon_fractions"], [0.2])
        self.assertEqual(result["terminated_count"], 1)
        self.assertEqual(result["truncated_only_count"], 1)
        self.assertEqual(result["unfinished_count"], 1)
        self.assertEqual(sum(result["histogram"]), 1)
        self.assertEqual(m.termination_timing_distribution([trajectory(horizon=None)])["missing_horizon_count"], 1)

    def test_action_bounds_asymmetric_and_entropy(self):
        result = m.action_saturation_entropy([[0, 2], [1, 4]], low=[0, 2], high=[2, 4])
        self.assertEqual(result["saturation_fraction"], 0.75)
        a = m.action_saturation_entropy(np.ones((4, 2)))
        self.assertEqual(a["histogram_entropy_per_dimension"], [0, 0])
        with self.assertRaises(ValueError):
            m.action_saturation_entropy([[2]])

    def test_gap_variance_does_not_pad_short_episodes(self):
        result = m.return_calibration_gap_variance([[1, 2, 3], [3, 4]])
        self.assertEqual(result["matched_step_variance"], [1, 1, None])
        self.assertEqual(result["matched_step_count"], [2, 2, 1])

    def test_perturbation_reproducible_and_fixed_actions(self):
        t = trajectory()
        def q(obs, actions=None):
            np.testing.assert_equal(actions, t.actions)
            return obs[:, 0] * 2 + actions[:, 0]
        a = m.critic_sensitivity(t, q, samples=3, seed=42)
        b = m.critic_sensitivity(t, q, samples=3, seed=42)
        self.assertEqual(a, b)
        self.assertGreater(a["mean"], 0)
        self.assertEqual(m.critic_sensitivity(t, critic)["mean"], 0)

    def test_shared_states_and_gaussian_kl(self):
        states = np.array([[0], [1], [2]])
        def first(obs):
            np.testing.assert_equal(obs, states)
            return np.zeros((len(obs), 1))
        second = lambda obs: np.ones((len(obs), 1))
        result = m.cross_agent_policy_distance(states, {"a": first, "b": second},
            {"a": lambda obs: (first(obs), np.ones((len(obs), 1))),
             "b": lambda obs: (second(obs), np.ones((len(obs), 1)))})["pairs"]["a__b"]
        self.assertEqual(result["mean_l2"], 1)
        self.assertEqual(result["kl_forward"], 0.5)
        self.assertEqual(result["kl_reverse"], 0.5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            m.Trajectory([[0], [1]], [[0]], [np.nan], [True], [False])
        with self.assertRaises(ValueError):
            m.Trajectory([[0], [1], [2]], [[0], [0]], [1, 1], [True, True], [False, False])
        with self.assertRaises(ValueError):
            m.td_error_anomaly(trajectory(), lambda o, a=None: np.zeros((len(o), 2)))
        with self.assertRaises(ValueError):
            m.return_calibration_gap(trajectory(), critic, gamma=1.1)

    def test_report_is_strict_json(self):
        json.dumps(m.evaluate_trajectory(trajectory(), critic), allow_nan=False)

    def test_normalizer_snapshot_is_independent(self):
        rms = SimpleNamespace(mean=np.ones(2), var=np.ones(2))
        model = attach_metrics_context(SimpleNamespace(), "test", obs_rms=rms)
        rms.mean[:] = 99
        np.testing.assert_equal(model.metrics_context["mean"], [1, 1])

    def test_collector_keeps_final_state_and_executed_actions(self):
        class Env:
            spec = SimpleNamespace(max_episode_steps=5)
            action_space = SimpleNamespace(shape=(1,), low=np.array([-1]), high=np.array([1]), dtype=np.float32)
            def reset(self, seed):
                self.i = 0
                return np.array([0.0]), {}
            def step(self, action):
                self.i += 1
                return np.array([float(self.i)]), 1, self.i == 2, False, {}
        trajectories = collect_trajectories(Env(), lambda obs: np.full((len(obs), 1), 3), episodes=2)
        np.testing.assert_equal(trajectories[0].obs, [[0], [1], [2]])
        np.testing.assert_equal(trajectories[0].actions, [[1], [1]])
        partial = collect_trajectories(Env(), lambda obs: np.zeros((len(obs), 1)), episodes=1, max_steps=1)[0]
        self.assertFalse(partial.terminated[-1] or partial.truncated[-1])

    def test_interface_routes_all_six_trainers_and_isolates_failure(self):
        called = []
        def fake_import(module):
            def train(**kwargs):
                called.append((module, kwargs))
                if module == "ppo_simple":
                    raise RuntimeError("test failure")
                return SimpleNamespace()
            function = next(fn for mod, fn in interface.AGENTS.values() if mod == module)
            return SimpleNamespace(**{function: train})
        def fake_evaluate(name, model, output, *args):
            return SimpleNamespace(policy=lambda obs: np.zeros((len(obs), 1))), [trajectory()], {}
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(interface.importlib, "import_module", side_effect=fake_import), patch.object(interface, "evaluate_agent", side_effect=fake_evaluate):
                result = interface.run_agents(steps=4, episodes=1, run_dir=directory)
            self.assertEqual(len(called), 6)
            self.assertEqual(result["ppo"]["status"], "failed")
            self.assertEqual(result["sac"]["status"], "ok")
            self.assertEqual(Path.cwd(), cwd)
            self.assertEqual(len(list(Path(directory).glob("*/cross_agent_metrics.json"))), 1)


if __name__ == "__main__":
    unittest.main()
