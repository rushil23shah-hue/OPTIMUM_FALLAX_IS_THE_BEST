"""Reward checks only: no model training or hackability report generation."""
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
from reward_wrapper import OptimizedWalkerReward, make_walker_env


class WalkerStub(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32)

    def reset(self, **kwargs):
        self.hull = SimpleNamespace(position=SimpleNamespace(x=1.0), angle=0.0, angularVelocity=0.0)
        self.game_over = False
        self.dx, self.terminated, self.truncated = 0.0, False, False
        return np.zeros(24, dtype=np.float32), {}

    def step(self, action):
        self.hull.position.x += self.dx
        return np.zeros(24, dtype=np.float32), 7.0, self.terminated, self.truncated, {}


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.env = OptimizedWalkerReward(WalkerStub())
        self.env.reset()

    def test_stationary_and_backward_cannot_earn_positive_reward(self):
        self.assertLess(self.env.step(np.zeros(4))[1], 0)
        self.env.unwrapped.dx = -0.1
        self.assertLess(self.env.step(np.zeros(4))[1], 0)
        self.env.unwrapped.dx = 0.1
        self.assertGreater(self.env.step(np.zeros(4))[1], 0)

    def test_failure_finish_and_truncation(self):
        self.env.unwrapped.terminated = True
        self.env.unwrapped.game_over = True
        self.assertEqual(self.env.step(np.ones(4))[1], -100)
        self.env.reset()
        self.env.unwrapped.terminated = True
        self.env.unwrapped.dx = 0.1
        self.assertGreater(self.env.step(np.zeros(4))[1], 0)
        self.env.reset()
        self.env.unwrapped.truncated = True
        _, reward, term, trunc, _ = self.env.step(np.zeros(4))
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertNotEqual(reward, -100)

    def test_clipping_reset_and_reward_accounting(self):
        _, reward, _, _, info = self.env.step(np.full(4, 10.0))
        self.assertAlmostEqual(reward, sum(info['reward_terms'].values()))
        self.assertEqual(info['native_reward'], 7.0)
        np.testing.assert_array_equal(self.env.previous_action, np.ones(4))
        first_cost = info['reward_terms']['action_change']
        self.assertEqual(self.env.step(np.ones(4))[4]['reward_terms']['action_change'], 0)
        self.env.reset()
        self.assertEqual(self.env.step(np.ones(4))[4]['reward_terms']['action_change'], first_cost)

    def test_real_walker_identical_physics_different_reward(self):
        with patch.dict(os.environ, WALKER_REWARD_PROFILE='original'):
            original = make_walker_env('BipedalWalker-v3')
        with patch.dict(os.environ, WALKER_REWARD_PROFILE='optimized_v1'):
            optimized = make_walker_env('BipedalWalker-v3')
        try:
            a, _ = original.reset(seed=42)
            b, _ = optimized.reset(seed=42)
            np.testing.assert_array_equal(a, b)
            rng = np.random.default_rng(12)
            differences = []
            for _ in range(100):
                action = rng.uniform(-1, 1, 4).astype(np.float32)
                a, r1, t1, c1, _ = original.step(action)
                b, r2, t2, c2, info = optimized.step(action)
                np.testing.assert_array_equal(a, b)
                self.assertEqual((t1, c1), (t2, c2))
                self.assertEqual(r1, info['native_reward'])
                self.assertAlmostEqual(r2, sum(info['reward_terms'].values()))
                differences.append(abs(r1-r2))
                if t1 or c1:
                    original.reset(seed=43)
                    optimized.reset(seed=43)
            self.assertGreater(max(differences), 0.01)
        finally:
            original.close()
            optimized.close()

    def test_ppo_vector_wrapper_compatibility(self):
        from ppo_simple import make_env
        with patch.dict(os.environ, WALKER_REWARD_PROFILE='optimized_v1'):
            env = gym.vector.SyncVectorEnv([make_env('BipedalWalker-v3', 0, i) for i in range(2)])
        try:
            env.reset(seed=0)
            _, rewards, _, _, info = env.step(np.zeros((2, 4), dtype=np.float32))
            self.assertTrue(np.isfinite(rewards).all())
            self.assertIn('native_reward', info)
        finally:
            env.close()

    def test_interface_isolates_all_six_agents_without_training(self):
        import interface
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'experiment'
            with patch('sys.argv', ['interface.py', '--experiment', str(output)]), patch(
                'interface.subprocess.run', return_value=SimpleNamespace(returncode=0)
            ) as run:
                self.assertEqual(interface.main(), 0)
            self.assertEqual(run.call_count, 6)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['cwd'], output.resolve())
                self.assertEqual(call.kwargs['env']['WALKER_REWARD_PROFILE'], 'optimized_v1')
            manifest = json.loads((output / 'experiment.json').read_text())
            self.assertEqual(manifest['initialization'], 'fresh')
            with patch('sys.argv', ['interface.py', '--experiment', str(output)]), patch(
                'interface.subprocess.run'
            ) as run:
                with self.assertRaises(SystemExit):
                    interface.main()
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
