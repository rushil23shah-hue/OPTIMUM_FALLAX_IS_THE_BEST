"""Simulation/inference and mocked launch tests. Never calls backward or step."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
import nasim

from fallax_toolkit.nasim import notebook_model as model
from fallax_toolkit.nasim.adapter import create_bank, create_models, Episode, bank_fingerprint, validate_bank, make_environment, _scenario_template
from fallax_toolkit.nasim.config import ENVIRONMENT, SOURCE_SHA256, make_manifest
from fallax_toolkit.nasim.runner import collect, preflight, load_checkpoint, save_checkpoint, save_json, read_manifest
from fallax_toolkit.nasim.report import analyze, PROFILE
from fallax_toolkit.server import Controller
from metrics import Trajectory, td_error_anomaly, return_calibration_gap, reward_concentration


class NasimIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.banks = create_bank(cls.root/'current', 'nasim_current')
        cls.old = create_bank(cls.root/'penalty', 'nasim_scan_penalty')
        cls.models = create_models()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_notebook_and_copied_definitions(self):
        original = Path(__file__).parent.parent/'pentesting-agent (5).ipynb'
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), SOURCE_SHA256)
        notebook = json.loads(original.read_text(encoding='utf-8'))
        definitions = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            try: tree = ast.parse(''.join(cell['source']))
            except SyntaxError: continue
            for item in tree.body:
                if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
                    definitions[item.name] = ast.dump(item)
        extracted = ast.parse(Path(model.__file__).read_text(encoding='utf-8'))
        for item in extracted.body:
            if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
                self.assertEqual(ast.dump(item), definitions[item.name], item.name)

    def test_bank_reproducibility_and_all_scenarios_load(self):
        again = create_bank(self.root/'repeat', 'nasim_current')
        self.assertEqual(bank_fingerprint(again), bank_fingerprint(self.banks))
        self.assertNotEqual(bank_fingerprint(self.old), bank_fingerprint(self.banks))
        self.assertEqual([len(v) for v in self.banks.values()], [36,12,20])
        for split in self.banks:
            for current, old in zip(self.banks[split], self.old[split]):
                a=yaml.safe_load(Path(current['path']).read_text())
                b=yaml.safe_load(Path(old['path']).read_text())
                for name in model.SCAN_ACTION_NAMES:
                    self.assertEqual(a.pop(name+'_cost'),0)
                    self.assertEqual(b.pop(name+'_cost'),1)
                self.assertEqual(a,b)
                for spec in (current,old):
                    env=nasim.load(spec['path'],fully_obs=False)
                    env.reset(seed=2026);env.close()

    def test_scan_penalty_is_negative_reward(self):
        episode=Episode(self.old['test'][0],'nasim_scan_penalty',2026)
        try:
            action=model.build_nasim_action('service_scan',(1,0),episode.env)
            _,reward,_,_,info=episode.env.step(action)
            self.assertEqual(action.cost,1)
            self.assertEqual(reward,float(info.get('value',0))-1)
        finally: episode.close()

    def test_preloaded_episodes_never_reenter_yaml_loader(self):
        _scenario_template.cache_clear()
        validate_bank(self.banks)
        specs = [s for split in self.banks.values() for s in split]
        # Simulate the observed parser failure after startup. All episodes must
        # still initialize without reentering NASim's file loader.
        with patch('fallax_toolkit.nasim.adapter.nasim.load', side_effect=TypeError("'FullLoader' object is not callable")):
            for index in range(2040):
                episode = Episode(specs[index % len(specs)], 'nasim_current', index)
                self.assertEqual(episode.steps, 0)
                episode.close()

    def test_cached_scenarios_match_file_loader_and_are_independent(self):
        for split in self.banks.values():
            for spec in split:
                cached = make_environment(spec)
                fresh = nasim.load(spec['path'], fully_obs=False)
                try:
                    a, _ = cached.reset(seed=27)
                    b, _ = fresh.reset(seed=27)
                    np.testing.assert_array_equal(a, b)
                    for _ in range(3):
                        action = model.build_nasim_action('service_scan', (1,0), cached)
                        np.random.seed(27)
                        result_a = cached.step(action)
                        np.random.seed(27)
                        result_b = fresh.step(action)
                        np.testing.assert_array_equal(result_a[0], result_b[0])
                        self.assertEqual(result_a[1:4], result_b[1:4])
                    other = make_environment(spec)
                    try:
                        self.assertIsNot(cached.scenario, other.scenario)
                        self.assertEqual(other.steps, 0)
                    finally: other.close()
                finally: cached.close(); fresh.close()

    def test_masked_inference_and_episode_boundaries_both_profiles(self):
        before=[p.detach().clone() for net in self.models for p in net.parameters()]
        for profile,banks in [('nasim_current',self.banks),('nasim_scan_penalty',self.old)]:
            specs=[next(s for s in banks['test'] if s['difficulty']==d) for d in range(3)]
            embeddings,rows,episodes=collect(self.models,specs,profile)
            self.assertEqual(len(episodes),3)
            self.assertTrue(all(e['length']<=400 for e in episodes))
            self.assertEqual(sum(r['terminated'] or r['truncated'] for r in rows),3)
            self.assertTrue(all(r['valid_actions']>0 for r in rows))
            self.assertTrue(all(np.isfinite(e).all() for e in embeddings))
            self.assertTrue(all(np.isfinite(r['reward']) for r in rows))
            self.assertTrue(all(abs(sum(v for k,v in r['reward_components'].items() if k!='total')-r['reward'])<1e-6 for r in rows))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,[p for n in self.models for p in n.parameters()])))

    def test_shared_metric_equivalence_and_no_untrained_verdict(self):
        embeddings,rows,_=collect(self.models,self.banks['test'][:2],'nasim_current',step_cap=12)
        r=analyze(embeddings,copy.deepcopy(rows),self.models[2],trained=False)
        self.assertIsNone(r['hackability_score'])
        self.assertFalse(r['trained_checkpoint_found'])
        trajectory=Trajectory(np.stack(embeddings),np.zeros((len(rows),1)),
                              np.array([r['reward'] for r in rows]),
                              np.array([r['terminated'] for r in rows]),np.array([r['truncated'] for r in rows]))
        def critic(obs,actions=None):
            with torch.no_grad(): return self.models[2](torch.as_tensor(obs,dtype=torch.float32)).squeeze(-1).numpy()
        for name,fn in [('td_error_anomaly',td_error_anomaly),('return_calibration_gap',return_calibration_gap),('reward_concentration',reward_concentration)]:
            self.assertEqual(r['raw_metrics'][name],fn(trajectory,critic))
        self.assertAlmostEqual(sum(PROFILE['weights'].values()),1)
        self.assertAlmostEqual(PROFILE['original_weight_coverage'],.5)
        self.assertNotIn('action_saturation',r['sub_scores'])
        self.assertTrue(all(row['signal'] is not None for row in r['timeline']['steps']))
        self.assertEqual(r['timeline']['steps'][-1]['boundary_reason'],'diagnostic_step_cap')

    def test_checkpoint_requires_training_and_matching_reward(self):
        # Temporary, explicitly incomplete fixture; never emits a trained report.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest=make_manifest();fingerprint=bank_fingerprint(self.banks)
            with self.assertRaises(FileNotFoundError): load_checkpoint(root,manifest,fingerprint)
            optimizer=torch.optim.Adam([p for n in self.models for p in n.parameters()])
            save_checkpoint(root,self.models,optimizer,manifest,0,fingerprint,complete=False)
            with self.assertRaisesRegex(ValueError,'completed trained'):load_checkpoint(root,manifest,fingerprint)
            with self.assertRaisesRegex(ValueError,'reward'):load_checkpoint(root,make_manifest('nasim_scan_penalty'),fingerprint)
            with self.assertRaisesRegex(ValueError,'Scenario'):load_checkpoint(root,manifest,'wrong')
            save_json(root/'experiment.json',manifest)
            self.assertEqual(read_manifest(root),manifest)

    def test_server_routes_one_agent_without_starting_training(self):
        with tempfile.TemporaryDirectory() as folder:
            controller=Controller(folder)
            data={'mode':'train','environment':ENVIRONMENT,'experiment':'nasim','agents':['graph_ppo'],'steps':102400}
            with patch('fallax_toolkit.server.threading.Thread'):
                job=controller.start(data)
            output=Path(folder)/'experiments/nasim'
            self.assertEqual(read_manifest(output)['agents'],['graph_ppo'])
            # Popen mocked: no process and no optimization are performed.
            with patch('fallax_toolkit.server.subprocess.Popen',return_value=SimpleNamespace(wait=lambda:0)) as process:
                controller._run(job['id'],output)
            command=process.call_args.args[0]
            self.assertIn('fallax_toolkit.nasim.runner',command)
            self.assertIn(str(output),command)
            self.assertEqual(controller.experiments()[0]['environment'],ENVIRONMENT)
            with patch('fallax_toolkit.server.threading.Thread'):
                report_job=controller.start({'mode':'report','experiment':'nasim','agents':['graph_ppo']})
            self.assertEqual(report_job['profile'],'nasim_current')

    def test_incompatible_agents_and_profiles_rejected(self):
        for environment,agent,profile in [(ENVIRONMENT,'ppo','nasim_current'),('BipedalWalker-v3','graph_ppo','original'),(ENVIRONMENT,'graph_ppo','optimized_v1')]:
            with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                Controller(folder).start({'mode':'train','environment':environment,'experiment':'x','agents':[agent],'profile':profile})

    def test_preflight_never_updates_weights(self):
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Training forbidden')), patch.object(torch.Tensor,'backward',side_effect=AssertionError('Backward forbidden')):
            result=preflight()
        self.assertEqual(result['optimizer_steps'],0)
        self.assertTrue(result['weights_unchanged'])
        self.assertTrue(result['ppo_loss_forward_checked'])


if __name__ == '__main__': unittest.main()
