"""Dashboard tests: isolated fixtures and mocked jobs, never real training."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fallax_toolkit.server import Controller, ThreadingHTTPServer, make_handler, write_json
from fallax_toolkit.timeline import build_timeline


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.controller = Controller(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_discovery_preserves_legacy_and_incomplete_experiment(self):
        write_json(self.root / 'runs/hackability_report.json', {'ppo': {'hackability_score': .2}})
        write_json(self.root / 'experiments/new/experiment.json', {'agents':['ppo'], 'reward':{'profile':'optimized_v1'}})
        items = self.controller.experiments()
        self.assertEqual([e['id'] for e in items], ['legacy','new'])
        self.assertTrue(items[0]['readonly'])
        self.assertFalse(items[1]['report_available'])
        self.assertEqual(items[1]['summary'], {})

    def test_paths_cannot_escape_workspace(self):
        for value in ('../outside', 'a/b', 'a\\b', '/absolute', '', None):
            with self.assertRaises(ValueError):
                self.controller.experiment(value)

    def test_training_records_selection_and_budget_without_launch(self):
        with patch('fallax_toolkit.server.threading.Thread') as thread:
            job = self.controller.start({'mode':'train','experiment':'fresh','agents':['ppo','sac'],'steps':500000})
            thread.return_value.start.assert_called_once()
        manifest = json.loads((self.root/'experiments/fresh/experiment.json').read_text())
        self.assertEqual(manifest['agents'], ['ppo','sac'])
        self.assertEqual(manifest['budgets']['sac'], 500000)
        self.assertEqual(job['status'], 'queued')
        with self.assertRaises(ValueError):
            self.controller.start({'mode':'train','experiment':'other','agents':['td3']})

    def test_validation_and_protection(self):
        cases = [
            {'mode':'train','experiment':'legacy','agents':['ppo']},
            {'mode':'train','experiment':'x','agents':[]},
            {'mode':'train','experiment':'x','agents':['unknown']},
            {'mode':'train','experiment':'x','agents':['ppo','ppo']},
            {'mode':'train','experiment':'x','agents':['ppo'],'steps':10},
            {'mode':'train','experiment':'x','agents':['ppo'],'profile':'bad'},
        ]
        for data in cases:
            with self.assertRaises(ValueError): self.controller.start(data)
        write_json(self.root/'experiments/taken/experiment.json', {'keep':True})
        with self.assertRaises(ValueError):
            self.controller.start({'mode':'train','experiment':'taken','agents':['ppo']})
        self.assertEqual(json.loads((self.root/'experiments/taken/experiment.json').read_text()), {'keep':True})

    def test_report_requires_completed_agents(self):
        write_json(self.root/'experiments/x/experiment.json', {'reward':{'profile':'original'}})
        write_json(self.root/'experiments/x/summary.json', {'ppo':'ok','sac':'failed'})
        with self.assertRaises(ValueError):
            self.controller.start({'mode':'report','experiment':'x','agents':['sac']})
        with patch('fallax_toolkit.server.threading.Thread'):
            job = self.controller.start({'mode':'report','experiment':'x','agents':['ppo']})
        self.assertEqual(job['profile'], 'original')

    def test_only_owned_jobs_can_be_stopped(self):
        with self.assertRaises(ValueError): self.controller.stop('external')
        with patch('fallax_toolkit.server.threading.Thread'):
            job = self.controller.start({'mode':'train','experiment':'x','agents':['ppo']})
        result = self.controller.stop(job['id'])
        self.assertEqual(result['status'], 'stopping')

    def test_worker_success_and_failure_lifecycle(self):
        with patch('fallax_toolkit.server.threading.Thread'):
            job = self.controller.start({'mode':'train','experiment':'x','agents':['ppo','td3']})
        processes = [SimpleNamespace(wait=lambda:0), SimpleNamespace(wait=lambda:1)]
        with patch('fallax_toolkit.server.subprocess.Popen', side_effect=processes) as process:
            self.controller._run(job['id'], self.root/'experiments/x')
        self.assertEqual(process.call_count, 2)
        self.assertEqual(self.controller.jobs[job['id']]['status'], 'failed')
        summary = json.loads((self.root/'experiments/x/summary.json').read_text())
        self.assertEqual(summary, {'ppo':'ok','td3':'failed: exit 1'})
        args = process.call_args_list[0]
        self.assertEqual(args.kwargs['cwd'], self.root/'experiments/x')
        self.assertEqual(args.kwargs['env']['WALKER_REWARD_PROFILE'], 'optimized_v1')

    def test_http_static_report_and_request_protection(self):
        write_json(self.root/'runs/hackability_report.json', {'ppo':{'hackability_score':.25}})
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.controller))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
        try:
            for path in ('/', '/app.js', '/style.css', '/api/state', '/api/report?experiment=legacy'):
                conn.request('GET', path)
                response = conn.getresponse()
                self.assertEqual(response.status, 200, path)
                self.assertTrue(response.read())
            conn.request('POST','/api/jobs',body='{}',headers={'Content-Type':'application/json'})
            response=conn.getresponse(); self.assertEqual(response.status,403); response.read()
            conn.request('POST','/api/jobs',body='{}',headers={'X-Fallax-Token':self.controller.token,'Origin':'https://untrusted.example'})
            response=conn.getresponse(); self.assertEqual(response.status,403); response.read()
            conn.request('GET','/api/state',headers={'Host':'untrusted.example'})
            response=conn.getresponse(); self.assertEqual(response.status,403); response.read()
        finally:
            conn.close();server.shutdown();server.server_close();worker.join()

    def test_timeline_boundaries_and_hotspot_rank(self):
        trajectory = SimpleNamespace(rewards=np.array([0.,10.,0.,0.]),
                    obs=np.zeros((4,2)),actions=np.array([[0.,0.],[1.,1.],[0.,0.],[0.,0.]]),
                    terminated=np.array([False,True,False,False]),truncated=np.zeros(4,dtype=bool))
        timeline = build_timeline(trajectory, lambda o,a: np.zeros(4))
        self.assertEqual(timeline['hotspots'][0], 2)
        self.assertIsNone(timeline['steps'][-1]['signal'])
        self.assertEqual(timeline['steps'][2]['episode'], 2)
        self.assertEqual(timeline['steps'][2]['episode_step'], 1)
        self.assertEqual(timeline['steps'][1]['td_residual'], 10)
        self.assertTrue(all(0<=r['signal']<=1 for r in timeline['steps'] if r['signal'] is not None))


if __name__ == '__main__':
    unittest.main()
