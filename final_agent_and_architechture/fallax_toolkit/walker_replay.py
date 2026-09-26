"""Record one fresh checkpoint evaluation episode; never train or replace reports."""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from run_hackability_report import AGENTS
from metrics import Trajectory
from fallax_toolkit.timeline import build_timeline
from fallax_toolkit.server import write_json
from reward_wrapper import reward_metadata


def record(root, agent, seed=42):
    manifest = root / 'experiment.json'
    profile = 'original'
    if manifest.exists():
        reward = json.loads(manifest.read_text(encoding='utf-8'))['reward']
        profile = reward['profile']
        if reward != reward_metadata(profile):
            raise ValueError('Checkpoint reward configuration does not match this runtime')
    os.environ['WALKER_REWARD_PROFILE'] = profile
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.chdir(root)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env, act, critic, trained = AGENTS[agent]()
    try:
        if not trained:
            raise ValueError('No trained checkpoint found')
        env.unwrapped.render_mode = 'rgb_array'
        obs, _ = env.reset(seed=seed)
        observations, actions, rewards, terms, truncs, evidence = [], [], [], [], [], []
        for i in range(env.spec.max_episode_steps or 1600):
            before_x = float(env.unwrapped.hull.position.x)
            action = act(obs)
            nxt, reward, terminated, truncated, info = env.step(action)
            frame = Image.fromarray(env.render())
            frame.thumbnail((600, 400))
            buffer = io.BytesIO()
            frame.save(buffer, format='JPEG', quality=72)
            observations.append(obs); actions.append(action); rewards.append(reward)
            terms.append(terminated); truncs.append(truncated)
            evidence.append({'frame': 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                             'native_reward': info.get('native_reward'),
                             'extrinsic_reward': info.get('optimized_reward'),
                             'reward_components': info.get('reward_terms', {}),
                             'forward_displacement': float(env.unwrapped.hull.position.x) - before_x,
                             'failure': bool(terminated and (env.unwrapped.game_over or env.unwrapped.hull.position.x < 0))})
            obs = nxt
            if terminated or truncated:
                break
        traj = Trajectory(obs=np.asarray(observations), actions=np.asarray(actions), rewards=np.asarray(rewards),
                          terminated=np.asarray(terms), truncated=np.asarray(truncs))
        timeline = build_timeline(traj, critic)
        for row, extra in zip(timeline['steps'], evidence):
            row.update(extra)
        result = {'agent': agent, 'profile': profile, 'seed': seed,
                  'created': datetime.now(timezone.utc).isoformat(), 'fps': env.metadata.get('render_fps', 50),
                  'reward_label': 'Normalized evaluation reward' if agent == 'ppo' else 'Evaluation reward',
                  'description': 'Fresh evaluation episode, separate from the aggregate report. Each frame shows the state after its recorded action. No training occurred.',
                  'timeline': timeline}
        write_json(root / 'runs' / 'replays' / (agent + '.json'), result)
        print(f'{agent}: recorded {len(rewards)} frames', flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--agents', nargs='+', choices=list(AGENTS), default=list(AGENTS))
    args = parser.parse_args()
    root = args.experiment.resolve()
    failures = {}
    for agent in args.agents:
        try:
            record(root, agent)
        except Exception as exc:
            failures[agent] = str(exc)
            print(f'{agent}: unavailable: {exc}', flush=True)
    write_json(root / 'runs' / 'replays' / 'status.json', {'errors': failures})


if __name__ == '__main__':
    main()
