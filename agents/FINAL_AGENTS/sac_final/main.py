from utilis import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
import argparse
import torch
import numpy as np


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(0.5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
parser.add_argument('--plot_rewards', type=str2bool, default=True,
                    help='Save a 100-episode moving-average training-reward plot')
opt = parser.parse_args()
if opt.dvc == 'cuda' and not torch.cuda.is_available():
    print('CUDA is unavailable; using CPU instead.')
    opt.dvc = 'cpu'
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def save_reward_plot(episode_scores):
    """Match TD3's plot: a 100-episode average of raw training returns."""
    if not episode_scores:
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('plots', exist_ok=True)
    episodes = range(1, len(episode_scores) + 1)
    running_average = [
        np.mean(episode_scores[max(0, index - 99):index + 1])
        for index in range(len(episode_scores))
    ]
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, running_average)
    plt.xlabel('Episode')
    plt.ylabel('Training return (100-episode average)')
    plt.title('SAC on BipedalWalker-v3')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'sac_learning_curve.png'), dpi=150)
    plt.close()


def main():
    EnvName = 'BipedalWalker-v3'
    BrifEnvName = 'BWv3'
    reward_env_index = 4

    # Build Env
    env = gym.make(EnvName, render_mode = "human" if opt.render else None)
    eval_env = gym.make(EnvName)
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:{EnvName}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName, opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, opt.max_action, agent, turns=1)
            print('EnvName:', BrifEnvName, 'score:', score)
    else:
        total_steps = 0
        episode_scores = []
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            episode_score = 0.0

            '''Interact & trian'''
            while not done:
                if total_steps < (5*opt.max_e_steps):
                    act = env.action_space.sample()  # act∈[-max,max]
                    a = Action_adapter_reverse(act, opt.max_action)  # a∈[-1,1]
                else:
                    a = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                    act = Action_adapter(a, opt.max_action)  # act∈[-max,max]
                s_next, r, dw, tr, info = env.step(act)  # dw: dead&win; tr: truncated
                episode_score += r  # Raw environment reward, matching TD3's plotted metric.
                r = Reward_adapter(r, reward_env_index)
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train if it's time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if (total_steps >= 2*opt.max_e_steps) and (total_steps % opt.update_every == 0):
                    for j in range(opt.update_every):
                        agent.train()

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, opt.max_action, agent, turns=3)
                    if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName, int(total_steps/1000))
            episode_scores.append(episode_score)
            if opt.write:
                writer.add_scalar('training_episode_reward', episode_score, global_step=len(episode_scores))
        if opt.plot_rewards:
            save_reward_plot(episode_scores)
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()
