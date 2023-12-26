import os
import time
import json
import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym

from agents.dqn import DQN
from agents.d3qn import D3QN
from agents.ddpg import DDPG
from agents.td3 import TD3
from envs.atari_wrappers import NoopResetEnv, FireResetEnv, EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame, PyTorchFrame, FrameStack
from utils.config_utils import config_parse
from utils.env_utils import setup_seed
from utils.metric_utils import AverageMeter, DiscountedMeter


def train(args):
    # initialize environment and agent
    env_name = args.env_name
    improve = args.improve
    if env_name in ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4']:
        if args.gui:
            train_env = gym.make(env_name, render_mode='human')
            val_env = gym.make(env_name, render_mode='human')
        else:
            train_env = gym.make(env_name)
            val_env = gym.make(env_name)
        train_env.action_space.seed(args.seed)
        val_env.action_space.seed(args.seed)
        train_env = NoopResetEnv(train_env)
        train_env = FireResetEnv(train_env)
        train_env = EpisodicLifeEnv(train_env)
        train_env = MaxAndSkipEnv(train_env)
        # val_env = NoopResetEnv(val_env)
        # val_env = FireResetEnv(val_env)
        # val_env = EpisodicLifeEnv(val_env)
        val_env = MaxAndSkipEnv(val_env)
        if '-ram' in env_name:
            state_mode = 'vec'
            state_dim = train_env.observation_space.shape[0]    # Nstate
        else:
            state_mode = 'img'
            train_env = WarpFrame(train_env)
            train_env = PyTorchFrame(train_env)
            train_env = FrameStack(train_env)
            val_env = WarpFrame(val_env)
            val_env = PyTorchFrame(val_env)
            val_env = FrameStack(val_env)
            state_dim = train_env.observation_space.shape       # (C, H, W)
        action_dim = train_env.action_space.n                   # Naction
        if not improve:
            algo_name = 'DQN'
            with open(os.path.join('configs', env_name + '.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = DQN(state_mode, state_dim, action_dim, 
                        config['buffer_size'], config['sync_freq'], config['epsilon_max'], config['epsilon_min'], config['epsilon_frac'], 
                        config['learn_start'], config['learn_freq'], 
                        config['reward_center'], config['reward_scale'], 
                        gamma, config['lr'], config['batch_size'], iteration_num, args.device)
        else:
            algo_name = 'D3QN'
            with open(os.path.join('configs', env_name + '-improve.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = D3QN(state_mode, state_dim, action_dim, 
                         config['buffer_size'], config['sync_freq'], config['epsilon_max'], config['epsilon_min'], config['epsilon_frac'], 
                         config['learn_start'], config['learn_freq'], 
                         config['reward_center'], config['reward_scale'], 
                         gamma, config['lr'], config['batch_size'], iteration_num, args.device)
    elif env_name in ['Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Ant-v2']:
        if args.gui:
            train_env = gym.make(env_name, render_mode='human')
            val_env = gym.make(env_name, render_mode='human')
        else:
            train_env = gym.make(env_name)
            val_env = gym.make(env_name)
        train_env.action_space.seed(args.seed)
        val_env.action_space.seed(args.seed)
        state_dim = train_env.observation_space.shape[0]        # Nstate
        action_dim = train_env.action_space.shape[0]            # Naction
        action_max = train_env.action_space.high                # (Naction,)
        action_min = train_env.action_space.low                 # (Naction,)
        state_mode = 'vec'
        if not improve:
            algo_name = 'DDPG'
            with open(os.path.join('configs', env_name + '.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = DDPG(state_mode, state_dim, action_dim, 
                         config['buffer_size'], config['tau'], config['noise_max'], config['noise_min'], config['noise_frac'], 
                         config['learn_start'], config['learn_freq'], action_max, action_min, 
                         gamma, config['actor_lr'], config['critic_lr'], config['batch_size'], iteration_num, args.device)
        else:
            algo_name = 'TD3'
            with open(os.path.join('configs', env_name + '-improve.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = TD3(state_mode, state_dim, action_dim, 
                        config['buffer_size'], config['tau'], config['noise_max'], config['noise_min'], config['noise_frac'], config['target_noise'], config['target_noise_clip'], 
                        config['learn_start'], config['learn_freq'], config['policy_learn_freq'], action_max, action_min, 
                        gamma, config['actor_lr'], config['critic_lr'], config['batch_size'], iteration_num, args.device)
    else:
        raise NotImplementedError
    print('train ' + algo_name + ' on ' + env_name)
    output_dir = os.path.join('outputs', env_name, algo_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tb'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'weights'), exist_ok=True)

    # train
    tb_writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tb'))
    reward_meter = AverageMeter()                       # overall reward, average over all iterations
    return_meter = DiscountedMeter(gamma)               # episode return, accumulated over one episode
    score_meter = DiscountedMeter(1)                    # episode score, sum over one episode
    s, _info = train_env.reset()
    state_sample = s.copy()
    current_episode = 0
    this_iteration_num = 0
    for iteration in tqdm.trange(iteration_num):
        if args.debug:
            print("episode:", current_episode, "iteration:", this_iteration_num, "overall iteration:", iteration)
        
        # interaction
        start_time = time.time()
        agent.set_mode('eval')
        a = agent.take_explore_action(s)
        this_iteration_num += 1
        end_time = time.time()
        if args.debug:
            print("interaction time:", end_time - start_time)
        
        # step
        start_time = time.time()
        s_prime, r, terminated, truncated, _info = train_env.step(a)
        end_time = time.time()
        if args.debug:
            print("step time:", end_time - start_time)
        # update transition
        start_time = time.time()
        agent.update_transition(s, a, r, s_prime, terminated)
        end_time = time.time()
        if args.debug:
            print("update transition time:", end_time - start_time)
        reward_meter.update(r)
        return_meter.update(r)
        score_meter.update(r)
        tb_writer.add_scalar('reward', r, iteration)
        tb_writer.add_scalar('mean_reward', reward_meter.avg, iteration)
        
        # learn
        start_time = time.time()
        agent.set_mode('train')
        loss = agent.learn(args.debug)
        if loss is not None:
            for loss_name in loss.keys():
                tb_writer.add_scalar(loss_name, loss[loss_name], iteration)
        end_time = time.time()
        if args.debug:
            print("learn time:", end_time - start_time)
        # update iteration
        start_time = time.time()
        agent.update_iteration(iteration, writer=tb_writer, state_sample=state_sample)
        end_time = time.time()
        if args.debug:
            print("update iteration time:", end_time - start_time)
        
        # update episode
        if terminated or truncated:
            print("episode:", current_episode, "iteration:", this_iteration_num, "overall iteration:", iteration, "return:", return_meter.val, "score:", score_meter.val)
            tb_writer.add_scalar('iteration_num', this_iteration_num, current_episode)
            tb_writer.add_scalar('return', return_meter.val, current_episode)
            tb_writer.add_scalar('score', score_meter.val, current_episode)
            start_time = time.time()
            agent.update_episode(current_episode, writer=tb_writer, state_sample=state_sample)
            end_time = time.time()
            if args.debug:
                print("update episode time:", end_time - start_time)
            s, _info = train_env.reset()
            return_meter.reset()
            score_meter.reset()
            current_episode += 1
            this_iteration_num = 0
        else:
            s = s_prime.copy()
        
        # validate
        if iteration % (iteration_num//100) == 0:
            agent.set_mode('eval')
            val_return_meter = DiscountedMeter(gamma)
            val_score_meter = DiscountedMeter(1)
            val_iteration_num = 0
            val_s, _info = val_env.reset()
            while True:
                val_iteration_num += 1
                val_a = agent.take_optimal_action(val_s)
                val_s_prime, val_r, val_terminated, val_truncated, _info = val_env.step(val_a)
                val_return_meter.update(val_r)
                val_score_meter.update(val_r)
                if val_terminated or val_truncated:
                    break
                val_s = val_s_prime.copy()
            print("validate:", "iteration:", iteration, "return:", val_return_meter.val, "score:", val_score_meter.val, "iteration_num:", val_iteration_num)
            tb_writer.add_scalar('val_return', val_return_meter.val, iteration)
            tb_writer.add_scalar('val_score', val_score_meter.val, iteration)
            tb_writer.add_scalar('val_iteration_num', val_iteration_num, iteration)
    train_env.close()
    val_env.close()
    agent.save(os.path.join(output_dir, 'weights'))


def test(args):
    # initialize environment and agent
    env_name = args.env_name
    improve = args.improve
    if env_name in ['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4']:
        if args.gui:
            test_env = gym.make(env_name, render_mode='human')
        else:
            test_env = gym.make(env_name)
        test_env.action_space.seed(args.seed)
        test_env = NoopResetEnv(test_env)
        # test_env = FireResetEnv(test_env)
        # test_env = EpisodicLifeEnv(test_env)
        test_env = MaxAndSkipEnv(test_env)
        if '-ram' in env_name:
            state_mode = 'vec'
            state_dim = test_env.observation_space.shape[0]     # Nstate
        else:
            state_mode = 'img'
            test_env = WarpFrame(test_env)
            test_env = PyTorchFrame(test_env)
            test_env = FrameStack(test_env)
            state_dim = test_env.observation_space.shape        # (C, H, W)
        action_dim = test_env.action_space.n                    # Naction
        if not improve:
            algo_name = 'DQN'
            with open(os.path.join('configs', env_name + '.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = DQN(state_mode, state_dim, action_dim, 
                        config['buffer_size'], config['sync_freq'], config['epsilon_max'], config['epsilon_min'], config['epsilon_frac'], 
                        config['learn_start'], config['learn_freq'], 
                        config['reward_center'], config['reward_scale'], 
                        gamma, config['lr'], config['batch_size'], iteration_num, args.device, is_train=False)
        else:
            algo_name = 'D3QN'
            with open(os.path.join('configs', env_name + '-improve.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = D3QN(state_mode, state_dim, action_dim, 
                         config['buffer_size'], config['sync_freq'], config['epsilon_max'], config['epsilon_min'], config['epsilon_frac'], 
                         config['learn_start'], config['learn_freq'], 
                         config['reward_center'], config['reward_scale'], 
                         gamma, config['lr'], config['batch_size'], iteration_num, args.device, is_train=False)
    elif env_name in ['Hopper-v2', 'Humanoid-v2', 'HalfCheetah-v2', 'Ant-v2']:
        if args.gui:
            test_env = gym.make(env_name, render_mode='human')
        else:
            test_env = gym.make(env_name)
        test_env.action_space.seed(args.seed)
        state_dim = test_env.observation_space.shape[0]         # Nstate
        action_dim = test_env.action_space.shape[0]             # Naction
        action_max = test_env.action_space.high                 # (Naction,)
        action_min = test_env.action_space.low                  # (Naction,)
        state_mode = 'vec'
        if not improve:
            algo_name = 'DDPG'
            with open(os.path.join('configs', env_name + '.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = DDPG(state_mode, state_dim, action_dim, 
                         config['buffer_size'], config['tau'], config['noise_max'], config['noise_min'], config['noise_frac'], 
                         config['learn_start'], config['learn_freq'], action_max, action_min, 
                         gamma, config['actor_lr'], config['critic_lr'], config['batch_size'], iteration_num, args.device, is_train=False)
        else:
            algo_name = 'TD3'
            with open(os.path.join('configs', env_name + '-improve.json'), 'r') as f:
                config = json.load(f)
            iteration_num = config['iteration_num']
            gamma = config['gamma']
            agent = TD3(state_mode, state_dim, action_dim, 
                        config['buffer_size'], config['tau'], config['noise_max'], config['noise_min'], config['noise_frac'], config['target_noise'], config['target_noise_clip'], 
                        config['learn_start'], config['learn_freq'], config['policy_learn_freq'], action_max, action_min, 
                        gamma, config['actor_lr'], config['critic_lr'], config['batch_size'], iteration_num, args.device, is_train=False)
    else:
        raise NotImplementedError
    print('test ' + algo_name + ' on ' + env_name)
    output_dir = os.path.join('outputs', env_name, algo_name)
    agent.load(os.path.join(output_dir, 'weights'))

    return_meter = DiscountedMeter(gamma)
    score_meter = DiscountedMeter(1)
    test_returns = []
    test_scores = []
    test_iteration_nums = []
    agent.set_mode('eval')
    for episode in tqdm.trange(10):
        # start a new episode
        return_meter.reset()
        score_meter.reset()
        this_iteration_num = 0
        s, _info = test_env.reset()
        # interaction
        while True:
            this_iteration_num += 1
            a = agent.take_optimal_action(s)
            s_prime, r, terminated, truncated, _info = test_env.step(a)
            return_meter.update(r)
            score_meter.update(r)
            if terminated or truncated:
                break
            s = s_prime.copy()
        test_returns.append(return_meter.val)
        test_scores.append(score_meter.val)
        test_iteration_nums.append(this_iteration_num)
    # print and save mean, median, max, min, std of returns, scores, iteration_nums
    print("return:", np.mean(test_returns), np.median(test_returns), np.max(test_returns), np.min(test_returns), np.std(test_returns))
    print("score:", np.mean(test_scores), np.median(test_scores), np.max(test_scores), np.min(test_scores), np.std(test_scores))
    print("iteration_num:", np.mean(test_iteration_nums), np.median(test_iteration_nums), np.max(test_iteration_nums), np.min(test_iteration_nums), np.std(test_iteration_nums))
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        print("return:", np.mean(test_returns), np.median(test_returns), np.max(test_returns), np.min(test_returns), np.std(test_returns), file=f)
        print("score:", np.mean(test_scores), np.median(test_scores), np.max(test_scores), np.min(test_scores), np.std(test_scores), file=f)
        print("iteration_num:", np.mean(test_iteration_nums), np.median(test_iteration_nums), np.max(test_iteration_nums), np.min(test_iteration_nums), np.std(test_iteration_nums), file=f)
    test_env.close()


if __name__ == '__main__':
    # config
    args = config_parse()
    setup_seed(args.seed)

    if not args.test:
        # train
        train(args)
    else:
        # test
        test(args)
