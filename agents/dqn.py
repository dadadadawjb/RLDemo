from typing import Tuple, Union, Dict, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from utils.data_utils import ReplayBuffer


class QMLP(nn.Module):
    def __init__(self, in_features:int, action_dim:int) -> None:
        super(QMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = 128),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 128, out_features = 64),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 64, out_features = 32),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 32, out_features = 32),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 32, out_features = action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class QCNN(nn.Module):
    def __init__(self, in_channels:int, action_dim:int) -> None:
        super(QCNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(inplace = True),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(in_features = 64 * 7 * 7, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class DQN(Agent):
    def __init__(self, state_mode:str, state_dim:Union[int, Tuple[int]], action_dim:int, 
                 buffer_size:int, sync_freq:int, epsilon_max:float, epsilon_min:float, epsilon_frac:float, 
                 learn_start:int, learn_freq:int, 
                 reward_center:float, reward_scale:float, 
                 gamma:float, lr:float, batch_size:int, iteration_num:int, device:str, is_train:bool=True) -> None:
        self.state_mode = state_mode
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.sync_freq = sync_freq
        self.sync_counter = 0
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_frac = epsilon_frac
        self.epsilon = epsilon_max
        self.epsilon_iteration_num = epsilon_frac * iteration_num
        self.epsilon_counter = 0
        self.learn_start = learn_start
        self.learn_freq = learn_freq
        self.learn_counter = 0

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.iteration_num = iteration_num
        self.device = device
        self.is_train = is_train

        self.reward_center = reward_center
        self.reward_scale = reward_scale

        if is_train:
            self.buffer = ReplayBuffer(state_mode, state_dim, 'disc', 1, buffer_size)

        if state_mode == 'vec':
            self.QNet = QMLP(state_dim, action_dim).to(self.device)
            if is_train:
                self.QNet_target = QMLP(state_dim, action_dim).to(self.device)
        elif state_mode == 'img':
            self.QNet = QCNN(state_dim[0], action_dim).to(self.device)
            if is_train:
                self.QNet_target = QCNN(state_dim[0], action_dim).to(self.device)
        else:
            raise ValueError(f"state_mode {state_mode} is not defined")
        if is_train:
            self.QNet_target.load_state_dict(self.QNet.state_dict())
            self.QNet_target.eval()

        if is_train:
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.QNet.parameters(), lr=lr)
    
    def take_explore_action(self, state:np.ndarray) -> int:
        # epsilon-greedy
        prob = np.random.random()
        if prob < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action = self.take_optimal_action(state)
        return action
    
    def take_optimal_action(self, state:np.ndarray) -> int:
        # greedy
        state = torch.from_numpy(state / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q = self.QNet(state)
            _, action = Q.max(1)
        return action.item()

    def update_transition(self, state:np.ndarray, action:int, reward:float, state_prime:np.ndarray, done:bool) -> None:
        self._update_buffer(state, action, reward, state_prime, done)

    def _update_buffer(self, state:np.ndarray, action:int, reward:float, state_prime:np.ndarray, done:bool) -> None:
        self.buffer.add(state.astype(np.float32), action, reward, state_prime.astype(np.float32), done)

    def _sample_buffer(self, batch_size:int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        batch_transitions = self.buffer.sample(batch_size)
        return batch_transitions

    def learn(self, debug:bool=False) -> Optional[Dict[str, float]]:
        if not self.is_train:
            raise ValueError("agent is not in train mode")
        
        self.learn_counter += 1
        if self.learn_counter <= self.learn_start or self.learn_counter % self.learn_freq != 0:
            return None

        start_time = time.time()
        # sample batch transitions
        batch_transitions = self._sample_buffer(self.batch_size)
        if batch_transitions is None:
            return None
        
        # unpack batch transitions
        batch_states, batch_actions, batch_rewards, batch_states_prime, batch_dones = batch_transitions
        states = torch.from_numpy(batch_states / 255.0).float().to(self.device)
        actions = torch.from_numpy(batch_actions).unsqueeze(1).long().to(self.device)
        rewards = torch.from_numpy((batch_rewards - self.reward_center) / self.reward_scale).float().to(self.device)
        next_states = torch.from_numpy(batch_states_prime / 255.0).float().to(self.device)
        dones = torch.from_numpy(batch_dones).float().to(self.device)
        end_time = time.time()
        if debug:
            print("data prepare time:", end_time - start_time)
        
        # calculate loss
        start_time = time.time()
        qs = self.QNet(states)                                      # (B, A)
        q = torch.gather(qs, dim=1, index=actions).squeeze(1)       # (B,)
        with torch.no_grad():
            qs_prime = self.QNet_target(next_states).detach()       # (B, A)
            q_prime = torch.max(qs_prime, dim=1)[0]                 # (B,)
            q_target = rewards + self.gamma * q_prime * (1 - dones) # (B,)
        loss = self.criterion(q, q_target)                          # q_target semi-gradient and target delay
        end_time = time.time()
        if debug:
            print("forward time", end_time - start_time)
        
        # update Q function
        start_time = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        end_time = time.time()
        if debug:
            print("backward time", end_time - start_time)
        
        return {'loss': loss.item()}
    
    def update_iteration(self, iteration:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        if writer is not None:
            writer.add_scalar('epsilon', self.epsilon, iteration)
            if state_sample is not None and iteration % (self.iteration_num//100) == 0:
                state = torch.from_numpy(state_sample / 255.0).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    qs = self.QNet(state)[0]
                writer.add_hparams({'sample/iteration': iteration}, 
                                   {f'sample/q_{a_idx:0>2d}': qs[a_idx].item() for a_idx in range(qs.shape[0])}, 
                                   run_name='run'+str(iteration).zfill(len(str(self.iteration_num))))
        
        self._sync_target()
        self._update_epsilon()
    
    def _sync_target(self) -> None:
        self.sync_counter += 1
        if self.sync_counter % self.sync_freq == 0:
            self.QNet_target.load_state_dict(self.QNet.state_dict())
    
    def _update_epsilon(self) -> None:
        fraction = min(1.0, self.epsilon_counter / self.epsilon_iteration_num)
        self.epsilon = self.epsilon_max + fraction * (self.epsilon_min - self.epsilon_max)
        self.epsilon_counter += 1

    def update_episode(self, episode:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        pass

    def set_mode(self, mode:str) -> None:
        if mode == 'train':
            self.QNet.train()
        elif mode == 'eval':
            self.QNet.eval()
        else:
            raise ValueError(f"mode {mode} is not defined")

    def save(self, path:str) -> None:
        torch.save(self.QNet.state_dict(), os.path.join(path, 'QNet.pth'))

    def load(self, path:str) -> None:
        self.QNet.load_state_dict(torch.load(os.path.join(path, 'QNet.pth'), map_location=self.device))
        if self.is_train:
            self.QNet_target.load_state_dict(self.QNet.state_dict())
            self.QNet_target.eval()
