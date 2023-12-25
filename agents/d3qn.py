from typing import Tuple, Union, Dict, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch.utils.tensorboard import SummaryWriter

from .dqn import DQN, QMLP, QCNN
from utils.data_utils import ReplayBuffer


class VAMLP(nn.Module):
    def __init__(self, in_features:int, action_dim:int) -> None:
        super(VAMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features = in_features, out_features = 128),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 128, out_features = 64),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 64, out_features = 32),
            nn.ReLU(inplace = True)
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 32),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 32, out_features = action_dim)
        )

        self.val_layer = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 32),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 32, out_features = 1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x)
        res = val + adv - adv.mean(dim=-1, keepdim=True)
        return res


class VACNN(nn.Module):
    def __init__(self, in_channels:int, action_dim:int) -> None:
        super(VACNN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(inplace = True),
            Rearrange('b c h w -> b (c h w)')
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(in_features = 64 * 7 * 7, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = action_dim)
        )

        self.val_layer = nn.Sequential(
            nn.Linear(in_features = 64 * 7 * 7, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 512, out_features = 1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        adv = self.adv_layer(x)
        val = self.val_layer(x)
        res = val + adv - adv.mean(dim=-1, keepdim=True)
        return res


class D3QN(DQN):
    def __init__(self, state_mode:str, state_dim:Union[int, Tuple[int]], action_dim:int, 
                 buffer_size:int, sync_freq:int, epsilon_max:float, epsilon_min:float, epsilon_frac:float, 
                 learn_start:int, learn_freq:int, 
                 reward_center:float, reward_scale:float, 
                 gamma:float, lr:float, batch_size:int, iteration_num:int, device:str, 
                 double:bool=True, dueling:bool=True, is_train:bool=True) -> None:
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

        self.double = double
        self.dueling = dueling

        if is_train:
            self.buffer = ReplayBuffer(state_mode, state_dim, 'disc', 1, buffer_size)

        if state_mode == 'vec':
            if not dueling:
                self.QNet = QMLP(state_dim, action_dim).to(self.device)
                if is_train:
                    self.QNet_target = QMLP(state_dim, action_dim).to(self.device)
            else:
                self.QNet = VAMLP(state_dim, action_dim).to(self.device)
                if is_train:
                    self.QNet_target = VAMLP(state_dim, action_dim).to(self.device)
        elif state_mode == 'img':
            if not dueling:
                self.QNet = QCNN(state_dim[0], action_dim).to(self.device)
                if is_train:
                    self.QNet_target = QCNN(state_dim[0], action_dim).to(self.device)
            else:
                self.QNet = VACNN(state_dim[0], action_dim).to(self.device)
                if is_train:
                    self.QNet_target = VACNN(state_dim[0], action_dim).to(self.device)
        else:
            raise ValueError(f"state_mode {state_mode} is not defined")
        if is_train:
            self.QNet_target.load_state_dict(self.QNet.state_dict())
            self.QNet_target.eval()

        if is_train:
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.QNet.parameters(), lr=lr)
    
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
            if not self.double:
                q_prime = torch.max(qs_prime, dim=1)[0]             # (B,)
            else:
                qs_prime_self = self.QNet(next_states).detach()     # (B, A)
                selected_action = torch.argmax(qs_prime_self, dim=1).unsqueeze(1)                   # (B, 1)
                q_prime = torch.gather(qs_prime, dim=1, index=selected_action).squeeze(1)           # (B,)
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
