from typing import Tuple, Union, Dict, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .ddpg import DDPG, PiMLP, QMLP
from utils.data_utils import ReplayBuffer


class TD3(DDPG):
    def __init__(self, state_mode:str, state_dim:Union[int, Tuple[int]], action_dim:int, 
                 buffer_size:int, tau:float, noise_max:float, noise_min:float, noise_frac:float, target_noise:float, target_noise_clip:float, 
                 learn_start:int, learn_freq:int, policy_learn_freq:int, max_action:np.ndarray, min_action:np.ndarray, 
                 gamma:float, actor_lr:float, critic_lr:float, batch_size:int, iteration_num:int, device:str, is_train:bool=True) -> None:
        self.state_mode = state_mode
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buffer_size = buffer_size
        self.tau = tau
        self.noise_max = noise_max
        self.noise_min = noise_min
        self.noise_frac = noise_frac
        self.noise = noise_max
        self.noise_iteration_num = noise_frac * iteration_num
        self.noise_counter = 0
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.learn_start = learn_start
        self.learn_freq = learn_freq
        self.learn_counter = 0
        self.policy_learn_freq = policy_learn_freq

        self.max_action = max_action
        self.min_action = min_action
        self.max_action_tensor = torch.from_numpy(max_action).float().to(device)
        self.min_action_tensor = torch.from_numpy(min_action).float().to(device)

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.iteration_num = iteration_num
        self.device = device
        self.is_train = is_train

        if is_train:
            self.buffer = ReplayBuffer(state_mode, state_dim, 'cont', action_dim, buffer_size)

        if state_mode == 'vec':
            self.actor = PiMLP(state_dim, action_dim, max_action, min_action).to(self.device)
            if is_train:
                self.actor_target = PiMLP(state_dim, action_dim, max_action, min_action).to(self.device)
                self.critic1 = QMLP(state_dim, action_dim).to(self.device)
                self.critic1_target = QMLP(state_dim, action_dim).to(self.device)
                self.critic2 = QMLP(state_dim, action_dim).to(self.device)
                self.critic2_target = QMLP(state_dim, action_dim).to(self.device)
        elif state_mode == 'img':
            raise NotImplementedError
        else:
            raise ValueError(f"state_mode {state_mode} is not defined")
        if is_train:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic1_target.eval()
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            self.critic2_target.eval()

        if is_train:
            self.critic_criterion = nn.MSELoss()
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam([*self.critic1.parameters(), *self.critic2.parameters()], lr=critic_lr)

    def learn(self, debug:bool=False) -> Optional[Dict[str, float]]:
        if not self.is_train:
            raise ValueError("agent is not in train mode")
        
        self.learn_counter += 1
        if self.learn_counter <= self.learn_start or self.learn_counter % self.learn_freq != 0:
            return None

        avg_critic_loss = 0
        for _ in range(self.policy_learn_freq):
            start_time = time.time()
            # sample batch transitions
            batch_transitions = self._sample_buffer(self.batch_size)
            if batch_transitions is None:
                return None
            
            # unpack batch transitions
            batch_states, batch_actions, batch_rewards, batch_states_prime, batch_dones = batch_transitions
            states = torch.from_numpy(batch_states).float().to(self.device)
            actions = torch.from_numpy(batch_actions).float().to(self.device)
            rewards = torch.from_numpy(batch_rewards).float().to(self.device)
            next_states = torch.from_numpy(batch_states_prime).float().to(self.device)
            dones = torch.from_numpy(batch_dones).float().to(self.device)
            end_time = time.time()
            if debug:
                print("data prepare time:", end_time - start_time)
            
            # calculate critic loss
            start_time = time.time()
            q1 = self.critic1(states, actions)                          # (B,)
            q2 = self.critic2(states, actions)                          # (B,)
            with torch.no_grad():
                noise = (self.max_action_tensor - 0.5 * (self.max_action_tensor + self.min_action_tensor)) * (torch.randn_like(actions) * self.target_noise).clamp(-self.target_noise_clip, self.target_noise_clip)
                a_prime = self.actor_target(next_states)                # (B, A)
                a_prime = (a_prime + noise).clamp(self.min_action_tensor, self.max_action_tensor)
                q1_prime = self.critic1_target(next_states, a_prime).detach()   # (B,)
                q2_prime = self.critic2_target(next_states, a_prime).detach()   # (B,)
                q_prime = torch.min(q1_prime, q2_prime)
                q_target = rewards + self.gamma * q_prime * (1 - dones) # (B,)
            critic_loss = self.critic_criterion(q1, q_target) + self.critic_criterion(q2, q_target)     # q_target semi-gradient and target delay
            end_time = time.time()
            if debug:
                print("forward critic time", end_time - start_time)
            
            # update critic
            start_time = time.time()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            end_time = time.time()
            if debug:
                print("backward critic time", end_time - start_time)
            
            avg_critic_loss += critic_loss.item()
        avg_critic_loss /= self.policy_learn_freq

        # calculate actor loss
        start_time = time.time()
        a = self.actor(states)                                      # (B, A)
        q_a = self.critic1(states, a)                               # (B,)
        actor_loss = -torch.mean(q_a)                               # deterministic policy gradient
        end_time = time.time()
        if debug:
            print("forward actor time", end_time - start_time)

        # update actor
        start_time = time.time()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        end_time = time.time()
        if debug:
            print("backward actor time", end_time - start_time)
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': avg_critic_loss}
    
    def update_iteration(self, iteration:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        if writer is not None:
            writer.add_scalar('noise', self.noise, iteration)
            if state_sample is not None and iteration % (self.iteration_num//100) == 0:
                state = torch.from_numpy(state_sample).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a = self.actor(state)
                    q = self.critic1(state, a)
                    a = a[0]
                    q = q.item()
                metric_dict = {f'sample/a_{a_idx:0>2d}': a[a_idx].item() for a_idx in range(a.shape[0])}
                metric_dict.update({'sample/q': q})
                writer.add_hparams({'sample/iteration': iteration}, 
                                   metric_dict, 
                                   run_name='run'+str(iteration).zfill(len(str(self.iteration_num))))
        
        self._sync_target()
        self._update_noise()
    
    def _sync_target(self) -> None:
        if self.learn_counter <= self.learn_start or self.learn_counter % self.learn_freq != 0:
            return None
        
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def set_mode(self, mode:str) -> None:
        if mode == 'train':
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
        elif mode == 'eval':
            self.actor.eval()
            if self.is_train:
                self.critic1.eval()
                self.critic2.eval()
        else:
            raise ValueError(f"mode {mode} is not defined")

    def save(self, path:str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(path, 'critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(path, 'critic2.pth'))
    
    def load(self, path:str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        if self.is_train:
            self.critic1.load_state_dict(torch.load(os.path.join(path, 'critic1.pth'), map_location=self.device))
            self.critic2.load_state_dict(torch.load(os.path.join(path, 'critic2.pth'), map_location=self.device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()
            self.critic1_target.load_state_dict(self.critic1.state_dict())
            self.critic1_target.eval()
            self.critic2_target.load_state_dict(self.critic2.state_dict())
            self.critic2_target.eval()
