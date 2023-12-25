from typing import Tuple, Union, Dict, Optional
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .agent import Agent
from utils.data_utils import ReplayBuffer


class PiMLP(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, max_action:np.ndarray, min_action:np.ndarray):
        super(PiMLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        action_center = 0.5 * (max_action + min_action)
        action_scale = max_action - action_center
        self.register_buffer('action_center', torch.from_numpy(action_center).unsqueeze(0))
        self.register_buffer('action_scale', torch.from_numpy(action_scale).unsqueeze(0))

    def forward(self, s):
        return self.action_scale * self.layers(s) + self.action_center


class QMLP(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        super(QMLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = self.layer1(s)
        x = self.layer2(torch.cat([x, a], 1))
        return x[..., 0]


class DDPG(Agent):
    def __init__(self, state_mode:str, state_dim:Union[int, Tuple[int]], action_dim:int, 
                 buffer_size:int, tau:float, noise_max:float, noise_min:float, noise_frac:float, 
                 learn_start:int, learn_freq:int, max_action:np.ndarray, min_action:np.ndarray, 
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
        self.learn_start = learn_start
        self.learn_freq = learn_freq
        self.learn_counter = 0

        self.max_action = max_action
        self.min_action = min_action

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
                self.critic = QMLP(state_dim, action_dim).to(self.device)
                self.critic_target = QMLP(state_dim, action_dim).to(self.device)
        elif state_mode == 'img':
            raise NotImplementedError
        else:
            raise ValueError(f"state_mode {state_mode} is not defined")
        if is_train:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.eval()

        if is_train:
            self.critic_criterion = nn.MSELoss()
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def take_explore_action(self, state:np.ndarray) -> np.ndarray:
        # noisy
        action = self.take_optimal_action(state)
        action = action + self.noise * (self.max_action - 0.5 * (self.max_action + self.min_action)) * np.random.randn(self.action_dim)
        action = np.clip(action, self.min_action, self.max_action)
        return action
    
    def take_optimal_action(self, state:np.ndarray) -> np.ndarray:
        # deterministic
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = self.actor(state)[0]
        action = action.cpu().detach().numpy()
        return action

    def update_transition(self, state:np.ndarray, action:np.ndarray, reward:float, state_prime:np.ndarray, done:bool) -> None:
        self._update_buffer(state, action, reward, state_prime, done)

    def _update_buffer(self, state:np.ndarray, action:np.ndarray, reward:float, state_prime:np.ndarray, done:bool) -> None:
        self.buffer.add(state.astype(np.float32), action.astype(np.float32), reward, state_prime.astype(np.float32), done)

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
        q = self.critic(states, actions)                            # (B,)
        with torch.no_grad():
            a_prime = self.actor_target(next_states)                # (B, A)
            q_prime = self.critic_target(next_states, a_prime).detach()     # (B,)
            q_target = rewards + self.gamma * q_prime * (1 - dones) # (B,)
        critic_loss = self.critic_criterion(q, q_target)            # q_target semi-gradient and target delay
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

        # calculate actor loss
        start_time = time.time()
        a = self.actor(states)                                      # (B, A)
        q_a = self.critic(states, a)                                # (B,)
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
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}
    
    def update_iteration(self, iteration:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        if writer is not None:
            writer.add_scalar('noise', self.noise, iteration)
            if state_sample is not None and iteration % (self.iteration_num//100) == 0:
                state = torch.from_numpy(state_sample).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    a = self.actor(state)
                    q = self.critic(state, a)
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
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _update_noise(self) -> None:
        fraction = min(1.0, self.noise_counter / self.noise_iteration_num)
        self.noise = self.noise_max + fraction * (self.noise_min - self.noise_max)
        self.noise_counter += 1

    def update_episode(self, episode:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        pass

    def set_mode(self, mode:str) -> None:
        if mode == 'train':
            self.actor.train()
            self.critic.train()
        elif mode == 'eval':
            self.actor.eval()
            if self.is_train:
                self.critic.eval()
        else:
            raise ValueError(f"mode {mode} is not defined")

    def save(self, path:str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
    
    def load(self, path:str) -> None:
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=self.device))
        if self.is_train:
            self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth'), map_location=self.device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_target.eval()
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_target.eval()
