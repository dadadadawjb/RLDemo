from typing import Tuple, Union, Optional
import numpy as np


class ReplayBuffer(object):
    def __init__(self, state_mode:str, state_dim:Union[int, Tuple[int]], action_mode:str, action_dim:int, capacity:int, allow_repeat:bool=True) -> None:
        self.state_mode = state_mode
        self.action_mode = action_mode
        self.state_dim = state_dim
        self.buffer_size = capacity
        self.allow_repeat = allow_repeat

        self.buffer_pointer = 0
        if state_mode == 'vec':
            self.state_buffer = np.zeros((capacity, state_dim))
            self.state_prime_buffer = np.zeros((capacity, state_dim))
        elif state_mode == 'img':
            self.state_buffer = np.zeros((capacity, *state_dim))
            self.state_prime_buffer = np.zeros((capacity, *state_dim))
        else:
            raise ValueError(f"state_mode {state_mode} is not defined")
        if action_mode == 'disc':
            self.action_buffer = np.zeros((capacity,), dtype=int)
        elif action_mode == 'cont':
            self.action_buffer = np.zeros((capacity, action_dim))
        else:
            raise ValueError(f"action_mode {action_mode} is not defined")
        self.reward_buffer = np.zeros((capacity,), dtype=float)
        self.done_buffer = np.zeros((capacity,), dtype=bool)

    def add(self, state:np.ndarray, action:Union[int, np.ndarray], reward:float, state_prime:np.ndarray, done:bool) -> None:
        if self.buffer_pointer == 0:
            self.state_buffer = self.state_buffer.astype(state.dtype)
            self.state_prime_buffer = self.state_prime_buffer.astype(state_prime.dtype)
            if self.action_mode == 'cont':
                self.action_buffer = self.action_buffer.astype(action.dtype)
        
        if not self.allow_repeat:
            if self.state_mode == 'vec':
                state_indices = np.where((self.state_buffer == state).all(axis=-1))[0]
                state_prime_indices = np.where((self.state_prime_buffer == state_prime).all(axis=-1))[0]
            elif self.state_mode == 'img':
                state_indices = np.where((self.state_buffer == state).all(axis=tuple(range(1, 1+len(self.state_dim)))))[0]
                state_prime_indices = np.where((self.state_prime_buffer == state_prime).all(axis=tuple(range(1, 1+len(self.state_dim)))))[0]
            else:
                raise ValueError(f"state_mode {self.state_mode} is not defined")
            if self.action_mode == 'disc':
                action_indices = np.where(self.action_buffer == action)[0]
            elif self.action_mode == 'cont':
                action_indices = np.where((self.action_buffer == action).all(axis=-1))[0]
            else:
                raise ValueError(f"action_mode {self.action_mode} is not defined")
            reward_indices = np.where(self.reward_buffer == reward)[0]
            done_indices = np.where(self.done_buffer == done)[0]

            indices = np.intersect1d(state_indices, state_prime_indices)
            indices = np.intersect1d(indices, action_indices)
            indices = np.intersect1d(indices, reward_indices)
            indices = np.intersect1d(indices, done_indices)
            if indices.size > 0:
                return
        
        index = self.buffer_pointer % self.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.state_prime_buffer[index] = state_prime
        self.done_buffer[index] = done
        self.buffer_pointer += 1

    def sample(self, batch_size:int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if self.buffer_pointer <= 0:
            raise ValueError("buffer_pointer should be greater than 0 to sample")

        if self.buffer_pointer < batch_size:
            return None
        
        # sample batch transitions
        all_num = min(self.buffer_pointer, self.buffer_size)
        indices = np.random.choice(all_num, batch_size, replace=False)
        batch_states = self.state_buffer[indices]
        batch_actions = self.action_buffer[indices]
        batch_rewards = self.reward_buffer[indices]
        batch_states_prime = self.state_prime_buffer[indices]
        batch_dones = self.done_buffer[indices]
        return (batch_states, batch_actions, batch_rewards, batch_states_prime, batch_dones)
    
    def all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.buffer_pointer <= 0:
            raise ValueError("buffer_pointer should be greater than 0 to return all")
        
        current_size = self.size()
        return (self.state_buffer[:current_size], self.action_buffer[:current_size], self.reward_buffer[:current_size], self.state_prime_buffer[:current_size], self.done_buffer[:current_size])
    
    def clear(self) -> None:
        self.state_buffer[...] = 0
        self.action_buffer[...] = 0
        self.reward_buffer[...] = 0
        self.state_prime_buffer[...] = 0
        self.done_buffer[...] = 0
        self.buffer_pointer = 0

    def size(self) -> int:
        return self.buffer_pointer if self.buffer_pointer < self.buffer_size else self.buffer_size
