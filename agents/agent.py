from typing import Union, Dict, Optional
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Agent(ABC):
    def __init__(self) -> None:
        super(Agent, self).__init__()
    
    @abstractmethod
    def set_mode(self, mode:str) -> None:
        pass

    @abstractmethod
    def save(self, path:str) -> None:
        pass

    @abstractmethod
    def load(self, path:str) -> None:
        pass

    @abstractmethod
    def take_optimal_action(self, state:np.ndarray) -> Union[int, np.ndarray]:
        # ensure returned action in correct range
        pass

    @abstractmethod
    def take_explore_action(self, state:np.ndarray) -> Union[int, np.ndarray]:
        # ensure returned action in correct range
        pass

    @abstractmethod
    def update_transition(self, state:np.ndarray, action:Union[int, np.ndarray], reward:float, state_prime:np.ndarray, done:bool) -> None:
        pass

    @abstractmethod
    def update_iteration(self, iteration:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        pass

    @abstractmethod
    def update_episode(self, episode:int, writer:Optional[SummaryWriter]=None, state_sample:Optional[np.ndarray]=None) -> None:
        pass

    @abstractmethod
    def learn(self, debug:bool=False) -> Optional[Dict[str, float]]:
        pass
