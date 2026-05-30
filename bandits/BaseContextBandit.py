from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseContextBandit(ABC):
    def __init__(self,
                 n_arms: int,
                 d_ctx: int,
                 seed: Optional[int] = None
    ) -> None:
        """
        Initialize algorithm

        Args:
            n_arms (int): number of arms
            d_ctx (int): context dimension
            seed (int|None):
        """
        self.n_arms = n_arms
        self.d_ctx = d_ctx
        self.seed = seed
        self.t = 0
        self.rng = np.random.default_rng(seed=seed)
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets algorithm before new independent run.
        Resets internal statistic but d_ctx, n_arms
        """
        ...
    
    @abstractmethod
    def choose(self, 
               user_ctx: Optional[np.ndarray] = None, 
               arms_ctx: Optional[np.ndarray] = None) -> int:
        """
        Chooses arm dependent on user and arms contexts at turn t

        Args:
            user_ctx (np.ndarray): user context shape=(d_u, )
            arms_ctx (np.ndarray): arms contexts shape(K, d_a)
    
        Returns:
            int: arm_id
        """
        ...
    
    @abstractmethod
    def update(self, 
               reward: float, 
               arm_id: Optional[int] = None,
               user_ctx: Optional[np.ndarray] = None, 
               arm_ctx: Optional[np.ndarray] = None) -> None:
        """
        Updates model after getting reward
        """
        ...