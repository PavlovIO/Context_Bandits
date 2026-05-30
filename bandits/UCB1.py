from .BaseContextBandit import BaseContextBandit
import numpy as np
from typing import Optional

class UCB1(BaseContextBandit):
    def __init__(self,
                  n_arms: int,
                  d_ctx: Optional[int] = None, 
                  seed: Optional[int] = None,
                  c: int = np.sqrt(2)
    ) -> None:
        super().__init__(n_arms, 0, seed)
        self.c = c
        self.reset()

    def reset(self) -> None:
        self.initialized = False
        self.arm_sum = np.zeros(self.n_arms)
        self.arm_choices = np.zeros(self.n_arms)

    def choose(self, 
               user_ctx: Optional[np.ndarray] = None, 
               arms_ctx: Optional[np.ndarray] = None) -> int:
        if not self.initialized:
            arm = self.t % self.n_arms
            if self.t+1 >= self.n_arms:
                self.initialized = True
            return arm

        avg = self.arm_sum / self.arm_choices
        bonus = self.c * np.sqrt(np.log(self.t + 1) / self.arm_choices)
        ucb = avg + bonus
        return int(np.argmax(ucb))
    
    def update(self, 
               reward: float, 
               arm_id: Optional[int] = None,
               user_ctx: Optional[np.ndarray] = None, 
               arm_ctx: Optional[np.ndarray] = None) -> None:
        assert (isinstance(arm_id, int))
        self.arm_sum[arm_id] = (self.arm_sum[arm_id]+reward)
        self.arm_choices[arm_id] += 1
        
        

        