from .BaseContextBandit import BaseContextBandit
import numpy as np
from typing import Optional

class ThompsonSampling(BaseContextBandit):
    def __init__(self, 
                 n_arms: int,
                 d_ctx: Optional[int] = None, 
                 seed: Optional[int] = None
    ) -> None:
        super().__init__(n_arms, 0, seed)
        self.reset()
    
    def reset(self):
        self.arm_sum = np.zeros(self.n_arms)
        self.arm_choices = np.zeros(self.n_arms)

        self.prior_alpha = 1.0
        self.prior_beta = 1.0

        self.alpha = np.full(self.n_arms, self.prior_alpha)
        self.beta = np.full(self.n_arms, self.prior_beta)

    def choose(self, 
               user_ctx: Optional[np.ndarray] = None, 
               arms_ctx: Optional[np.ndarray] = None) -> int:
        theta_distribution = self.rng.beta(a=self.alpha, b=self.beta)
        arm = int(np.argmax(theta_distribution))
        return arm

    def update(self, 
               reward: float, 
               arm_id: Optional[int] = None,
               user_ctx: Optional[np.ndarray] = None, 
               arm_ctx: Optional[np.ndarray] = None) -> None:
        if reward == 1:
            self.alpha[arm_id] += 1
        else:
            self.beta[arm_id] += 1
        assert (isinstance(arm_id, int))
        self.arm_sum[arm_id] = (self.arm_sum[arm_id]+reward)
        self.arm_choices[arm_id] += 1
    
        