from .BaseContextBandit import BaseContextBandit
import numpy as np
from typing import Optional


class LinUCBShared(BaseContextBandit):
    def __init__(self,
                  n_arms: int,
                  d_ctx: int,
                  seed: Optional[int] = None,
                  alpha: float = 1.0
    ) -> None:
        super().__init__(n_arms, d_ctx, seed)
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.A = np.eye(self.d_ctx, dtype=np.float64)
        self.b = np.zeros(self.d_ctx, dtype=np.float64)

    def choose(self, 
               user_ctx: Optional[np.ndarray] = None, 
               arms_ctx: Optional[np.ndarray] = None) -> int:
        assert (isinstance(user_ctx, np.ndarray))
        assert (isinstance(arms_ctx, np.ndarray))
        theta = np.linalg.solve(self.A, self.b)
        
        user_tile = np.tile(user_ctx, (self.n_arms, 1))

        x = np.hstack([user_tile, arms_ctx])

        A_inv_x = np.linalg.solve(self.A, x.T).T
        p = x @ theta
        ucb_bonus = self.alpha * np.sqrt(np.sum(A_inv_x * x, axis=1))
        a_t = int(np.argmax(p + ucb_bonus))

        return a_t
    
    def update(self, 
               reward: float, 
               arm_id: Optional[int] = None,
               user_ctx: Optional[np.ndarray] = None, 
               arm_ctx: Optional[np.ndarray] = None) -> None:
        assert (isinstance(user_ctx, np.ndarray))
        assert (isinstance(arm_ctx, np.ndarray))
        x = np.concat([user_ctx, arm_ctx])
        self.A += np.outer(x, x)
        self.b += reward * x 
        self.t += 1


