from .BaseContextBandit import BaseContextBandit
import numpy as np
from typing import Optional

class LinTSShared(BaseContextBandit):
    def __init__(self,
                  n_arms: int,
                  d_ctx: int,
                  seed: Optional[int] = None,
                  sigma: float = 0.5
    ) -> None:
        super().__init__(n_arms, d_ctx, seed)
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        self.A = np.eye(self.d_ctx, dtype=np.float64)
        self.b = np.zeros(self.d_ctx, dtype=np.float64)

    def choose(self, 
               user_ctx: Optional[np.ndarray] = None, 
               arms_ctx: Optional[np.ndarray] = None) -> int:
        assert (isinstance(user_ctx, np.ndarray))
        assert (isinstance(arms_ctx, np.ndarray))
        user_tile = np.tile(user_ctx, (self.n_arms, 1))
        x = np.hstack([user_tile, arms_ctx])

        # семплирование используя разложение Холецкого
        mu = np.linalg.solve(self.A, self.b)
        L = np.linalg.cholesky(self.A)
        z = self.rng.standard_normal(self.d_ctx)
        theta_samples = mu + self.sigma * np.linalg.solve(L.T, z)
        expected_reward = x @ theta_samples

        a_t = int(np.argmax(expected_reward))
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