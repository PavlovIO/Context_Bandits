import numpy as np
from typing import Optional

class ts_policy():
    def __init__(self, K_arms: int, seed: Optional[int] = None):
        """
        Args:
            K_arms: количество рук
        """
        self.rng = np.random.default_rng(seed=seed)

        self.K_arms = K_arms

        self.prior_alpha = 1.0
        self.prior_beta = 1.0

        self.alpha = np.full(self.K_arms, self.prior_alpha)
        self.beta = np.full(self.K_arms, self.prior_beta)

        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        self.arms_idx = list(range(self.K_arms))

    def set_arms_idx(self, idx):
        self.arms_idx = list(idx)

    def get_arms_idx(self):
        return self.arms_idx

    def choose(self, context: Optional[np.ndarray] = None):
        theta_distribution = self.rng.beta(a=self.alpha, b=self.beta)
        
        max_theta = np.max(theta_distribution)
        candidate_arms = np.where(theta_distribution == max_theta)[0]
        arm = int(self.rng.choice(candidate_arms))
        return self.arms_idx[arm]

    def update(self, arm_id: int, reward: float, context: Optional[np.ndarray] = None):
        updated_arm = self.arms_idx.index(arm_id)
        if reward == 1:
            self.alpha[updated_arm] += 1
        else:
            self.beta[updated_arm] += 1
        self.arm_sum[updated_arm] += reward
        self.arm_choices[updated_arm] += 1

    def reset(self):
        self.alpha = np.full(self.K_arms, self.prior_alpha)
        self.beta = np.full(self.K_arms, self.prior_beta)

        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

    
        