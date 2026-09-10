import numpy as np
from typing import Optional

class ucb1_policy():
    def __init__(self, K_arms: int, c: float = np.sqrt(2), seed: Optional[int] = None):
        """
        Args:
            K_arms: количество рук
            c: гиперпараметр
        """
        self.rng = np.random.default_rng(seed=seed)

        self.c = c
        self.K_arms = K_arms

        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        self.arms_idx = list(range(self.K_arms))
    
    def set_arms_idx(self, idx):
        self.arms_idx = list(idx)

    def get_arms_idx(self):
        return self.arms_idx

    def choose(self, context: Optional[np.ndarray] = None):
        if not all(self.arm_choices):
            arm_id = int(np.argmin(self.arm_choices))
            return self.arms_idx[arm_id]

        total_steps = sum(self.arm_choices)
        avg = self.arm_sum / self.arm_choices
        ucb = self.c * np.sqrt(np.log(total_steps) / self.arm_choices)
        p = avg + ucb

        max_ucb = np.max(p)
        candidate_arms = np.where(p == max_ucb)[0]
        arm = int(self.rng.choice(candidate_arms))
        return self.arms_idx[arm]

    def update(self, arm_id: int, reward: float, context: Optional[np.ndarray] = None):
        updated_arm = self.arms_idx.index(arm_id)

        self.arm_choices[updated_arm] += 1
        self.arm_sum[updated_arm] += reward

    def reset(self):
        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)
        self.arms_idx = list(range(self.K_arms))
        
        

        