import numpy as np
from typing import Optional

class lints_disjoint_arm():
    """
    Класс руки для алгоритма lints. Реализует семплирование theta и хранение и обновление структур
    """
    def __init__(self, arm_index: int, d: int, v: float):

        self.arm_index = arm_index
        self.v = v

        self.d = d
        
        self.B = np.identity(d)
        self.f = np.zeros((d, 1))

        self.B_inv = np.identity(d)
        self.theta_hat = np.zeros((d, 1))

        self.recompute_inv = 100
        self.counter = 0
        
    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        self.theta_hat = np.dot(self.B_inv, self.f)

        cov = (self.v ** 2) * self.B_inv
        
        theta_sample = rng.multivariate_normal(
            mean=self.theta_hat.flatten(), 
            cov=cov
        )
        
        return theta_sample
    
    def arm_update(self, reward: float, context: np.ndarray):
        x = context.reshape(-1, 1)
        
        self.B += np.dot(x, x.T)
        self.f += reward * x
        #Обновление обратной матрицы по формуле Шермана-Моррисона, а также точный пересчет обратной матрицы для числовой стабильности
        self.counter += 1 
        if self.counter % self.recompute_inv == 0:
            self.B_inv = np.linalg.inv(self.B)
        else:
            B_inv_X = np.dot(self.B_inv, x)
            denominator = 1.0 + float(np.dot(x.T, B_inv_X).item())
            self.B_inv -= np.dot(B_inv_X, B_inv_X.T) / denominator

    def reset_arm(self):
        self.B = np.identity(self.d)
        self.f = np.zeros((self.d, 1))
        self.B_inv = np.identity(self.d)
        self.theta_hat = np.zeros((self.d, 1))
        self.counter = 0

class lints_policy():
    def __init__(self, K_arms: int, d: int, v: float, seed: Optional[int] = None):
        """
        Args:
            K_arms: количество рук
            d: размерность контекста
            v: гиперпараметр
        """
        self.K_arms = K_arms
        self.v = v
        self.rng = np.random.default_rng(seed=seed)

        self.lints_arms = [lints_disjoint_arm(arm_index=i, d=d, v=v) for i in range(K_arms)]

        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        self.arms_idx = list(range(K_arms))

    def set_arms_idx(self, idx):
        for i in range(self.K_arms):
            self.lints_arms[i].arm_index = idx[i]
        self.arms_idx = list(idx)
    
    def get_arms_idx(self):
        return self.arms_idx
    
    def choose(self, context: np.ndarray) -> int:

        x = context.reshape(-1, 1)
        
        highest_score = -np.inf
        candidate_arms = []

        arms_idx = self.get_arms_idx()
        
        for i, arm_index in enumerate(arms_idx):
            # Сэмплируем theta для текущей руки
            theta_sample = self.lints_arms[i].sample_theta(self.rng)
            
            # Вычисляем score = theta^T * x
            score = float(np.dot(theta_sample.T, x).item())
            
            # Выбираем руку с максимальным score
            if score > highest_score:
                highest_score = score
                candidate_arms = [arm_index]
            elif score == highest_score:
                candidate_arms.append(arm_index)

        chosen_arm = int(self.rng.choice(candidate_arms))
        return chosen_arm

    def update(self, arm_id: int, reward: float, context: np.ndarray):
        arms_idx = self.get_arms_idx()
        updated_arm = arms_idx.index(arm_id)

        self.lints_arms[updated_arm].arm_update(reward, context)

        self.arm_choices[updated_arm] += 1
        self.arm_sum[updated_arm] += reward

    def reset(self):
        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        for arm in self.lints_arms:
            arm.reset_arm()
    