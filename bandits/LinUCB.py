import numpy as np
from typing import Optional

# Create class object for a single linear ucb disjoint arm
class linucb_disjoint_arm():
    """
    Класс руки для алгоритма linucb. Реализует расчет верхней доверительной границы и хранение и обновление структур
    """
    def __init__(self, arm_index: int, d: int, alpha: float):

        self.arm_index = arm_index
        self.alpha = alpha

        self.d = d
 
        self.A = np.identity(d)
        self.b = np.zeros((d,1))

        self.A_inv = np.identity(d)
        self.theta_hat = np.zeros((d,1))

        self.recompute_inv = 100
        self.counter = 0

    def calc_UCB(self, context: np.ndarray):
        self.theta_hat = np.dot(self.A_inv, self.b)
        x = context.reshape([-1,1])

        mean = np.dot(self.theta_hat.T,x)

        ucb = np.sqrt(x.T @ self.A_inv @ x)
        p = mean +  self.alpha * ucb
        return p
    
    def arm_update(self, reward: float, context: np.ndarray):
        x = context.reshape([-1,1])
        self.A += np.dot(x, x.T)
        self.b += reward * x
        self.counter += 1

        #Обновление обратной матрицы по формуле Шермана-Моррисона, а также точный пересчет обратной матрицы для числовой стабильности
        if self.counter % self.recompute_inv == 0:
            self.A_inv = np.linalg.inv(self.A)
        else:
            A_inv_x = np.dot(self.A_inv, x)
            denominator = 1.0 + float(np.dot(x.T, A_inv_x).item())
            self.A_inv -= np.dot(A_inv_x, A_inv_x.T) / denominator

    def reset_arm(self):
        self.A = np.identity(self.d)
        self.b = np.zeros((self.d,1))
        self.A_inv = np.identity(self.d)

class linucb_policy():
    def __init__(self, K_arms: int, d: int, alpha: float, seed: Optional[int]=None):
        self.K_arms = K_arms
        self.rng = np.random.default_rng(seed=seed)
        self.linucb_arms = [linucb_disjoint_arm(arm_index = i, d = d, alpha = alpha) for i in range(K_arms)]

        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        self.arms_idx = list(range(self.K_arms))

    def set_arms_idx(self, idx):
        for i in range(self.K_arms):
            self.linucb_arms[i].arm_index = idx[i]
        self.arms_idx = list(idx)

    def get_arms_idx(self):
        return self.arms_idx

    def choose(self, context: np.ndarray):
        highest_ucb = -np.inf
        candidate_arms = []

        arms_idx = self.get_arms_idx()
        
        for i, arm_index in enumerate(arms_idx):
            
            arm_ucb = self.linucb_arms[i].calc_UCB(context)

            if arm_ucb > highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms = [arm_index]
            elif arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)
        # Tie braker
        chosen_arm = int(self.rng.choice(candidate_arms))
        return chosen_arm

    def update(self, arm_id: int, reward: float, context: np.ndarray):
        arms_idx = self.get_arms_idx()
        updated_arm = arms_idx.index(arm_id)

        self.linucb_arms[updated_arm].arm_update(reward, context)

        self.arm_choices[updated_arm] += 1
        self.arm_sum[updated_arm] += reward

    def reset(self):
        self.arm_sum = np.zeros(self.K_arms)
        self.arm_choices = np.zeros(self.K_arms)

        for arm in self.linucb_arms:
            arm.reset_arm()
