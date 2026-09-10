from .LinTS import *
from .LinUCB import *
from .ThompsonSampling import *
from .UCB1 import *
import numpy as np
from typing import Optional, Union

class MAB():
    """
    Интерфейс для сбора статистики для многоруких бандитов
    Статистика:
        'history_rewards' - история полученных наград
        'history_actions' - история выбраных действий на шаге
        'history_regrets' - история сожалений
        'history_best_actions' - история лучших действий на шаге
        'cumulative_regret' - кумулятивная награда
        'arm_sum' - общая награда для рук
        'arm_choices' - количество выборов рук
        'time' - горизонт планирования
    """
    def __init__(self, policy: Union[lints_policy, linucb_policy, ucb1_policy, ts_policy], 
                 K_arms: int, D_ctx: int) -> None:
        """
        Args:
            policy: политика алгоритма (lints_policy, linucb_policy, ucb1_policy, ts_policy)
            K_arms: количество рук
            D_ctx: размерность контекста
        """
        self.K_arms = K_arms
        self.D_ctx = D_ctx
        self.t = 0

        self.policy = policy

        self.history_rewards: list[float] = []
        self.history_actions: list[int] = []
        self.history_best_actions: list[int] = []
        self.history_regrets: list[float] = []

    def reset(self) -> None:
        self.t = 0
        self.history_rewards.clear()
        self.history_actions.clear()
        self.history_best_actions.clear()
        self.history_regrets.clear()

        self.policy.reset()

    def choose(self, context: Optional[np.ndarray] = None) -> int:
        arm_id = self.policy.choose(context)

        return arm_id

    def update(self, arm_id: int, reward: float, best_arm_id: int,
               best_reward: float, context: Optional[np.ndarray] = None) -> None:
        self.t += 1

        self.history_rewards.append(reward)
        self.history_actions.append(arm_id)
        self.history_best_actions.append(best_arm_id)

        instant_regret = best_reward - reward
        self.history_regrets.append(instant_regret)

        self.policy.update(arm_id, reward, context)

    def get_statistics(self) -> dict:
        statistics = {
            'history_rewards': self.history_rewards,
            'history_actions': self.history_actions,
            'history_regrets': self.history_regrets,
            'history_best_actions': self.history_best_actions,
            'cumulative_regret': np.cumsum(self.history_regrets),
            'arm_sum': self.policy.arm_sum,
            'arm_choices': self.policy.arm_choices,
            'time': self.t
        }
        return statistics
        
    def get_cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.history_regrets)
    
    def get_average_reward(self, window: Optional[int] = None) -> np.ndarray:
        rewards = np.array(self.history_rewards)
        if window is None:
            return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        else:
            cumsum = np.cumsum(rewards)
            cumsum[window:] = cumsum[window:] - cumsum[:-window]
            return cumsum[window - 1:] / window
    