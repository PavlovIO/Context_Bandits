import numpy as np
from typing import Optional, Tuple

class SyntheticEnvironment:  
    """
    Синтетическая среда для экспериментов, реализующая получение контекста, награды и оптимального действия.
    Имеет три сценария влияния контекста на награду:
        strong - награда сильно зависит только от контекста
        weak - награда слабо зависит от контекста, так же зависит от базовой средней награды руки
        noise - награда не зависит от контекста 
    """ 
    def __init__(self, K_arms: int, d: int, scenario: str = 'strong', 
                 seed: Optional[int] = None):
        """
        Args:
            K_arms: количество действий (рук)
            d: размерность контекста
            scenario: сценарий ('strong', 'weak', 'noise')
        """
        self.K_arms = K_arms
        self.d = d
        self.scenario = scenario
        self.rng = np.random.default_rng(seed=seed)
        
        if scenario == 'strong':
            self.base_rewards = np.zeros(K_arms) 
            self.thetas = self.rng.standard_normal((K_arms, d)) * 2.0
            
        elif scenario == 'weak':
            self.base_rewards = self.rng.uniform(0.4, 0.9, K_arms)
            self.thetas = self.rng.standard_normal((K_arms, d)) * 0.1
            
        elif scenario == 'noise':
            self.base_rewards = self.rng.uniform(0.4, 0.9, K_arms)
            self.thetas = np.zeros((K_arms, d))
            
        else:
            raise ValueError(f"Unknown scenario: {scenario}. Use 'strong', 'weak', or 'noise'.")
    
    def get_context(self) -> np.ndarray:
        return self.rng.standard_normal(self.d)
    
    def get_reward_and_optimal(self, arm_id: int, context: np.ndarray) -> Tuple[float, int, float]:
        expected_rewards = self._compute_expected_rewards(context)

        best_arm_id = int(np.argmax(expected_rewards))
        best_reward = expected_rewards[best_arm_id]

        actual_reward = float(self.rng.random() < expected_rewards[arm_id])
        
        return actual_reward, best_arm_id, best_reward
    
    def _compute_expected_rewards(self, context: np.ndarray) -> np.ndarray:
        linear_scores = np.dot(self.thetas, context)
        scores = linear_scores + self.base_rewards
        expected_rewards = self._sigmoid(scores)
        
        return expected_rewards
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def get_true_parameters(self) -> dict:
        return {
            'thetas': self.thetas,
            'base_rewards': self.base_rewards,
            'scenario': self.scenario
        }
   