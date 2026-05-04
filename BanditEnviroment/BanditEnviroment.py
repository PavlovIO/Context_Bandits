import numpy as np

class BanditEnviroment:
    """
    Class for modeling Context Bandit Enviroment
    
    Attributes:
        arm_ctx (np.ndarray): arms context
        user_ctx (np.ndarray): users context
        theta (np.ndarray): true values
        n_arms (int): number of arms
        reward_noise_std (np.ndarray):
        rng (numpy.random.Generator):
    """
    def __init__(self,
                 arm_context: np.ndarray,             # shape (n_arms, d_a) 
                 user_context: np.ndarray,            # shape (n_users, d_u) 
                 true_weights: np.ndarray,            # shape (n_users, d_a+d_u)
                 reward_noise_std: float = 0.0,
                 seed: int | None = None
    ) -> None:
        """
        Initialize enviroment

        Args:
            arm_context (np.ndarray): array (n_arms, d_a) of arms contexts
            user_context (np.ndarray): array (n_users, d_u) of users contexts
            true_weights (np.ndarray): array (d_u + d_a) of weights for true reward
            reward_noise_std (float): 
            seed (int): 
        """
        self.rng = np.random.default_rng(seed=seed)
        self.arm_ctx = arm_context
        self.user_ctx = user_context
        self.theta = true_weights
        self.reward_noise_std = reward_noise_std

        self.n_arms = self.arm_ctx.shape[0]

    def reset(self, user_id: int) -> np.ndarray:
        """Emulates new user and returns his context"""
        if (0 <= user_id <= self.user_ctx.shape[0]):
            raise ValueError(f"user_id {user_id} is out of bounds: [0, {self.user_ctx.shape[0]-1}]")
        self.current_user = user_id
        self.current_user_ctx = self.user_ctx[user_id].copy()
        return self.current_user_ctx

    def step(self, arm_id: int) -> float:
        """Generates stochastic reward for chosen arm"""
        if not hasattr(self, 'current_user_ctx'):
            raise RuntimeError("Use reset(user_id) before step()")
        if (0 <= arm_id <= self.arm_ctx.shape[0]):
            raise ValueError(f"arm_id {arm_id} is out of bounds: [0, {self.arm_ctx.shape[0]-1}]")
        x = np.concat([self.current_user_ctx, self.arm_ctx[arm_id]])
        logits = self.theta @ x
        p = 1 / (1 + np.exp(-logits))
        if self.reward_noise_std != 0.0:
            p = np.clip(p + self.rng.normal(0, self.reward_noise_std), 0.0, 1.0)
        return float(self.rng.binomial(1, p))
    
    def get_expected_reward(self, arm_id: int) -> float:
        "Returns expected reward for arm without noise"
        if not hasattr(self, 'current_user_ctx'):
            raise RuntimeError("Use reset(user_id) before get_expected_reward()")
        if (0 <= arm_id <= self.arm_ctx.shape[0]):
            raise ValueError(f"arm_id {arm_id} is out of bounds: [0, {self.arm_ctx.shape[0]-1}]")
        x = np.concat([self.current_user_ctx, self.arm_ctx[arm_id]])
        logits = self.theta @ x
        p = 1.0 / (1.0 + np.exp(-logits))
        return float(self.rng.binomial(1, p))

    def get_optimal_arm(self) -> int:
        """Returns arm_id with max reward"""
        if not hasattr(self, 'current_user_ctx'):
            raise RuntimeError("Use reset(user_id) before get_optimal_arm()")
        
        user_vec = np.repeat(self.current_user_ctx[np.newaxis, :], self.n_arms ,axis=0)
        join_ctx = np.hstack([user_vec, self.arm_ctx])
        logits = self.theta @ join_ctx
        expected = 1.0 / (1.0 + np.exp(- logits))
        return int(np.argmax(expected))

    def get_user_context(self, user_id: int) -> np.ndarray:
        return self.user_ctx[user_id]

    def get_arm_context(self, arm_id: int) -> np.ndarray:
        return self.arm_ctx[arm_id]

    def get_arm_contexts(self) -> np.ndarray:
        return self.arm_ctx
    
    def get_user_contextx(self) -> np.ndarray:
        return self.user_ctx


    