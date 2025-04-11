from gymnasium import spaces
import numpy as np
import gym


class TemptationBridgeEnv(gym.Env):
    def __init__(self, alpha=0.2, beta=0.1, lambda_loyalty=1.0, max_steps=100):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_loyalty = lambda_loyalty
        self.max_steps = max_steps
        self.current_step = 0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.loyalty_score = 1.0
        self.total_reward_log = []
        self.loyalty_log = []
        self.betrayals_log = []

    def reset(self, seed=None, options=None):
        self.loyalty_score = 1.0
        self.current_step = 0
        self.total_reward_log = []
        self.loyalty_log = []
        self.betrayals_log = []
        return np.array([self.loyalty_score], dtype=np.float32), {}

    def step(self, action):
        done = self.current_step >= self.max_steps
        reward = 0

        if action == 0:
            base_reward = 1.0
            delta_loyalty = self.beta * (1 - self.loyalty_score)
            self.betrayals_log.append(0)
        elif action == 1:
            base_reward = 3.0
            delta_loyalty = -self.alpha
            self.betrayals_log.append(1)
        else:
            base_reward = 0
            delta_loyalty = 0

        self.loyalty_score = np.clip(self.loyalty_score + delta_loyalty, 0.0, 1.0)
        reward = base_reward + self.lambda_loyalty * self.loyalty_score

        self.total_reward_log.append(reward)
        self.loyalty_log.append(self.loyalty_score)
        self.current_step += 1

        return np.array([self.loyalty_score], dtype=np.float32), reward, done, False, {}

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Loyalty: {self.loyalty_score:.2f}")

    def close(self):
        pass
