import numpy as np
from collections import defaultdict
from typing import List, Tuple


class MonteCarloAgent:
    """Monte Carlo-based RL Agent"""
    
    def __init__(self, exploration_rate=0.3, learning_rate=0.1):
        self.q_table = defaultdict(float)
        self.returns = defaultdict(list)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.episodes_trained = 0
    
    def get_state_key(self, grid: np.ndarray, shapes: List[int]) -> str:
        """Convert state to string key for Q-table"""
        grid_str = ''.join(map(str, grid.flatten()))
        shapes_str = ','.join(map(str, sorted(shapes)))
        return f"{grid_str}|{shapes_str}"

    def get_action_key(self, state_key: str, action: Tuple) -> str:
        """Convert state-action pair to string key"""
        return f"{state_key}#{action}"
