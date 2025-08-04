import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Optional
from game_environment import TenTenGame


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

    def choose_action(self, game: TenTenGame, available_shapes: List[int], training=True) -> Optional[Tuple]:
        """Choose action using epsilon-greedy policy"""
        valid_actions = []
        
        # Generate all possible actions (shape_id, position, order_index)
        for order_idx, shape_id in enumerate(available_shapes):
            positions = game.get_valid_placements(shape_id)
            for pos in positions:
                valid_actions.append((shape_id, pos[0], pos[1], order_idx))
        
        if not valid_actions:
            return None
        
        if training and random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Choose best action based on Q-values
        state_key = self.get_state_key(game.get_state(), available_shapes)
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            action_key = self.get_action_key(state_key, action)
            q_value = self.q_table[action_key]
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action if best_action else random.choice(valid_actions)
  