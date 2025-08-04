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
    
    def train_episode(self, game: TenTenGame) -> int:
        """Train on a single episode using Monte Carlo method"""
        episode_history = []
        game.reset()
        total_reward = 0
        
        while True:
            # Generate 3 random shapes
            available_shapes = random.sample(range(19), 3)
            
            if game.is_game_over(available_shapes):
                break
            
            # Play all 3 shapes
            remaining_shapes = available_shapes.copy()
            
            while remaining_shapes:
                state = game.get_state()
                action = self.choose_action(game, remaining_shapes, training=True)
                
                if action is None:
                    break
                
                shape_id, row, col, _ = action
                state_key = self.get_state_key(state, remaining_shapes)
                action_key = self.get_action_key(state_key, action)
                
                reward, game_over = game.place_shape(shape_id, row, col)
                total_reward += reward
                
                episode_history.append((action_key, reward))
                remaining_shapes.remove(shape_id)
                
                if game_over:
                    break
        
        # Update Q-values using Monte Carlo returns
        G = 0
        for i in reversed(range(len(episode_history))):
            action_key, reward = episode_history[i]
            G = reward + G  # No discount factor for simplicity
            self.returns[action_key].append(G)
            self.q_table[action_key] = np.mean(self.returns[action_key])
        
        self.episodes_trained += 1
        return total_reward
    
    def train(self, episodes: int = 1000, verbose: bool = True):
        """Train the agent for specified number of episodes"""
        game = TenTenGame()
        
        for episode in range(episodes):
            score = self.train_episode(game)
            
            if verbose and episode % 100 == 0:
                print(f"Episode {episode}, Score: {score}, Q-table size: {len(self.q_table)}")
        
        # Decay exploration rate
        self.exploration_rate *= 0.995
        print(f"Training completed. Final exploration rate: {self.exploration_rate:.3f}")
    
    def get_best_sequence(self, game: TenTenGame, available_shapes: List[int]) -> List[Tuple]:
        """Get the best sequence of moves for given shapes"""
        best_sequence = []
        temp_game = TenTenGame()
        temp_game.grid = game.grid.copy()
        temp_game.score = game.score
        
        remaining_shapes = available_shapes.copy()
        
        while remaining_shapes:
            action = self.choose_action(temp_game, remaining_shapes, training=False)
            
            if action is None:
                break
            
            shape_id, row, col, _ = action
            best_sequence.append((shape_id, row, col))
            
            temp_game.place_shape(shape_id, row, col)
            remaining_shapes.remove(shape_id)
        
        return best_sequence
    