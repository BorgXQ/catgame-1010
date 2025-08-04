import numpy as np
import random
import json
from collections import defaultdict
from typing import List, Tuple, Optional
from game_environment import TenTenGame


class MonteCarloAgent:
    """Monte Carlo-based RL Agent"""
    
    def __init__(self, exploration_rate=0.3, learning_rate=0.1, discount_factor=0.95):
        self.q_table = defaultdict(float)
        self.returns = defaultdict(list)
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.episodes_trained = 0
        self.episode_rewards = []  # Track performance
    
    def get_state_key(self, grid: np.ndarray, shapes: List[int]) -> str:
        """Convert state to string key - simplified for better generalization"""
        # Use grid density and structure patterns
        row_densities = [np.sum(grid[i, :]) for i in range(10)]
        col_densities = [np.sum(grid[:, j]) for j in range(10)]
        
        # Add structural features
        holes = self._count_holes(grid)
        edges = self._count_edge_blocks(grid)
        islands = self._count_isolated_regions(grid)
        
        state_features = (
            tuple(row_densities),
            tuple(col_densities), 
            holes,
            edges,
            islands,
            tuple(sorted(shapes))
        )
        
        return str(hash(state_features))
    
    def _count_holes(self, grid: np.ndarray) -> int:
        """Count isolated empty spaces that are hard to fill"""
        holes = 0
        for i in range(1, 9):  # Skip edges
            for j in range(1, 9):
                if grid[i, j] == 0:
                    # Check if surrounded by filled cells
                    neighbors = [
                        grid[i-1, j], grid[i+1, j], 
                        grid[i, j-1], grid[i, j+1]
                    ]
                    if sum(neighbors) >= 3:  # Mostly surrounded
                        holes += 1
        return holes
    
    def _count_edge_blocks(self, grid: np.ndarray) -> int:
        """Count blocks on edges (generally good for line clearing)"""
        edges = np.sum(grid[0, :]) + np.sum(grid[9, :])  # Top and bottom
        edges += np.sum(grid[:, 0]) + np.sum(grid[:, 9])  # Left and right
        return edges
    
    def _count_isolated_regions(self, grid: np.ndarray) -> int:
        """Count separate empty regions"""
        visited = np.zeros_like(grid, dtype=bool)
        regions = 0
        
        def dfs(i, j):
            if i < 0 or i >= 10 or j < 0 or j >= 10 or visited[i, j] or grid[i, j] == 1:
                return
            visited[i, j] = True
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)
        
        for i in range(10):
            for j in range(10):
                if grid[i, j] == 0 and not visited[i, j]:
                    dfs(i, j)
                    regions += 1
        
        return regions

    def calculate_reward(self, game, prev_grid: np.ndarray, action: Tuple) -> float:
        """Calculate reward"""
        shape_id, row, col = action[:3]
        current_grid = game.get_state()
        
        # Base reward from line clearing
        lines_cleared_reward = game.score - np.sum(prev_grid)  # Score increase
        
        # Penalize creating holes and isolated spaces
        prev_holes = self._count_holes(prev_grid)
        curr_holes = self._count_holes(current_grid)
        hole_penalty = (curr_holes - prev_holes) * -10
        
        # Reward edge placement (helps with line clearing)
        edge_bonus = 0
        shape_coords = game.shapes[shape_id]
        for dr, dc in shape_coords:
            pos_r, pos_c = row + dr, col + dc
            if pos_r == 0 or pos_r == 9 or pos_c == 0 or pos_c == 9:
                edge_bonus += 2
        
        # Reward completing rows/columns or getting close
        completion_bonus = 0
        for i in range(10):
            row_filled = np.sum(current_grid[i, :])
            col_filled = np.sum(current_grid[:, i])
            
            # Big bonus for completing lines
            if row_filled == 10:
                completion_bonus += 50
            elif row_filled >= 8:  # Close to completion
                completion_bonus += row_filled * 2
                
            if col_filled == 10:
                completion_bonus += 50
            elif col_filled >= 8:
                completion_bonus += col_filled * 2
        
        # Penalize fragmentation
        prev_regions = self._count_isolated_regions(prev_grid)
        curr_regions = self._count_isolated_regions(current_grid)
        fragmentation_penalty = (curr_regions - prev_regions) * -5
        
        # Height penalty - prefer keeping pieces low
        height_penalty = 0
        for dr, dc in shape_coords:
            pos_r = row + dr
            height_penalty += (9 - pos_r) * -0.5  # Slight penalty for height
        
        # Density bonus - reward placing pieces in denser areas
        density_bonus = 0
        for dr, dc in shape_coords:
            pos_r, pos_c = row + dr, col + dc
            # Count neighbors
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = pos_r + di, pos_c + dj
                    if 0 <= ni < 10 and 0 <= nj < 10 and prev_grid[ni, nj] == 1:
                        neighbors += 1
            density_bonus += neighbors * 0.5
        
        total_reward = (
            lines_cleared_reward * 3 +  # Primary objective
            completion_bonus +
            edge_bonus +
            hole_penalty +
            fragmentation_penalty +
            height_penalty +
            density_bonus
        )
        
        return total_reward

    def choose_action(self, game, available_shapes: List[int], training=True) -> Optional[Tuple]:
        """Action selection"""
        valid_actions = []
        
        # Generate all possible actions
        for order_idx, shape_id in enumerate(available_shapes):
            positions = game.get_valid_placements(shape_id)
            for pos in positions:
                valid_actions.append((shape_id, pos[0], pos[1], order_idx))
        
        if not valid_actions:
            return None
        
        state_key = self.get_state_key(game.get_state(), available_shapes)
        
        if training:
            # Epsilon-greedy with decay
            if random.random() < self.exploration_rate:
                return random.choice(valid_actions)
            
            # Boltzmann exploration
            if len(valid_actions) > 1:
                temperatures = []
                for action in valid_actions:
                    action_key = f"{state_key}#{action}"
                    q_value = self.q_table[action_key]
                    temperatures.append(q_value)
                
                # Convert to probabilities
                if max(temperatures) > min(temperatures):
                    temperatures = np.array(temperatures)
                    temperatures = np.exp(temperatures / 0.1)  # Temperature parameter
                    probabilities = temperatures / np.sum(temperatures)
                    return np.random.choice(valid_actions, p=probabilities)
        
        # Choose best action
        best_action = None
        best_value = float('-inf')
        
        for action in valid_actions:
            action_key = f"{state_key}#{action}"
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
    
    def save_model(self, filename: str):
        """Save trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'exploration_rate': self.exploration_rate,
            'episodes_trained': self.episodes_trained
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filename: str):
        """Load trained model"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.q_table = defaultdict(float, model_data['q_table'])
        self.exploration_rate = model_data['exploration_rate']
        self.episodes_trained = model_data['episodes_trained']
