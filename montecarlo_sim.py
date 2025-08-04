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
    
    def train_episode(self, game) -> int:
        """Train on a single episode with improved reward calculation"""
        episode_history = []
        game.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite episodes
        
        while steps < max_steps:
            # Generate 3 random shapes
            available_shapes = random.sample(range(19), 3)
            
            if game.is_game_over(available_shapes):
                break
            
            # Play all 3 shapes with better reward tracking
            remaining_shapes = available_shapes.copy()
            round_states = []
            
            while remaining_shapes and steps < max_steps:
                prev_grid = game.get_state().copy()
                prev_score = game.score
                
                action = self.choose_action(game, remaining_shapes, training=True)
                if action is None:
                    break
                
                shape_id, row, col, _ = action
                state_key = self.get_state_key(prev_grid, remaining_shapes)
                action_key = f"{state_key}#{action}"
                
                # Execute action
                game_reward, game_over = game.place_shape(shape_id, row, col)
                
                # Calculate advanced reward
                strategic_reward = self.calculate_advanced_reward(game, prev_grid, action)
                total_reward_step = game_reward + strategic_reward
                
                total_reward += total_reward_step
                episode_history.append((action_key, total_reward_step))
                
                remaining_shapes.remove(shape_id)
                steps += 1
                
                if game_over:
                    # Heavy penalty for game over
                    episode_history.append((action_key, -100))
                    break
        
        # Update Q-values using discounted returns
        G = 0
        for i in reversed(range(len(episode_history))):
            action_key, reward = episode_history[i]
            G = reward + self.discount_factor * G
            
            if action_key not in self.returns:
                self.returns[action_key] = []
            self.returns[action_key].append(G)
            
            # Use incremental mean for better stability
            old_avg = self.q_table[action_key]
            n = len(self.returns[action_key])
            self.q_table[action_key] = old_avg + (G - old_avg) / n
        
        self.episodes_trained += 1
        self.episode_rewards.append(total_reward)
        
        # Keep only recent performance history
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
        
        return total_reward
    
    def train(self, episodes: int = 2000, verbose: bool = True):
        """Train with improved parameters and monitoring"""
        from game_environment import TenTenGame  # Import here to avoid circular import
        game = TenTenGame()
        
        best_avg_score = float('-inf')
        patience = 0
        max_patience = 200
        
        for episode in range(episodes):
            score = self.train_episode(game)
            
            # Adaptive exploration decay
            if episode > 100 and episode % 50 == 0:
                recent_avg = np.mean(self.episode_rewards[-50:])
                if recent_avg > best_avg_score:
                    best_avg_score = recent_avg
                    patience = 0
                else:
                    patience += 1
                    
                # Decay exploration rate when performance plateaus
                if patience > 10:
                    self.exploration_rate *= 0.98
                    patience = 0
            
            if verbose and episode % 100 == 0:
                recent_avg = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                print(f"Episode {episode}, Recent Avg Score: {recent_avg:.2f}, "
                      f"Exploration Rate: {self.exploration_rate:.3f}, Q-table size: {len(self.q_table)}")
        
        # Final exploration rate reduction for deployment
        self.exploration_rate = max(0.05, self.exploration_rate * 0.5)
        print(f"Training completed. Final exploration rate: {self.exploration_rate:.3f}")
        print(f"Final average score (last 100 episodes): {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def get_best_sequence(self, game, available_shapes: List[int]) -> List[Tuple]:
        """Get the best sequence with look-ahead evaluation"""
        from game_environment import TenTenGame  # Import here to avoid circular import
        
        # Try all possible orders of the 3 shapes
        import itertools
        best_sequence = []
        best_total_reward = float('-inf')
        
        for shape_order in itertools.permutations(available_shapes):
            temp_game = TenTenGame()
            temp_game.grid = game.grid.copy()
            temp_game.score = game.score
            
            sequence = []
            total_reward = 0
            
            for shape_id in shape_order:
                prev_grid = temp_game.get_state().copy()
                action = self.choose_action(temp_game, [shape_id], training=False)
                
                if action is None:
                    total_reward = float('-inf')  # Invalid sequence
                    break
                
                shape_id_action, row, col, _ = action
                sequence.append((shape_id_action, row, col))
                
                game_reward, game_over = temp_game.place_shape(shape_id_action, row, col)
                strategic_reward = self.calculate_advanced_reward(temp_game, prev_grid, action)
                total_reward += game_reward + strategic_reward
                
                if game_over:
                    total_reward -= 200  # Heavy penalty
                    break
            
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_sequence = sequence
        
        return best_sequence
    
    def save_model(self, filename: str):
        """Save trained model with additional metadata"""
        model_data = {
            'q_table': dict(self.q_table),
            'exploration_rate': self.exploration_rate,
            'episodes_trained': self.episodes_trained,
            'episode_rewards': self.episode_rewards[-100:],  # Save recent performance
            'discount_factor': self.discount_factor
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filename: str):
        """Load trained model"""
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        self.q_table = defaultdict(float, model_data['q_table'])
        self.exploration_rate = model_data.get('exploration_rate', 0.1)
        self.episodes_trained = model_data.get('episodes_trained', 0)
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.discount_factor = model_data.get('discount_factor', 0.95)
