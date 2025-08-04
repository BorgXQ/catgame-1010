import numpy as np
from typing import List, Tuple


class TenTenGame:
    """Game Environment"""

    def __init__(self):
        self.grid_size = 10
        self.reset()
        self.shapes = self._define_shapes()
    
    def reset(self):
        """Reset the game to initial state"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        return self.get_state()
    
    def _define_shapes(self):
        """Define all 19 unique shapes"""
        shapes = {
            0: [(0, 0)],  # 1-block
            1: [(0, 0), (0, 1)],  # 2-block --
            2: [(0, 0), (1, 0)],  # 2-block |
            3: [(0, 0), (0, 1), (0, 2)],  # 3-block --
            4: [(0, 0), (1, 0), (2, 0)],  # 3-block |
            5: [(0, 0), (0, 1), (0, 2), (0, 3)],  # 4-block --
            6: [(0, 0), (1, 0), (2, 0), (3, 0)],  # 4-block |
            7: [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],  # 5-block --
            8: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],  # 5-block |
            9: [(0, 0), (0, 1), (1, 0), (1, 1)],  # 2x2 square
            10: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],  # 3x3 square
            11: [(0, 0), (1, 0), (1, 1)],  # L-shape 1
            12: [(0, 1), (1, 0), (1, 1)],  # L-shape 2
            13: [(0, 0), (0, 1), (1, 1)],  # L-shape 3
            14: [(0, 0), (0, 1), (1, 0)],  # L-shape 4
            15: [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],  # L-shape 5
            16: [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)],  # L-shape 6
            17: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],  # L-shape 7
            18: [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]   # L-shape 8
        }
        return shapes
    
    def get_state(self):
        """Get current state representation"""
        return self.grid.copy()
    
    def get_valid_placements(self, shape_id: int) -> List[Tuple[int, int]]:
        """Get all valid positions where a shape can be placed"""
        shape = self.shapes[shape_id]
        valid_positions = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.can_place_shape(shape, row, col):
                    valid_positions.append((row, col))
        
        return valid_positions
    
    def can_place_shape(self, shape: List[Tuple[int, int]], start_row: int, start_col: int) -> bool:
        """Check if shape can be placed at given position"""
        for dr, dc in shape:
            new_row, new_col = start_row + dr, start_col + dc
            if (new_row < 0 or new_row >= self.grid_size or 
                new_col < 0 or new_col >= self.grid_size or 
                self.grid[new_row, new_col] != 0):
                return False
        return True
    
    def place_shape(self, shape_id: int, row: int, col: int) -> Tuple[int, bool]:
        """Place shape and return (reward, game_over)"""
        shape = self.shapes[shape_id]
        
        if not self.can_place_shape(shape, row, col):
            return -100, True  # Invalid move penalty
        
        # Place shape
        for dr, dc in shape:
            self.grid[row + dr, col + dc] = 1
        
        # Clear full lines and calculate reward
        reward = self._clear_lines()
        
        # Check if game is over (no valid moves for any remaining shapes)
        game_over = False
        
        return reward, game_over
    
    def _clear_lines(self) -> int:
        """Clear full rows and columns, return reward"""
        lines_cleared = 0
        blocks_cleared = 0
        
        # Check rows
        rows_to_clear = []
        for i in range(self.grid_size):
            if np.all(self.grid[i, :] == 1):
                rows_to_clear.append(i)
        
        # Check columns
        cols_to_clear = []
        for j in range(self.grid_size):
            if np.all(self.grid[:, j] == 1):
                cols_to_clear.append(j)
        
        # Clear rows
        for row in rows_to_clear:
            blocks_cleared += np.sum(self.grid[row, :])
            self.grid[row, :] = 0
            lines_cleared += 1
        
        # Clear columns
        for col in cols_to_clear:
            blocks_cleared += np.sum(self.grid[:, col])
            self.grid[:, col] = 0
            lines_cleared += 1
        
        reward = blocks_cleared + lines_cleared * 10  # Bonus for clearing lines
        self.score += reward
        return reward
    
    def is_game_over(self, available_shapes: List[int]) -> bool:
        """Check if any of the available shapes can be placed"""
        for shape_id in available_shapes:
            if len(self.get_valid_placements(shape_id)) > 0:
                return False
        return True
