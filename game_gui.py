import tkinter as tk
from tkinter import messagebox
from game_environment import TenTenGame
from montecarlo_sim import MonteCarloAgent

class TenTenGUI:
    """Game GUI Interface"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RL Solver Interface")
        self.root.geometry("800x700")
        
        self.game = TenTenGame()
        self.agent = MonteCarloAgent()
        
        # Try to load pre-trained model
        try:
            self.agent.load_model("tenten_model.json")
            print(f"Loaded model with {self.agent.episodes_trained} episodes trained")
        except FileNotFoundError:
            print("No pre-trained model found. Training new model...")
            self.agent.train(episodes=500)
            self.agent.save_model("tenten_model.json")
        
        self.setup_gui()
        self.selected_shapes = []
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Grid frame
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=10)
        
        self.grid_buttons = []
        for i in range(10):
            row = []
            for j in range(10):
                btn = tk.Button(grid_frame, width=3, height=1, 
                               command=lambda r=i, c=j: self.toggle_cell(r, c))
                btn.grid(row=i, column=j, padx=1, pady=1)
                row.append(btn)
            self.grid_buttons.append(row)
        
        # Shape selection frame
        shape_frame = tk.Frame(self.root)
        shape_frame.pack(pady=10)
        
        tk.Label(shape_frame, text="Select 3 shapes (0-18):").pack()
        
        self.shape_vars = []
        shape_entry_frame = tk.Frame(shape_frame)
        shape_entry_frame.pack()
        
        for i in range(3):
            var = tk.StringVar()
            entry = tk.Entry(shape_entry_frame, textvariable=var, width=5)
            entry.pack(side=tk.LEFT, padx=5)
            self.shape_vars.append(var)
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Clear Grid", command=self.clear_grid).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Get Best Moves", command=self.get_best_moves).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Train More", command=self.train_more).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        self.result_text = tk.Text(self.root, height=15, width=80)
        self.result_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
        
        self.update_grid_display()
    
    def toggle_cell(self, row: int, col: int):
        """Toggle grid cell between empty and filled"""
        self.game.grid[row, col] = 1 - self.game.grid[row, col]
        self.update_grid_display()
    
    def update_grid_display(self):
        """Update the visual grid display"""
        for i in range(10):
            for j in range(10):
                if self.game.grid[i, j] == 1:
                    self.grid_buttons[i][j].config(bg='blue', text='■')
                else:
                    self.grid_buttons[i][j].config(bg='white', text='')
    
    def clear_grid(self):
        """Clear the entire grid"""
        self.game.reset()
        self.update_grid_display()
    
    def get_best_moves(self):
        """Get and display the best sequence of moves"""
        try:
            # Get selected shapes
            shapes = []
            for var in self.shape_vars:
                shape_id = int(var.get())
                if 0 <= shape_id <= 18:
                    shapes.append(shape_id)
                else:
                    raise ValueError(f"Shape ID {shape_id} is out of range (0-18)")
            
            if len(shapes) != 3:
                raise ValueError("Please enter exactly 3 shape IDs")
            
            # Get best sequence
            sequence = self.agent.get_best_sequence(self.game, shapes)
            
            # Display results
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Current Grid State:\n")
            for i in range(10):
                row_str = ''.join(['■' if self.game.grid[i, j] == 1 else '□' for j in range(10)])
                self.result_text.insert(tk.END, f"{row_str}\n")
            
            self.result_text.insert(tk.END, f"\nSelected Shapes: {shapes}\n\n")
            self.result_text.insert(tk.END, "Recommended Move Sequence:\n")
            
            for i, (shape_id, row, col) in enumerate(sequence):
                self.result_text.insert(tk.END, f"{i+1}. Place shape {shape_id} at position ({row}, {col})\n")
                
                # Show shape pattern
                shape_pattern = self.game.shapes[shape_id]
                self.result_text.insert(tk.END, f"   Shape pattern: {shape_pattern}\n")
            
            if not sequence:
                self.result_text.insert(tk.END, "No valid moves available!\n")
            
            # Simulate the moves to show expected outcome
            temp_game = TenTenGame()
            temp_game.grid = self.game.grid.copy()
            total_reward = 0
            
            self.result_text.insert(tk.END, f"\nSimulated Outcome:\n")
            for shape_id, row, col in sequence:
                reward, _ = temp_game.place_shape(shape_id, row, col)
                total_reward += reward
                self.result_text.insert(tk.END, f"After placing shape {shape_id}: +{reward} points\n")
            
            self.result_text.insert(tk.END, f"\nTotal expected reward: {total_reward} points\n")
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def train_more(self):
        """Train the model with additional episodes"""
        episodes = 200
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Training {episodes} additional episodes...\n")
        self.root.update()
        
        self.agent.train(episodes=episodes, verbose=False)
        self.agent.save_model("tenten_model.json")
        
        self.result_text.insert(tk.END, f"Training completed! Total episodes: {self.agent.episodes_trained}\n")
        self.result_text.insert(tk.END, f"Model saved. Q-table size: {len(self.agent.q_table)}\n")
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

# Example usage and training script
if __name__ == "__main__":
    # Option 1: Run the GUI interface
    print("Starting 1010! RL Solver GUI...")
    gui = TenTenGUI()
    gui.run()
    
    # Option 2: Train and test programmatically (uncomment to use)
    """
    # Create and train agent
    agent = MonteCarloAgent()
    print("Training agent...")
    agent.train(episodes=1000)
    
    # Save the trained model
    agent.save_model("tenten_model.json")
    
    # Test the agent
    game = TenTenGame()
    # Set up a test scenario
    game.grid[9, :5] = 1  # Fill bottom row partially
    test_shapes = [0, 1, 9]  # Single block, 2-block horizontal, 2x2 square
    
    print("Test scenario:")
    print("Grid state:", game.grid)
    print("Available shapes:", test_shapes)
    
    best_sequence = agent.get_best_sequence(game, test_shapes)
    print("Best move sequence:", best_sequence)
    """