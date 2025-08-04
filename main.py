import os
import sys
from game_environment import TenTenGame
from montecarlo_sim import MonteCarloAgent
from game_gui import TenTenGUI

def train_new_model(episodes=3000):
    """Train a new model from scratch"""
    print(f"Training new model for {episodes} episodes...")
    print("This may take a few minutes...")
    
    agent = MonteCarloAgent(
        exploration_rate=0.8,  # Start with high exploration
        learning_rate=0.1,
        discount_factor=0.95
    )
    
    # Train the agent
    agent.train(episodes=episodes, verbose=True)
    
    # Save the model
    agent.save_model("tenten_model.json")
    print(f"Model saved as 'tenten_model.json'")
    
    return agent

def test_model_performance(agent, num_tests=10):
    """Test the trained model performance"""
    print(f"\nTesting model performance over {num_tests} games...")
    
    game = TenTenGame()
    total_scores = []
    
    for test in range(num_tests):
        game.reset()
        game_score = 0
        moves = 0
        
        while moves < 50:  # Limit moves to prevent infinite games
            # Generate random shapes
            import random
            available_shapes = random.sample(range(19), 3)
            
            if game.is_game_over(available_shapes):
                break
                
            # Get best sequence
            sequence = agent.get_best_sequence(game, available_shapes)
            
            if not sequence:
                break
                
            # Execute the sequence
            for shape_id, row, col in sequence:
                prev_score = game.score
                reward, game_over = game.place_shape(shape_id, row, col)
                game_score = game.score
                moves += 1
                
                if game_over:
                    break
            
            if game.is_game_over(available_shapes):
                break
        
        total_scores.append(game_score)
        print(f"Test game {test + 1}: Score = {game_score}, Moves = {moves}")
    
    avg_score = sum(total_scores) / len(total_scores)
    max_score = max(total_scores)
    print(f"\nAverage Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Q-table size: {len(agent.q_table)}")
    
    return avg_score

def load_or_train_model():
    """Load existing model or train a new one"""
    model_file = "tenten_model.json"
    
    if os.path.exists(model_file):
        print(f"Loading existing model: {model_file}")
        agent = MonteCarloAgent()
        try:
            agent.load_model(model_file)
            print(f"Model loaded successfully! Episodes trained: {agent.episodes_trained}")
            
            # Test if model seems reasonable
            if len(agent.q_table) < 100:
                print("Model seems undertrained, training more...")
                agent.train(episodes=2000, verbose=True)
                agent.save_model(model_file)
            
            return agent
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            return train_new_model()
    else:
        print("No existing model found. Training new model...")
        return train_new_model()

def demonstrate_strategy():
    """Demonstrate the AI strategy on a sample game state"""
    print("\n" + "="*50)
    print("STRATEGY DEMONSTRATION")
    print("="*50)
    
    game = TenTenGame()
    agent = load_or_train_model()
    
    # Create a test scenario
    print("Setting up test scenario...")
    
    # Fill some rows partially to create line-clearing opportunities
    game.grid[8, :7] = 1  # Row 8 almost full
    game.grid[9, :5] = 1  # Row 9 half full
    game.grid[:6, 0] = 1  # Column 0 mostly full
    
    print("Current grid state:")
    for i in range(10):
        row_str = ''.join(['■' if game.grid[i, j] == 1 else '□' for j in range(10)])
        print(f"{i}: {row_str}")
    
    # Test with specific shapes that can create strategies
    test_shapes = [3, 9, 1]  # 3-block horizontal, 2x2 square, 2-block horizontal
    print(f"\nAvailable shapes: {test_shapes}")
    print("Shape 3: 3-block horizontal line")
    print("Shape 9: 2x2 square") 
    print("Shape 1: 2-block horizontal")
    
    # Get AI recommendation
    sequence = agent.get_best_sequence(game, test_shapes)
    
    print(f"\nAI Recommended sequence:")
    if sequence:
        for i, (shape_id, row, col) in enumerate(sequence):
            print(f"{i+1}. Place shape {shape_id} at position ({row}, {col})")
    else:
        print("No valid moves found!")
    
    # Simulate the moves
    if sequence:
        temp_game = TenTenGame()
        temp_game.grid = game.grid.copy()
        temp_game.score = game.score
        
        print(f"\nSimulating moves:")
        total_reward = 0
        for i, (shape_id, row, col) in enumerate(sequence):
            prev_score = temp_game.score
            reward, game_over = temp_game.place_shape(shape_id, row, col)
            score_gain = temp_game.score - prev_score
            total_reward += score_gain
            
            print(f"After move {i+1}: +{score_gain} points (Total: {temp_game.score})")
            
            # Show grid after this move
            print("Grid state:")
            for row_idx in range(10):
                row_str = ''.join(['■' if temp_game.grid[row_idx, j] == 1 else '□' for j in range(10)])
                print(f"{row_idx}: {row_str}")
            print()
        
        print(f"Total points gained: {total_reward}")

if __name__ == "__main__":
    print("1010! RL Solver")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
            agent = train_new_model(episodes)
            test_model_performance(agent)
            
        elif command == "test":
            agent = load_or_train_model()
            test_model_performance(agent)
            
        elif command == "demo":
            demonstrate_strategy()
            
        elif command == "gui":
            print("Starting GUI...")
            class TenTenGUI(TenTenGUI):
                def __init__(self):
                    super().__init__()
                    
                    # Replace agent
                    self.agent = MonteCarloAgent()
                    
                    # Try to load model
                    try:
                        self.agent.load_model("tenten_model.json")
                        print(f"Loaded model with {self.agent.episodes_trained} episodes trained")
                    except FileNotFoundError:
                        print("No model found. Loading/training model...")
                        self.agent = load_or_train_model()
            
            gui = TenTenGUI()
            gui.run()
            
        else:
            print("Unknown command. Available commands:")
            print("  train [episodes] - Train a new model")
            print("  test - Test model performance")
            print("  demo - Demonstrate AI strategy")
            print("  gui - Run GUI interface")
    
    else:
        # Default: run GUI
        print("Starting GUI interface...")
        print("Use 'python main.py [command]' for other options")
        
        # Import and modify GUI to use  agent
        class TenTenGUI(TenTenGUI):
            def __init__(self):
                # Initialize basic GUI first
                import tkinter as tk
                self.root = tk.Tk()
                self.root.title("1010! RL Solver Interface")
                self.root.geometry("800x700")
                
                self.game = TenTenGame()
                
                # Use  agent
                self.agent = load_or_train_model()
                
                self.setup_gui()
                self.selected_shapes = []
        
        gui = TenTenGUI()
        gui.run()
