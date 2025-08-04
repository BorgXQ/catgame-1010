from game_gui import TenTenGUI

if __name__ == "__main__":
    # Option 1: Run GUI interface
    print("Starting interface...")
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