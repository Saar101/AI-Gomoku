"""
Self-Play Data Generation
Generate training data by having PUCT play against itself
"""

import os
import pickle
import time
from datetime import datetime
from Gomoku import Gomoku
from GameNetwork import GameNetwork
from PUCTPlayer import PUCTPlayer


class SelfPlayGame:
    """Container for self-play game data"""
    
    def __init__(self):
        self.states = []  # List of encoded game states
        self.policies = []  # List of MCTS policy distributions
        self.current_player = []  # List of which player is to move
        self.outcome = None  # Final game result (1, -1, or 0)
    
    def add_position(self, encoded_state, policy_dict, player):
        """Add a position to the game"""
        self.states.append(encoded_state)
        self.policies.append(policy_dict)
        self.current_player.append(player)
    
    def set_outcome(self, result):
        """Set the final game outcome"""
        self.outcome = result
    
    def get_training_samples(self):
        """
        Convert game data to training samples
        
        Returns:
            List of (state, policy, value) tuples where:
            - state: encoded board state
            - policy: dict of move -> probability
            - value: outcome from that player's perspective
        """
        samples = []
        
        for state, policy, player in zip(self.states, self.policies, self.current_player):
            # Value is outcome from this player's perspective
            if self.outcome == 0:
                value = 0.0  # Draw
            elif self.outcome == player:
                value = 1.0  # This player won
            else:
                value = -1.0  # This player lost
            
            samples.append((state, policy, value))
        
        return samples


def play_self_play_game(network, board_size=9, num_simulations=100, temperature=1.0):
    """
    Play one self-play game
    
    Args:
        network: GameNetwork instance
        board_size: size of the board
        num_simulations: MCTS simulations per move
        temperature: controls randomness (1.0 = stochastic, 0 = deterministic)
    
    Returns:
        SelfPlayGame object containing game data
    """
    game = Gomoku(size=board_size)
    player = PUCTPlayer(network, c_puct=1.0, num_simulations=num_simulations)
    
    game_data = SelfPlayGame()
    move_count = 0
    max_moves = board_size * board_size
    
    while game.status() is None and move_count < max_moves:
        # Get action probabilities from MCTS
        action_probs = player.get_action_probs(game, temperature=temperature)
        
        if not action_probs:
            break
        
        # Save current position
        encoded_state = game.encode()
        current_player = game.to_move
        game_data.add_position(encoded_state, action_probs, current_player)
        
        # Choose move (stochastic during training)
        import random
        moves, probs = zip(*action_probs.items())
        chosen_move = random.choices(moves, weights=probs)[0]
        
        # Apply move
        game.make_move(chosen_move)
        move_count += 1
    
    # Set final outcome
    outcome = game.status()
    if outcome is None:
        outcome = 0  # Draw if max moves reached
    
    game_data.set_outcome(outcome)
    
    return game_data


def generate_self_play_data(network, num_games=100, board_size=9, 
                           num_simulations=100, temperature=1.0,
                           save_path=None, verbose=True):
    """
    Generate training data from multiple self-play games
    
    Args:
        network: GameNetwork instance
        num_games: number of games to play
        board_size: size of the board
        num_simulations: MCTS simulations per move
        temperature: move selection randomness
        save_path: path to save data (None = don't save)
        verbose: print progress
    
    Returns:
        List of (state, policy, value) training samples
    """
    all_samples = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING SELF-PLAY DATA")
        print(f"{'='*60}")
        print(f"Games: {num_games}")
        print(f"Board: {board_size}x{board_size}")
        print(f"Simulations: {num_simulations}")
        print(f"Temperature: {temperature}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for game_num in range(num_games):
        game_start = time.time()
        
        # Play one game
        game_data = play_self_play_game(network, board_size, num_simulations, temperature)
        samples = game_data.get_training_samples()
        all_samples.extend(samples)
        
        game_time = time.time() - game_start
        
        # Print progress
        if verbose and (game_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (game_num + 1)
            remaining = avg_time * (num_games - game_num - 1)
            
            outcome_str = {1: "P1 Win", -1: "P2 Win", 0: "Draw"}[game_data.outcome]
            
            print(f"Game {game_num + 1}/{num_games} | "
                  f"{outcome_str} | "
                  f"{len(samples)} positions | "
                  f"{game_time:.1f}s | "
                  f"ETA: {remaining/60:.1f}m")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total games: {num_games}")
        print(f"Total positions: {len(all_samples)}")
        print(f"Avg positions per game: {len(all_samples)/num_games:.1f}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"{'='*60}\n")
    
    # Save data if path provided
    if save_path:
        save_training_data(all_samples, save_path)
        if verbose:
            print(f"✓ Data saved to {save_path}\n")
    
    return all_samples


def save_training_data(samples, filepath):
    """Save training samples to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)


def load_training_data(filepath):
    """Load training samples from file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    """Generate self-play data with default settings"""
    print("\n" + "🎮 SELF-PLAY DATA GENERATION 🎮".center(60))
    
    # Create network
    board_size = 9
    network = GameNetwork(board_size=board_size, hidden_size=128)
    
    # Generate data (start with small number for testing)
    num_games = 50  # Start small, can increase to 10,000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"training_data/selfplay_{board_size}x{board_size}_{num_games}games_{timestamp}.pkl"
    
    samples = generate_self_play_data(
        network=network,
        num_games=num_games,
        board_size=board_size,
        num_simulations=100,
        temperature=1.0,
        save_path=save_path,
        verbose=True
    )
    
    # Show sample statistics
    print("Sample Statistics:")
    print(f"  States: {len(samples)} positions")
    print(f"  State shape: {len(samples[0][0])} features")
    print(f"  Policy moves: {len(samples[0][1])} (varies by position)")
    print(f"  Value range: [{min(s[2] for s in samples):.1f}, {max(s[2] for s in samples):.1f}]")
    
    # Count outcomes
    outcomes = [s[2] for s in samples]
    wins = sum(1 for v in outcomes if v > 0)
    losses = sum(1 for v in outcomes if v < 0)
    draws = sum(1 for v in outcomes if v == 0)
    
    print(f"\nOutcome Distribution:")
    print(f"  Wins:   {wins} ({wins/len(outcomes)*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/len(outcomes)*100:.1f}%)")
    print(f"  Draws:  {draws} ({draws/len(outcomes)*100:.1f}%)")


if __name__ == "__main__":
    main()
