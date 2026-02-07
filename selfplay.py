"""
Self-Play Data Generation
Generate training data by having MCTS play against itself (without neural network)
As per Exercise 6 requirements: Use MCTSPlayer to generate self-play games WITHOUT network guidance
"""

import os
import json
import pickle
import numpy as np
import time
import random
from datetime import datetime
from Gomoku import Gomoku
from MCTSPlayer import MCTSPlayer
from MCTSNode import MCTSNode


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


def get_mcts_policy(game, mcts_player, iterations=800):
    """
    Run MCTS and return visit count distribution as policy
    
    Args:
        game: Gomoku game instance
        mcts_player: MCTSPlayer instance
        iterations: number of MCTS iterations
    
    Returns:
        dict: move -> probability (based on visit counts)
    """
    legal_moves = game.legal_moves()
    
    if not legal_moves or game.status() is not None:
        return {}
    
    # Create root node and run MCTS
    root = MCTSNode(parent=None, move=None, untried_moves=legal_moves)
    root_player = game.to_move
    
    mcts_player._run_mcts(game, root, iterations, root_player)
    
    # Get visit counts from children
    visit_counts = {}
    for move, child in root.children.items():
        visit_counts[move] = child.visits
    
    # Convert to probabilities
    total_visits = sum(visit_counts.values())
    if total_visits == 0:
        # Shouldn't happen, but fallback to uniform
        return {move: 1.0 / len(legal_moves) for move in legal_moves}
    
    policy = {move: visits / total_visits for move, visits in visit_counts.items()}
    
    return policy


def play_self_play_game(board_size=9, mcts_iterations=800, temperature=1.0):
    """
    Play one self-play game using pure MCTS (no neural network)
    
    Args:
        board_size: size of the board
        mcts_iterations: number of MCTS iterations per move
        temperature: controls randomness (1.0 = stochastic, 0 = deterministic)
    
    Returns:
        SelfPlayGame object containing game data
    """
    game = Gomoku(size=board_size)
    player = MCTSPlayer(exploration_c=1.41421356237)
    
    game_data = SelfPlayGame()
    move_count = 0
    max_moves = board_size * board_size
    
    while game.status() is None and move_count < max_moves:
        # Get visit count distribution from MCTS
        policy = get_mcts_policy(game, player, mcts_iterations)
        
        if not policy:
            break
        
        # Save current position
        encoded_state = game.encode()
        current_player = game.to_move
        game_data.add_position(encoded_state, policy, current_player)
        
        # Choose move based on temperature
        if temperature == 0:
            # Deterministic: choose most visited
            chosen_move = max(policy.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic: sample proportional to visits^(1/temperature)
            if temperature == 1.0:
                # Use visit counts directly
                moves, probs = zip(*policy.items())
                chosen_move = random.choices(moves, weights=probs)[0]
            else:
                # Apply temperature
                temp_policy = {m: (p ** (1.0/temperature)) for m, p in policy.items()}
                total = sum(temp_policy.values())
                temp_policy = {m: p/total for m, p in temp_policy.items()}
                moves, probs = zip(*temp_policy.items())
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


def generate_self_play_data(num_games=100, board_size=9, 
                           mcts_iterations=800, temperature=1.0,
                           save_path=None, verbose=True):
    """
    Generate training data from multiple MCTS self-play games (WITHOUT neural network)
    
    Args:
        num_games: number of games to play
        board_size: size of the board
        mcts_iterations: MCTS iterations per move
        temperature: move selection randomness
        save_path: path to save data (None = don't save)
        verbose: print progress
    
    Returns:
        List of (state, policy, value) training samples
    """
    all_samples = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING MCTS SELF-PLAY DATA (No Network)")
        print(f"{'='*60}")
        print(f"Games: {num_games}")
        print(f"Board: {board_size}x{board_size}")
        print(f"MCTS Iterations: {mcts_iterations}")
        print(f"Temperature: {temperature}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for game_num in range(num_games):
        game_start = time.time()
        
        # Play one game using pure MCTS
        game_data = play_self_play_game(board_size, mcts_iterations, temperature)
        samples = game_data.get_training_samples()
        all_samples.extend(samples)
        
        game_time = time.time() - game_start
        
        # Print progress every game
        if verbose:
            elapsed = time.time() - start_time
            avg_time = elapsed / (game_num + 1)
            remaining = avg_time * (num_games - game_num - 1)
            
            outcome_str = {1: "P1 Win", -1: "P2 Win", 0: "Draw"}[game_data.outcome]
            
            print(f"[PROGRESS] Game {game_num + 1}/{num_games} | "
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

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npz":
        states = np.array([s[0] for s in samples], dtype=np.float32)
        values = np.array([s[2] for s in samples], dtype=np.float32)
        policy_moves = []
        policy_probs = []

        for policy in [s[1] for s in samples]:
            if policy:
                moves, probs = zip(*policy.items())
                policy_moves.append(np.array(moves, dtype=np.int16))
                policy_probs.append(np.array(probs, dtype=np.float32))
            else:
                policy_moves.append(np.empty((0, 2), dtype=np.int16))
                policy_probs.append(np.empty((0,), dtype=np.float32))

        np.savez_compressed(
            filepath,
            states=states,
            values=values,
            policy_moves=np.array(policy_moves, dtype=object),
            policy_probs=np.array(policy_probs, dtype=object)
        )
        return

    if ext == ".npy":
        np.save(filepath, np.array(samples, dtype=object), allow_pickle=True)
        return

    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)


def save_training_chunks(samples, board_size, base_dir, timestamp, chunk_size=5000, save_format="pkl", verbose=True):
    """
    Save samples into chunk files and update a manifest so future runs
    can train on all accumulated data.
    """
    os.makedirs(base_dir, exist_ok=True)
    chunk_files = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        chunk_idx = i // chunk_size
        ext = save_format.lower().lstrip(".")
        chunk_filename = f"mcts_selfplay_{board_size}x{board_size}_{timestamp}_chunk{chunk_idx:03d}.{ext}"
        chunk_path = os.path.join(base_dir, chunk_filename)
        save_training_data(chunk, chunk_path)
        chunk_files.append(chunk_path)

    manifest_path = os.path.join(base_dir, f"manifest_{board_size}x{board_size}.json")
    manifest = {
        "board_size": board_size,
        "chunk_files": [],
        "total_samples": 0
    }

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except Exception:
            if verbose:
                print("[WARN] Failed to read existing manifest, creating a new one.")

    # Append new chunks
    manifest["board_size"] = board_size
    manifest["chunk_files"].extend(chunk_files)
    manifest["total_samples"] = manifest.get("total_samples", 0) + len(samples)

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"✓ Saved {len(chunk_files)} chunk files")
        print(f"✓ Updated manifest: {manifest_path}")

    return manifest_path


def load_training_data(filepath):
    """Load training samples from file"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".npz":
        data = np.load(filepath, allow_pickle=True)
        states = data["states"]
        values = data["values"]
        policy_moves = data["policy_moves"]
        policy_probs = data["policy_probs"]

        samples = []
        for i in range(len(states)):
            moves_arr = policy_moves[i]
            probs_arr = policy_probs[i]
            policy = {}
            if moves_arr is not None and len(moves_arr) > 0:
                for move, prob in zip(moves_arr, probs_arr):
                    row, col = int(move[0]), int(move[1])
                    policy[(row, col)] = float(prob)
            samples.append((states[i].tolist(), policy, float(values[i])))
        return samples

    if ext == ".npy":
        data = np.load(filepath, allow_pickle=True)
        return data.tolist()

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    """Generate MCTS self-play data with default settings"""
    print("\n" + "[MCTS] SELF-PLAY DATA GENERATION".center(60))
    
    # Generate data using pure MCTS (no network)
    board_size = 9
    num_games = 100  # Reduced for faster iteration
    mcts_iterations = 10000  # MCTS iterations per move (increased for strength)
    
    # WARNING BEFORE STARTING
    print("\n" + "="*60)
    print("[WARNING] This will generate 100 self-play games")
    print("Data will be appended to the existing training set")
    print(f"Estimated time: depends on CPU speed")
    print("Progress will be shown every 10 games")
    print("="*60)
    print("\nWaiting for your confirmation to start...")
    print("Type 'yes' to begin: ", end="", flush=True)
    
    response = input().strip().lower()
    
    if response != "yes":
        print("Cancelled. Exiting.")
        return
    
    print("\n[START] Beginning 100 game generation...")
    print("="*60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_format = "npz"
    save_path = f"training_data/mcts_selfplay_{board_size}x{board_size}_{num_games}games_{timestamp}.{save_format}"
    
    samples = generate_self_play_data(
        num_games=num_games,
        board_size=board_size,
        mcts_iterations=mcts_iterations,
        temperature=1.0,
        save_path=save_path,
        verbose=True
    )

    # Save chunks and update manifest for accumulated training
    manifest_path = save_training_chunks(
        samples=samples,
        board_size=board_size,
        base_dir="training_data",
        timestamp=timestamp,
        chunk_size=5000,
        save_format=save_format,
        verbose=True
    )
    
    # Show sample statistics
    print("\n" + "="*60)
    print("[COMPLETE] Self-play data generation finished!")
    print("="*60)
    print("\nSample Statistics:")
    print(f"  Total positions: {len(samples)}")
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
    print(f"\nData saved to: {save_path}")
    print(f"Training manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
