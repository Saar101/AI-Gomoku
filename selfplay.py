"""
Self-Play Data Generation (STRONG MCTS, matches play_vs_mcts behavior)

Generate training data by having MCTS play against itself (without neural network)
BUT ensure the MCTS behavior matches play_vs_mcts:
- same tactical shortcuts (immediate win / immediate block)
- same final move selection policy (highest visits, tie-break by value)
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
        self.states = []          # List of encoded game states
        self.policies = []        # List of MCTS policy distributions
        self.current_player = []  # List of which player is to move
        self.outcome = None       # Final game result (1, -1, or 0)

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
            - value: absolute outcome (+1 black win, -1 white win, 0 draw)
        """
        samples = []

        value = float(self.outcome) if self.outcome is not None else 0.0

        for state, policy, _player in zip(self.states, self.policies, self.current_player):

            samples.append((state, policy, value))

        return samples


def _one_hot_policy(move, legal_moves):
    """Return a policy dict that is 1.0 on `move` (if legal) and 0.0 otherwise."""
    if move is None:
        return {}
    if move not in legal_moves:
        # Fallback (should not happen)
        return {m: 1.0 / len(legal_moves) for m in legal_moves} if legal_moves else {}
    return {move: 1.0}


def get_mcts_policy_and_move(game, player, iterations=10000):
    """
    Run "strong MCTS" exactly like play_vs_mcts:
    1) immediate win
    2) immediate block
    3) otherwise run MCTS and pick move using _select_best_move

    Returns:
        (policy_dict, chosen_move)
    """
    legal_moves = game.legal_moves()
    if not legal_moves or game.status() is not None:
        return {}, None

    # --- 1) Immediate win (same behavior as choose_move) ---
    winning_move = player._find_immediate_winning_move(game, legal_moves)
    if winning_move is not None:
        return _one_hot_policy(winning_move, legal_moves), winning_move

    # --- 2) Immediate block (same behavior as choose_move) ---
    blocking_move = player._find_immediate_blocking_move(game, legal_moves)
    if blocking_move is not None:
        return _one_hot_policy(blocking_move, legal_moves), blocking_move

    # --- 3) Full MCTS (same as choose_move) ---
    root_player = game.to_move
    root = MCTSNode(parent=None, move=None, untried_moves=legal_moves)

    # Run MCTS exactly as in the real player
    player._run_mcts(game, root, iterations, root_player)

    # Visit-count policy
    visit_counts = {move: child.visits for move, child in root.children.items()}
    total_visits = sum(visit_counts.values())

    if total_visits <= 0:
        # Fallback uniform
        policy = {m: 1.0 / len(legal_moves) for m in legal_moves}
    else:
        policy = {m: v / total_visits for m, v in visit_counts.items()}

    # Choose move EXACTLY like play_vs_mcts
    chosen_move = player._select_best_move(root, root_player, legal_moves)

    # Safety fallback
    if chosen_move is None:
        chosen_move = random.choice(legal_moves)

    return policy, chosen_move


def play_self_play_game(board_size=9, mcts_iterations=10000, temperature=0.0):
    """
    Play one self-play game using STRONG MCTS that matches play_vs_mcts behavior.

    Args:
        board_size: size of the board
        mcts_iterations: number of MCTS iterations per move (play_vs_mcts uses 10000)
        temperature:
            0.0 = deterministic (recommended to match play_vs_mcts strength)
            >0  = sample from policy^(1/temperature)

    Returns:
        SelfPlayGame object containing game data
    """
    game = Gomoku(size=board_size)
    player = MCTSPlayer(exploration_c=1.41421356237)

    game_data = SelfPlayGame()
    move_count = 0
    max_moves = board_size * board_size

    while game.status() is None and move_count < max_moves:
        policy, best_move = get_mcts_policy_and_move(game, player, mcts_iterations)

        if not policy or best_move is None:
            break

        # Save current position
        encoded_state = game.encode()
        current_player = game.to_move
        game_data.add_position(encoded_state, policy, current_player)

        # Choose move based on temperature (optional)
        if temperature == 0.0:
            chosen_move = best_move
        else:
            # sample from policy^(1/temperature)
            moves, probs = zip(*policy.items())
            if temperature != 1.0:
                probs = [p ** (1.0 / temperature) for p in probs]
                s = sum(probs)
                probs = [p / s for p in probs]
            chosen_move = random.choices(moves, weights=probs)[0]

        game.make_move(chosen_move)
        move_count += 1

    outcome = game.status()
    if outcome is None:
        outcome = 0

    game_data.set_outcome(outcome)
    return game_data


def generate_self_play_data(num_games=100, board_size=9,
                           mcts_iterations=10000, temperature=0.0,
                           save_path=None, verbose=True):
    """
    Generate training data from multiple STRONG MCTS self-play games.

    Args:
        num_games: number of games to play
        board_size: size of the board
        mcts_iterations: MCTS iterations per move
        temperature: move selection randomness (0.0 = deterministic strongest)
        save_path: path to save data (None = don't save)
        verbose: print progress

    Returns:
        List of (state, policy, value) training samples
    """
    all_samples = []

    if verbose:
        print(f"\n{'='*60}")
        print("GENERATING STRONG MCTS SELF-PLAY DATA (matches play_vs_mcts)")
        print(f"{'='*60}")
        print(f"Games: {num_games}")
        print(f"Board: {board_size}x{board_size}")
        print(f"MCTS Iterations: {mcts_iterations}")
        print(f"Temperature: {temperature}")
        print(f"{'='*60}\n")

    start_time = time.time()

    for game_num in range(num_games):
        game_start = time.time()

        game_data = play_self_play_game(board_size, mcts_iterations, temperature)
        samples = game_data.get_training_samples()
        all_samples.extend(samples)

        game_time = time.time() - game_start

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
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total games: {num_games}")
        print(f"Total positions: {len(all_samples)}")
        print(f"Avg positions per game: {len(all_samples)/num_games:.1f}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"{'='*60}\n")

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
            policy_probs=np.array(policy_probs, dtype=object),
        )
        return

    if ext == ".npy":
        np.save(filepath, np.array(samples, dtype=object), allow_pickle=True)
        return

    with open(filepath, "wb") as f:
        pickle.dump(samples, f)


def save_training_chunks(samples, board_size, base_dir, timestamp, chunk_size=5000, save_format="pkl", verbose=True):
    """Save samples into chunk files and update a manifest."""
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
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            if verbose:
                print("[WARN] Failed to read existing manifest, creating a new one.")

    manifest["board_size"] = board_size
    manifest["chunk_files"].extend(chunk_files)
    manifest["total_samples"] = manifest.get("total_samples", 0) + len(samples)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(f"✓ Saved {len(chunk_files)} chunk files")
        print(f"✓ Updated manifest: {manifest_path}")

    return manifest_path


def main():
    print("\n" + "[MCTS] SELF-PLAY DATA GENERATION (STRONG)".center(60))

    board_size = 9
    num_games = 100
    mcts_iterations = 10000

    print("\n" + "=" * 60)
    print("[WARNING] This will generate 100 strong self-play games")
    print("This version MATCHES play_vs_mcts behavior (win/block + robust child).")
    print(f"MCTS iterations per move: {mcts_iterations}")
    print("Temperature default is 0.0 (deterministic strongest).")
    print("=" * 60)
    print("\nType 'yes' to begin: ", end="", flush=True)

    response = input().strip().lower()
    if response != "yes":
        print("Cancelled. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_format = "npz"
    save_path = f"training_data/mcts_selfplay_{board_size}x{board_size}_{num_games}games_{timestamp}.{save_format}"

    samples = generate_self_play_data(
        num_games=num_games,
        board_size=board_size,
        mcts_iterations=mcts_iterations,
        temperature=0.0,
        save_path=save_path,
        verbose=True
    )

    manifest_path = save_training_chunks(
        samples=samples,
        board_size=board_size,
        base_dir="training_data",
        timestamp=timestamp,
        chunk_size=5000,
        save_format=save_format,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("[COMPLETE] Self-play data generation finished!")
    print("=" * 60)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Data saved to: {save_path}")
    print(f"Training manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()