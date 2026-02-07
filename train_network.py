"""
Train Neural Network on Self-Play Data
Trains value head (predict outcomes) and policy head (predict MCTS visit counts)
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
from datetime import datetime
from Gomoku import Gomoku
from GameNetwork import GameNetwork, GameNetworkOptimizer


def load_training_data(filepath):
    """Load self-play training data"""
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
        data = pickle.load(f)
    return data


def load_training_manifest(manifest_path):
    """Load manifest and return chunk file list and metadata"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    chunk_files = manifest.get("chunk_files", [])
    return chunk_files, manifest


def prepare_batch(samples, batch_size):
    """
    Prepare mini-batches from training data
    
    Args:
        samples: list of (state, policy_dict, value) tuples
        batch_size: batch size
    
    Yields:
        (states_tensor, values_tensor, policies_tensor)
    """
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        
        # Extract components
        states = [s[0] for s in batch]
        policies = [s[1] for s in batch]
        values = [s[2] for s in batch]
        
        # Convert states to tensor
        states_tensor = torch.tensor(states, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        
        # Convert policies to tensor (one-hot over 225 moves for 15x15, 81 for 9x9)
        # Get board size from state length
        state_len = len(states[0])
        if state_len == 163:  # 9x9 board
            num_moves = 81
        elif state_len == 451:  # 15x15 board
            num_moves = 225
        else:
            raise ValueError(f"Unknown board size for state length {state_len}")
        
        policies_tensor = torch.zeros((len(batch), num_moves), dtype=torch.float32)
        
        for batch_idx, policy_dict in enumerate(policies):
            for move, prob in policy_dict.items():
                row, col = move
                board_size = int(num_moves ** 0.5)
                move_idx = row * board_size + col
                policies_tensor[batch_idx, move_idx] = prob
        
        yield states_tensor, values_tensor, policies_tensor


def iter_batches_from_files(file_paths, batch_size):
    """
    Stream training batches from a list of pickle files.
    Loads one file at a time to keep memory usage bounded.
    """
    for path in file_paths:
        samples = load_training_data(path)
        for batch in prepare_batch(samples, batch_size):
            yield batch


def train_epoch(network, optimizer, data_loader, epoch, total_epochs):
    """
    Train for one epoch
    
    Args:
        network: GameNetwork instance
        optimizer: GameNetworkOptimizer instance
        data_loader: generator yielding batches
        epoch: current epoch number
        total_epochs: total number of epochs
    
    Returns:
        dict with average losses for the epoch
    """
    total_loss = 0.0
    value_loss = 0.0
    policy_loss = 0.0
    num_batches = 0
    
    network.train()  # Set to training mode
    
    for batch_states, batch_values, batch_policies in data_loader:
        # Training step
        losses = optimizer.train_step(batch_states, batch_values, batch_policies)
        
        total_loss += losses['total_loss']
        value_loss += losses['value_loss']
        policy_loss += losses['policy_loss']
        num_batches += 1
    
    # Average losses
    avg_total = total_loss / num_batches
    avg_value = value_loss / num_batches
    avg_policy = policy_loss / num_batches
    
    return {
        'total_loss': avg_total,
        'value_loss': avg_value,
        'policy_loss': avg_policy
    }


def train_network(data_path, board_size=9, hidden_size=128, 
                 epochs=10, batch_size=32, learning_rate=0.001,
                 save_path=None, verbose=True):
    """
    Train neural network on self-play data
    
    Args:
        data_path: path to training data pickle file
        board_size: size of game board
        hidden_size: network hidden layer size
        epochs: number of training epochs
        batch_size: training batch size
        learning_rate: learning rate
        save_path: where to save trained model
        verbose: print progress
    
    Returns:
        trained network, training history
    """
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING NEURAL NETWORK")
        print(f"{'='*60}")
        print(f"Data: {data_path}")
        print(f"Board: {board_size}x{board_size}")
        print(f"Hidden: {hidden_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch: {batch_size}")
        print(f"LR: {learning_rate}")
        print(f"{'='*60}\n")
    
    # Load data
    if verbose:
        print(f"Loading training data...")

    using_manifest = data_path.endswith(".json") and os.path.exists(data_path)

    if using_manifest:
        chunk_files, manifest = load_training_manifest(data_path)
        if not chunk_files:
            raise ValueError("Manifest has no chunk files to train on.")
        total_samples = manifest.get("total_samples", None)
        if verbose:
            total_str = f"{total_samples}" if total_samples is not None else "unknown"
            print(f"✓ Loaded manifest with {len(chunk_files)} chunk files")
            print(f"✓ Total samples (manifest): {total_str}")
            print("Chunk files used for training:")
            for path in chunk_files:
                print(f"  - {path}")
    else:
        samples = load_training_data(data_path)
        print(f"✓ Loaded {len(samples)} training samples")
    
    # Create network and optimizer
    network = GameNetwork(board_size=board_size, hidden_size=hidden_size)
    optimizer = GameNetworkOptimizer(network, learning_rate=learning_rate)
    
    if verbose:
        total_params = sum(p.numel() for p in network.parameters())
        print(f"✓ Created network with {total_params:,} parameters\n")
    
    # Training loop
    history = {
        'epoch': [],
        'total_loss': [],
        'value_loss': [],
        'policy_loss': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Create data loader for this epoch
        if using_manifest:
            data_loader = iter_batches_from_files(chunk_files, batch_size)
        else:
            data_loader = prepare_batch(samples, batch_size)
        
        # Train one epoch
        losses = train_epoch(network, optimizer, data_loader, epoch, epochs)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['total_loss'].append(losses['total_loss'])
        history['value_loss'].append(losses['value_loss'])
        history['policy_loss'].append(losses['policy_loss'])
        
        if verbose and ((epoch + 1) % 1 == 0):
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Total: {losses['total_loss']:.4f} | "
                  f"Value: {losses['value_loss']:.4f} | "
                  f"Policy: {losses['policy_loss']:.4f} | "
                  f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Avg epoch time: {total_time/epochs:.1f}s")
        print(f"Final losses:")
        print(f"  Total:  {history['total_loss'][-1]:.4f}")
        print(f"  Value:  {history['value_loss'][-1]:.4f}")
        print(f"  Policy: {history['policy_loss'][-1]:.4f}")
    
    # Save model if path provided
    if save_path:
        network.save(save_path)
        if verbose:
            print(f"✓ Model saved to {save_path}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return network, history


def evaluate_network(network, samples, board_size=9, num_samples=100):
    """
    Evaluate network on test samples
    
    Args:
        network: GameNetwork instance
        samples: list of training samples
        board_size: board size
        num_samples: number of samples to evaluate
    
    Returns:
        dict with evaluation metrics
    """
    network.eval()
    
    test_samples = samples[:min(num_samples, len(samples))]
    
    # Get network predictions
    states = torch.tensor([s[0] for s in test_samples], dtype=torch.float32)
    
    with torch.no_grad():
        predicted_values, predicted_policies = network.forward(states)
    
    predicted_values = predicted_values.squeeze().numpy()
    predicted_policies = predicted_policies.numpy()
    
    # Compare with ground truth
    true_values = [s[2] for s in test_samples]
    true_policies = [s[1] for s in test_samples]
    
    # Value accuracy
    value_errors = [abs(predicted_values[i] - true_values[i]) 
                   for i in range(len(test_samples))]
    avg_value_error = sum(value_errors) / len(value_errors)
    
    # Policy accuracy (top-1 accuracy)
    policy_correct = 0
    for i, true_policy in enumerate(true_policies):
        if true_policy:
            true_top_move = max(true_policy.items(), key=lambda x: x[1])[0]
            row, col = true_top_move
            true_top_idx = row * board_size + col
            
            pred_top_idx = predicted_policies[i].argmax()
            
            if pred_top_idx == true_top_idx:
                policy_correct += 1
    
    policy_accuracy = policy_correct / len(test_samples)
    
    return {
        'avg_value_error': avg_value_error,
        'policy_top1_accuracy': policy_accuracy
    }


def load_samples_for_evaluation(data_path, num_samples=200):
    """Load a limited number of samples for evaluation."""
    using_manifest = data_path.endswith(".json") and os.path.exists(data_path)

    if not using_manifest:
        samples = load_training_data(data_path)
        return samples[:min(num_samples, len(samples))]

    chunk_files, _ = load_training_manifest(data_path)
    collected = []

    for path in chunk_files:
        samples = load_training_data(path)
        for sample in samples:
            collected.append(sample)
            if len(collected) >= num_samples:
                return collected

    return collected


def main():
    """Train network on MCTS self-play data"""
    print("\n" + "🧠 NEURAL NETWORK TRAINING 🧠".center(60))
    
    # Find latest self-play data file or manifest
    data_dir = "training_data"
    if not os.path.exists(data_dir):
        print(f"Error: No training data directory found!")
        return
    
    manifest_9 = os.path.join(data_dir, "manifest_9x9.json")
    manifest_15 = os.path.join(data_dir, "manifest_15x15.json")

    if os.path.exists(manifest_9) or os.path.exists(manifest_15):
        if os.path.exists(manifest_9):
            data_path = manifest_9
            board_size = 9
        else:
            data_path = manifest_15
            board_size = 15
        print(f"Using manifest: {os.path.basename(data_path)}")
    else:
        files = [
            f for f in os.listdir(data_dir)
            if f.startswith("mcts_selfplay") and f.endswith((".pkl", ".npz", ".npy"))
        ]
        if not files:
            print(f"Error: No self-play data files found in {data_dir}!")
            return
        
        latest_file = sorted(files)[-1]
        data_path = os.path.join(data_dir, latest_file)
        
        print(f"Using data: {latest_file}")
        
        # Determine board size from filename
        if "9x9" in latest_file:
            board_size = 9
        elif "15x15" in latest_file:
            board_size = 15
        else:
            board_size = 9  # default
    
    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/gomoku_trained_{board_size}x{board_size}_{timestamp}.pth"
    
    network, history = train_network(
        data_path=data_path,
        board_size=board_size,
        hidden_size=256,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        save_path=save_path,
        verbose=True
    )
    
    # Evaluate
    print(f"Evaluating network on test samples...")
    
    eval_samples = load_samples_for_evaluation(data_path, num_samples=200)
    metrics = evaluate_network(network, eval_samples, board_size=board_size, num_samples=200)
    
    print(f"\nEvaluation Results:")
    print(f"  Value MAE: {metrics['avg_value_error']:.4f}")
    print(f"  Policy Top-1: {metrics['policy_top1_accuracy']:.2%}")


if __name__ == "__main__":
    main()
