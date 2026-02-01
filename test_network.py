"""
Test suite for GameNetwork (Neural Network for Gomoku)
"""

import torch
import os
from Gomoku import Gomoku
from GameNetwork import GameNetwork, GameNetworkOptimizer


def test_network_creation():
    """Test that network can be created with correct dimensions."""
    print("\n" + "="*60)
    print("TEST 1: Network Creation")
    print("="*60)
    
    network = GameNetwork(board_size=15, hidden_size=128)
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"✓ Network created successfully")
    print(f"  Board size: 15x15")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {network.get_device()}")
    
    assert network.board_size == 15
    assert network.num_moves == 225
    print(f"\n✅ Network creation test passed!\n")


def test_forward_pass():
    """Test forward pass with Gomoku encoded state."""
    print("="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    game = Gomoku(size=15)
    network = GameNetwork(board_size=15)
    
    # Encode game state
    encoded = game.encode()
    print(f"✓ Encoded state shape: {len(encoded)}")
    
    # Forward pass
    value, policy_logits = network.forward(encoded)
    
    print(f"✓ Value shape: {value.shape}")
    print(f"✓ Policy logits shape: {policy_logits.shape}")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
    print(f"  Policy logits range: [{policy_logits.min().item():.3f}, {policy_logits.max().item():.3f}]")
    
    # Check dimensions
    assert value.shape == (1, 1), "Value should be (1, 1)"
    assert policy_logits.shape == (1, 225), "Policy should be (1, 225)"
    
    # Check value is in correct range after tanh
    assert -1 <= value.item() <= 1, "Value should be in [-1, 1]"
    
    print(f"\n✅ Forward pass test passed!\n")


def test_batch_forward():
    """Test batched forward pass."""
    print("="*60)
    print("TEST 3: Batch Forward Pass")
    print("="*60)
    
    game = Gomoku(size=15)
    network = GameNetwork(board_size=15)
    
    # Create batch of 3 game states
    batch_encoded = []
    for i in range(3):
        if i > 0:
            game.make_move((i, i))
        batch_encoded.append(game.encode())
        if i > 0:
            game.unmake_move((i, i))
    
    batch_tensor = torch.tensor(batch_encoded, dtype=torch.float32)
    print(f"✓ Batch shape: {batch_tensor.shape}")
    
    # Forward pass
    values, policy_logits = network.forward(batch_tensor)
    
    print(f"✓ Batch values shape: {values.shape}")
    print(f"✓ Batch policy shape: {policy_logits.shape}")
    
    assert values.shape == (3, 1), "Batch values should be (3, 1)"
    assert policy_logits.shape == (3, 225), "Batch policy should be (3, 225)"
    
    print(f"\n✅ Batch forward pass test passed!\n")


def test_predict():
    """Test predict function with game object."""
    print("="*60)
    print("TEST 4: Predict Function")
    print("="*60)
    
    game = Gomoku(size=15)
    network = GameNetwork(board_size=15)
    
    # Make a few moves
    moves = [(7, 7), (7, 8), (8, 7)]
    for move in moves:
        game.make_move(move)
    
    print(f"✓ Made {len(moves)} moves")
    print(f"  Legal moves: {len(game.legal_moves())}")
    
    # Predict
    value, policy_dict = network.predict(game)
    
    print(f"✓ Value prediction: {value:.4f}")
    print(f"✓ Policy dict size: {len(policy_dict)} moves")
    print(f"  Top 3 moves:")
    
    sorted_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    for move, prob in sorted_moves:
        print(f"    {move}: {prob:.4f}")
    
    # Check predictions
    assert -1 <= value <= 1, "Value should be in [-1, 1]"
    assert len(policy_dict) == len(game.legal_moves()), "Policy should cover all legal moves"
    assert abs(sum(policy_dict.values()) - 1.0) < 0.01, "Policy should sum to ~1.0"
    
    print(f"\n✅ Predict function test passed!\n")


def test_save_load():
    """Test saving and loading network weights."""
    print("="*60)
    print("TEST 5: Save and Load")
    print("="*60)
    
    # Create and make prediction with network1
    network1 = GameNetwork(board_size=15, hidden_size=64)
    game = Gomoku(size=15)
    
    value1, policy1 = network1.predict(game)
    print(f"✓ Network1 value: {value1:.4f}")
    
    # Save weights
    save_path = "models/test_network.pth"
    network1.save(save_path)
    
    assert os.path.exists(save_path), "File should exist"
    print(f"✓ Saved to {save_path}")
    
    # Create new network and load weights
    network2 = GameNetwork(board_size=15, hidden_size=64)
    network2.load(save_path)
    
    value2, policy2 = network2.predict(game)
    print(f"✓ Network2 value: {value2:.4f}")
    
    # Check they match
    assert abs(value1 - value2) < 1e-6, "Values should match after load"
    
    # Get top moves from both
    sorted_p1 = sorted(policy1.items(), key=lambda x: x[1], reverse=True)[:3]
    sorted_p2 = sorted(policy2.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print(f"  Network1 top move: {sorted_p1[0]}")
    print(f"  Network2 top move: {sorted_p2[0]}")
    
    # Clean up
    os.remove(save_path)
    os.rmdir("models")
    print(f"✓ Cleaned up test files")
    
    print(f"\n✅ Save/Load test passed!\n")


def test_legal_moves_masking():
    """Test that illegal moves are properly masked."""
    print("="*60)
    print("TEST 6: Legal Moves Masking")
    print("="*60)
    
    game = Gomoku(size=15)
    network = GameNetwork(board_size=15)
    
    # Make some moves to reduce legal moves
    occupied_moves = [(7, 7), (7, 8), (8, 7), (8, 8), (6, 7)]
    for move in occupied_moves:
        game.make_move(move)
    
    legal_moves = game.legal_moves()
    print(f"✓ Legal moves: {len(legal_moves)}")
    print(f"✓ Occupied moves: {len(occupied_moves)}")
    
    # Get prediction
    value, policy_dict = network.predict(game, legal_moves)
    
    # Check that only legal moves are in policy
    for move in policy_dict.keys():
        assert move in legal_moves, f"Move {move} should be legal"
    
    # Check that occupied moves are not in policy
    for move in occupied_moves:
        assert move not in policy_dict, f"Move {move} should not be in policy (occupied)"
    
    print(f"✓ All policy moves are legal")
    print(f"✓ No occupied moves in policy")
    
    print(f"\n✅ Legal moves masking test passed!\n")


def test_training_step():
    """Test that training step works."""
    print("="*60)
    print("TEST 7: Training Step")
    print("="*60)
    
    network = GameNetwork(board_size=15, hidden_size=64)
    optimizer = GameNetworkOptimizer(network, learning_rate=0.001)
    
    # Create dummy training data
    game = Gomoku(size=15)
    batch_size = 4
    
    encoded_states = []
    target_values = []
    target_policies = []
    
    for i in range(batch_size):
        encoded_states.append(game.encode())
        target_values.append(0.5 if i % 2 == 0 else -0.5)
        
        # Create dummy policy (uniform over all moves)
        policy = torch.zeros(225)
        policy[i * 10:(i+1) * 10] = 1.0 / 10  # Spread probability over 10 moves
        target_policies.append(policy)
    
    encoded_states = torch.tensor(encoded_states, dtype=torch.float32)
    target_values = torch.tensor(target_values, dtype=torch.float32)
    target_policies = torch.stack(target_policies)
    
    print(f"✓ Created batch of {batch_size} samples")
    
    # Get initial prediction
    with torch.no_grad():
        initial_value, _ = network.forward(encoded_states[0:1])
    initial_val = initial_value.item()
    
    print(f"  Initial value prediction: {initial_val:.4f}")
    
    # Training step
    losses = optimizer.train_step(encoded_states, target_values, target_policies)
    
    print(f"✓ Training step completed")
    print(f"  Total loss: {losses['total_loss']:.4f}")
    print(f"  Value loss: {losses['value_loss']:.4f}")
    print(f"  Policy loss: {losses['policy_loss']:.4f}")
    
    # Get updated prediction
    with torch.no_grad():
        updated_value, _ = network.forward(encoded_states[0:1])
    updated_val = updated_value.item()
    
    print(f"  Updated value prediction: {updated_val:.4f}")
    print(f"  Value changed by: {abs(updated_val - initial_val):.4f}")
    
    # Check that losses are reasonable
    assert losses['total_loss'] > 0, "Loss should be positive"
    assert losses['value_loss'] >= 0, "Value loss should be non-negative"
    assert losses['policy_loss'] >= 0, "Policy loss should be non-negative"
    
    print(f"\n✅ Training step test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "🧪 GAMENETWORK TEST SUITE 🧪".center(60))
    
    try:
        test_network_creation()
        test_forward_pass()
        test_batch_forward()
        test_predict()
        test_save_load()
        test_legal_moves_masking()
        test_training_step()
        
        print("\n" + "="*60)
        print("🎉 ALL NETWORK TESTS PASSED! 🎉")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
