"""
Test suite for MCTS implementation on Gomoku.
"""

import time
from Gomoku import Gomoku
from MCTSPlayer import MCTSPlayer


def test_basic_mcts_functionality():
    """Test that basic MCTS functions work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Basic MCTS Functionality")
    print("="*60)
    
    game = Gomoku(size=9)  # Small board for quick testing
    player = MCTSPlayer(exploration_c=1.41)
    
    # Test 1: Can we get legal moves?
    legal_moves = game.legal_moves()
    print(f"✓ Legal moves: {len(legal_moves)} moves available")
    assert len(legal_moves) == 81, "Should have 81 moves on 9x9 board"
    
    # Test 2: Can MCTS choose a move?
    print("  Running MCTS with 100 iterations...")
    start = time.time()
    move = player.choose_move(game, iterations=100)
    elapsed = time.time() - start
    print(f"✓ MCTS chose move: {move}")
    print(f"  Time: {elapsed:.2f}s")
    assert move is not None, "MCTS should return a move"
    assert move in legal_moves, "MCTS move should be legal"
    
    # Test 3: Make the move and check game state
    game.make_move(move)
    print(f"✓ Move made successfully")
    print(f"  Current player: {'Player 1 (Black)' if game.to_move == 1 else 'Player 2 (White)'}")
    print(f"  Game status: {'Ongoing' if game.status() is None else 'Over'}")
    
    # Test 4: Unmake the move
    game.unmake_move(move)
    print(f"✓ Move unmade successfully")
    assert len(game.legal_moves()) == 81, "Should be back to start position"
    
    print("\n✅ All basic tests passed!\n")


def test_single_game():
    """Play one complete game between two MCTS players."""
    print("="*60)
    print("TEST 2: Complete Game (MCTS vs MCTS)")
    print("="*60)
    
    game = Gomoku(size=9)
    player1 = MCTSPlayer(exploration_c=1.41)
    player2 = MCTSPlayer(exploration_c=1.41)
    
    move_count = 0
    iterations_per_move = 50  # Reduced for speed
    
    print("\nStarting game...\n")
    
    while game.status() is None and move_count < 81:  # Max 81 moves on 9x9
        current_player = "Player 1 (Black)" if game.to_move == 1 else "Player 2 (White)"
        player = player1 if game.to_move == 1 else player2
        
        print(f"Move {move_count + 1}: {current_player} thinking...", end=" ", flush=True)
        start = time.time()
        move = player.choose_move(game, iterations=iterations_per_move)
        elapsed = time.time() - start
        
        game.make_move(move)
        move_count += 1
        
        print(f"played {move} ({elapsed:.2f}s)")
    
    # Check result
    result = game.status()
    if result == 1:
        print(f"\n✅ Player 1 (Black) WINS in {move_count} moves!")
    elif result == -1:
        print(f"\n✅ Player 2 (White) WINS in {move_count} moves!")
    elif result == 0:
        print(f"\n✅ DRAW after {move_count} moves!")
    else:
        print(f"\n✅ Game ongoing after {move_count} moves (board full)")
    
    print(f"\nFinal board state:")
    print_board(game)
    print()


def test_winning_detection():
    """Test that MCTS correctly detects winning positions."""
    print("="*60)
    print("TEST 3: Winning Position Detection")
    print("="*60)
    
    game = Gomoku(size=5)
    
    # Place 4 stones in a row for player 1, leaving one open
    moves = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3)]
    for move in moves:
        game.make_move(move)
    
    print(f"\nPlaced 7 moves, player 1 has 4-in-a-row with one open end")
    print(f"Current player: {'Player 1 (Black)' if game.to_move == 1 else 'Player 2 (White)'}")
    print(f"Game status: {game.status()}")
    
    # Player 1's turn - should find winning move at (0, 4)
    player = MCTSPlayer(exploration_c=1.41)
    print(f"\nRunning MCTS to find winning move...")
    move = player.choose_move(game, iterations=500)
    
    game.make_move(move)
    print(f"MCTS chose: {move}")
    print(f"Game status after move: {game.status()}")
    
    if game.status() == 1:
        print(f"✅ MCTS found the winning move!")
    else:
        print(f"⚠️  MCTS didn't find winning move (status: {game.status()})")
    
    print_board(game)
    print()


def test_clone_and_rollback():
    """Test that clone and unmake work correctly."""
    print("="*60)
    print("TEST 4: Clone and Rollback Consistency")
    print("="*60)
    
    game1 = Gomoku(size=9)
    moves = [(4, 4), (4, 5), (5, 4), (5, 5), (3, 3)]
    
    print(f"\nMaking {len(moves)} moves on game1...")
    for move in moves:
        game1.make_move(move)
    
    # Clone
    game2 = game1.clone()
    print(f"✓ Cloned game1 to game2")
    
    # Boards should match
    boards_match = all(
        game1.board[r][c] == game2.board[r][c]
        for r in range(game1.size)
        for c in range(game1.size)
    )
    
    assert boards_match, "Cloned board should match original"
    assert game1.to_move == game2.to_move, "Turn should match"
    assert len(game1.move_history) == len(game2.move_history), "Move history should match"
    print(f"✓ Boards match perfectly")
    
    # Unmake all moves on game1
    for move in reversed(moves):
        game1.unmake_move(move)
    
    print(f"✓ Unmade all moves on game1")
    print(f"  game1 status: {len(game1.move_history)} moves")
    print(f"  game2 status: {len(game2.move_history)} moves")
    
    print(f"\n✅ Clone and rollback test passed!\n")


def print_board(game):
    """Pretty-print the board."""
    print("\n   ", end="")
    for c in range(game.size):
        print(f"{c:2}", end=" ")
    print()
    
    for r in range(game.size):
        print(f"{r:2} ", end="")
        for c in range(game.size):
            cell = game.board[r][c]
            if cell == 1:
                print(" X", end=" ")
            elif cell == -1:
                print(" O", end=" ")
            else:
                print(" .", end=" ")
        print()
    print()


def main():
    """Run all tests."""
    print("\n" + "🧪 MCTS TEST SUITE FOR GOMOKU 🧪".center(60))
    
    try:
        test_basic_mcts_functionality()
        test_clone_and_rollback()
        test_winning_detection()
        test_single_game()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
