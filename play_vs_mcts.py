# play_vs_mcts_simple.py
# Simple 9x9 Gomoku vs MCTS.
# Input format: row,col  (example: 3,3)
# Prints the board after every move.

from Gomoku import Gomoku, print_board
from MCTSPlayer import MCTSPlayer


def parse_move(s: str, size: int):
    s = s.strip().lower()
    if s in {"q", "quit", "exit"}:
        return None

    # allow: "3,3"  or "3 3"  or "3 , 3"
    s = s.replace(" ", "")
    if "," in s:
        parts = s.split(",")
        if len(parts) != 2 or parts[0] == "" or parts[1] == "":
            raise ValueError("Bad format")
        r = int(parts[0])
        c = int(parts[1])
    else:
        # fallback: "3 3"
        parts = s.split()
        if len(parts) != 2:
            raise ValueError("Bad format")
        r = int(parts[0])
        c = int(parts[1])

    if not (0 <= r < size and 0 <= c < size):
        raise ValueError("Out of bounds")
    return (r, c)


def main():
    game = Gomoku(size=9)
    ai = MCTSPlayer()

    # You can tweak this number if AI is too weak/slow
    iterations = 10000

    # Human plays X (first). AI plays O (second).
    human_player = 1

    print("=== Gomoku 9x9 vs MCTS ===")
    print("You are X, AI is O")
    print("Enter moves as: row,col   (example: 3,3)")
    print("Type q to quit.\n")

    print_board(game)

    while game.status() is None:
        if game.to_move == human_player:
            # Human move
            while True:
                raw = input("Your move (row,col): ")
                try:
                    mv = parse_move(raw, game.size)
                    if mv is None:
                        print("Bye!")
                        return
                    game.make_move(mv)
                    break
                except Exception as e:
                    print(f"Invalid input/move. Try again. ({e})")

            print("\nBoard after your move:")
            print_board(game)

        else:
            # AI move
            mv = ai.choose_move(game, iterations)
            if mv is None:
                break
            game.make_move(mv)
            print(f"\nAI played: {mv[0]},{mv[1]}")
            print("Board after AI move:")
            print_board(game)

    res = game.status()
    if res == 1:
        print("\nGame Over: X wins!")
    elif res == -1:
        print("\nGame Over: O wins!")
    else:
        print("\nGame Over: Draw!")


if __name__ == "__main__":
    main()