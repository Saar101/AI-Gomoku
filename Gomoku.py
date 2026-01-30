from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


Move = Tuple[int, int]


@dataclass
class Gomoku:
	size: int = 15

	def __post_init__(self) -> None:
		self.make_init()

	def make_init(self) -> None:
		"""Start a new game."""
		self.board: List[List[int]] = [[0 for _ in range(self.size)] for _ in range(self.size)]
		self.to_move: int = 1
		self.move_history: List[Move] = []
		self._status: Optional[int] = None

	def make_move(self, move: Move) -> None:
		"""Apply a move to the current game state."""
		if self._status is not None:
			raise ValueError("Game is already over.")
		row, col = move
		if not self._is_within(row, col):
			raise ValueError("Move out of bounds.")
		if self.board[row][col] != 0:
			raise ValueError("Cell is not empty.")

		player = self.to_move
		self.board[row][col] = player
		self.move_history.append(move)

		if self._check_five_from(row, col, player):
			self._status = player
		elif self._board_full():
			self._status = 0
		else:
			self._status = None

		self.to_move *= -1

	def unmake_move(self, move: Move) -> None:
		"""Reverse a move."""
		if not self.move_history:
			raise ValueError("No moves to unmake.")
		last_move = self.move_history.pop()
		if last_move != move:
			self.move_history.append(last_move)
			raise ValueError("Move to unmake is not the last move.")

		row, col = move
		if not self._is_within(row, col):
			raise ValueError("Move out of bounds.")
		if self.board[row][col] == 0:
			raise ValueError("Cell is already empty.")

		self.board[row][col] = 0
		self.to_move *= -1
		self._status = self._compute_status_full()

	def clone(self) -> "Gomoku":
		"""Create a deep copy of the current game state."""
		new_game = Gomoku(self.size)
		new_game.board = [row[:] for row in self.board]
		new_game.to_move = self.to_move
		new_game.move_history = list(self.move_history)
		new_game._status = self._status
		return new_game

	def encode(self) -> List[int]:
		"""Encode the game state as a binary vector."""
		current = self.to_move
		opponent = -self.to_move
		current_plane: List[int] = []
		opponent_plane: List[int] = []
		for r in range(self.size):
			for c in range(self.size):
				cell = self.board[r][c]
				current_plane.append(1 if cell == current else 0)
				opponent_plane.append(1 if cell == opponent else 0)
		turn_plane = [1 if self.to_move == 1 else 0]
		return current_plane + opponent_plane + turn_plane

	def decode(self, action_index: int) -> Move:
		"""Translate an action index into a move in the game."""
		if action_index < 0 or action_index >= self.size * self.size:
			raise ValueError("Action index out of bounds.")
		row = action_index // self.size
		col = action_index % self.size
		return row, col

	def status(self) -> Optional[int]:
		"""Return game result if over, or None if ongoing."""
		return self._status

	def legal_moves(self) -> List[Move]:
		"""Return list of legal moves in current position."""
		if self._status is not None:
			return []
		moves: List[Move] = []
		for r in range(self.size):
			for c in range(self.size):
				if self.board[r][c] == 0:
					moves.append((r, c))
		return moves

	def _is_within(self, row: int, col: int) -> bool:
		return 0 <= row < self.size and 0 <= col < self.size

	def _board_full(self) -> bool:
		for row in self.board:
			for cell in row:
				if cell == 0:
					return False
		return True

	def _check_five_from(self, row: int, col: int, player: int) -> bool:
		directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
		for dr, dc in directions:
			count = 1
			count += self._count_in_direction(row, col, dr, dc, player)
			count += self._count_in_direction(row, col, -dr, -dc, player)
			if count >= 5:
				return True
		return False

	def _count_in_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
		count = 0
		r, c = row + dr, col + dc
		while self._is_within(r, c) and self.board[r][c] == player:
			count += 1
			r += dr
			c += dc
		return count

	def _compute_status_full(self) -> Optional[int]:
		for r in range(self.size):
			for c in range(self.size):
				player = self.board[r][c]
				if player != 0 and self._check_five_from(r, c, player):
					return player
		if self._board_full():
			return 0
		return None


def print_board(game: Gomoku) -> None:
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


if __name__ == "__main__":
    game = Gomoku(size=15)
    print("Welcome to Gomoku!")
    print("Player 1 = X, Player 2 = O")
    print_board(game)
    
    while game.status() is None:
        player_name = "Player 1 (X)" if game.to_move == 1 else "Player 2 (O)"
        print(f"\n{player_name}'s turn")
        try:
            row = int(input("Enter row: "))
            col = int(input("Enter col: "))
            game.make_move((row, col))
            print_board(game)
        except (ValueError, IndexError) as e:
            print(f"Invalid move: {e}")
            continue
    
    result = game.status()
    if result == 1:
        print("\n🎉 Player 1 (X) wins!")
    elif result == -1:
        print("\n🎉 Player 2 (O) wins!")
    elif result == 0:
        print("\nDraw!")
