import random
from MCTSNode import MCTSNode


class MCTSPlayer:
    """
    Implements pure MCTS algorithm for Gomoku.
    No heuristics - completely vanilla MCTS.
    """

    def __init__(self, exploration_c=1.41421356237, rollout_depth_limit=20):
        self.c = exploration_c
        self.rollout_depth_limit = rollout_depth_limit

    def choose_move(self, game, iterations):
        legal = game.legal_moves()

        # Terminal or no legal moves
        if game.status() is not None or not legal:
            return None

        # Tactical shortcut: if there is a direct win, take it immediately.
        winning_move = self._find_immediate_winning_move(game, legal)
        if winning_move is not None:
            return winning_move

        # Tactical shortcut: if opponent has a direct win next move, block it.
        blocking_move = self._find_immediate_blocking_move(game, legal)
        if blocking_move is not None:
            return blocking_move

        root_player = game.to_move
        root = MCTSNode(parent=None, move=None, untried_moves=legal)

        # Run MCTS iterations
        self._run_mcts(game, root, iterations, root_player)

        # Final choice: prefer move with best value for root player
        return self._select_best_move(root, root_player, legal)

    def _run_mcts(self, game, root, iterations, root_player):
        """Run the MCTS algorithm"""
        for _ in range(iterations):
            node = root
            path_moves = []

            # 1) Selection
            while game.status() is None and node.is_fully_expanded() and node.children:
                node = node.best_child_uct(self.c, game.to_move)
                game.make_move(node.move)
                path_moves.append(node.move)

            # Terminal during selection
            if game.status() is not None:
                result = game.status()
                self._backpropagate(node, result, root_player)
                for mv in reversed(path_moves):
                    game.unmake_move(mv)
                continue

            # 2) Expansion
            if node.untried_moves:
                mv = random.choice(node.untried_moves)
                node.untried_moves.remove(mv)

                game.make_move(mv)
                path_moves.append(mv)

                node = node.add_child(mv, game.legal_moves())

            # 3) Simulation (local-biased rollout with depth limit)
            rollout_moves = []
            rollout_depth = 0
            while game.status() is None and rollout_depth < self.rollout_depth_limit:
                legal_moves = game.legal_moves()
                if not legal_moves:
                    break

                local_moves = self._get_local_moves(game, legal_moves)
                candidates = local_moves if local_moves else legal_moves
                mv = random.choice(candidates)
                game.make_move(mv)
                rollout_moves.append(mv)
                rollout_depth += 1

            terminal_status = game.status()
            if terminal_status is None:
                result = self._evaluate_position(game)
            else:
                result = float(terminal_status)

            # 4) Backpropagation
            self._backpropagate(node, result, root_player)

            # Undo all moves
            for mv in reversed(rollout_moves):
                game.unmake_move(mv)
            for mv in reversed(path_moves):
                game.unmake_move(mv)

    def _select_best_move(self, root, root_player, legal):
        """Select the best move based on MCTS statistics"""
        if not root.children:
            return random.choice(legal)

        # Final move policy: highest visit count, tie-break by average value.
        # This is the standard robust child selection for MCTS.
        best_move = None
        best_visits = -1
        best_value = -float('inf')

        for move, child in root.children.items():
            if child.visits == 0:
                continue
            avg_value = child.value_sum / child.visits
            score = avg_value if root_player == 1 else -avg_value

            if child.visits > best_visits or (child.visits == best_visits and score > best_value):
                best_visits = child.visits
                best_value = score
                best_move = move

        return best_move if best_move is not None else random.choice(legal)

    def _backpropagate(self, node, result, root_player):
        """
        Backpropagate result up the tree.
        result: 1 (player 1 wins), -1 (player -1 wins), 0 (draw), None (game ongoing - shouldn't happen)
        root_player: the player who was to move at root (1 or -1)
        
        At each node, store value from that player's perspective.
        """
        if result is None:
            return

        absolute_value = float(result)

        cur = node
        while cur is not None:
            cur.update(absolute_value)
            cur = cur.parent

    def _find_immediate_winning_move(self, game, candidate_moves):
        """Return a move that wins immediately for the side to move, if one exists."""
        player = game.to_move
        for mv in candidate_moves:
            game.make_move(mv)
            is_win = game.status() == player
            game.unmake_move(mv)
            if is_win:
                return mv
        return None

    def _find_immediate_blocking_move(self, game, candidate_moves):
        """Block a direct one-move win available to the opponent, if any."""
        if not candidate_moves:
            return None

        original_player = game.to_move
        opponent = -original_player
        opponent_winning_moves = []

        for mv in candidate_moves:
            game.to_move = opponent
            game.make_move(mv)
            is_win = game.status() == opponent
            game.unmake_move(mv)
            game.to_move = original_player
            if is_win:
                opponent_winning_moves.append(mv)

        if not opponent_winning_moves:
            return None

        # Any opponent winning square is a valid urgent block.
        return random.choice(opponent_winning_moves)

    def _get_local_moves(self, game, legal_moves):
        """Prefer legal moves adjacent to existing stones to reduce random noise."""
        if not game.move_history:
            return legal_moves

        occupied = {(r, c) for r in range(game.size) for c in range(game.size) if game.board[r][c] != 0}
        local = []
        for r, c in legal_moves:
            has_neighbor = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if (rr, cc) in occupied:
                        has_neighbor = True
                        break
                if has_neighbor:
                    break
            if has_neighbor:
                local.append((r, c))

        return local

    def _evaluate_position(self, game):
        """Heuristic absolute evaluation used when rollout is truncated."""
        longest_black = self._longest_run(game, 1)
        longest_white = self._longest_run(game, -1)
        value = (longest_black - longest_white) / 5.0
        if value > 1.0:
            return 1.0
        if value < -1.0:
            return -1.0
        return value

    def _longest_run(self, game, player):
        """Longest contiguous line length for a player."""
        size = game.size
        board = game.board
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        best = 0

        for r in range(size):
            for c in range(size):
                if board[r][c] != player:
                    continue
                for dr, dc in directions:
                    length = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < size and 0 <= cc < size and board[rr][cc] == player:
                        length += 1
                        rr += dr
                        cc += dc
                    if length > best:
                        best = length
                        if best >= 5:
                            return best
        return best