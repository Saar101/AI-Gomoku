import random
from MCTSNode import MCTSNode


class MCTSPlayer:
    """
    Implements pure MCTS algorithm for Gomoku.
    No heuristics - completely vanilla MCTS.
    """

    def __init__(self, exploration_c=1.41421356237):
        self.c = exploration_c

    def choose_move(self, game, iterations):
        legal = game.legal_moves()

        # Terminal or no legal moves
        if game.status() is not None or not legal:
            return None

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
                node = node.best_child_uct(self.c)
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

            # 3) Simulation (random rollout)
            rollout_moves = []
            while game.status() is None:
                mv = random.choice(game.legal_moves())
                game.make_move(mv)
                rollout_moves.append(mv)

            result = game.status()

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

        # Children of root are moves made by root_player
        # Their values represent outcomes from root_player's perspective
        # So we want the child with the HIGHEST value
        
        best_move = None
        best_value = -float('inf')
        
        for move, child in root.children.items():
            if child.visits == 0:
                continue
            avg_value = child.value_sum / child.visits
            
            if avg_value > best_value:
                best_value = avg_value
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
        
        cur = node
        
        # Count depth to determine who made the leaf move
        depth = 0
        temp = cur
        while temp is not None and temp.parent is not None:
            depth += 1
            temp = temp.parent
        
        # Determine if root_player made the move at this leaf
        # If root_player=1 and depth is odd, player 1 made leaf move
        # If root_player=-1 and depth is even, player -1 made leaf move
        is_player1_at_leaf = (depth % 2 == 1) if root_player == 1 else (depth % 2 == 0)
        
        cur = node
        level_player_is_player1 = is_player1_at_leaf
        
        while cur is not None and cur.parent is not None:  # Skip root
            # What value does this player see?
            # If this player is player 1 and result=1: +1 (win)
            # If this player is player 1 and result=-1: -1 (loss)
            # If this player is player -1 and result=1: -1 (loss)
            # If this player is player -1 and result=-1: +1 (win)
            # If result=0 (draw): 0 for everyone
            
            if level_player_is_player1:
                perspective_value = result  # +1 if player1 wins, -1 if player-1 wins, 0 if draw
            else:
                perspective_value = -result  # flip for player -1
            
            cur.update(perspective_value)
            
            # Move to parent (which was made by the other player)
            level_player_is_player1 = not level_player_is_player1
            cur = cur.parent