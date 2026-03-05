# MCTSNode.py
import math

class MCTSNode:
    def __init__(self, parent=None, move=None, untried_moves=None):
        self.parent = parent
        self.move = move
        self.children = {}
        self.untried_moves = list(untried_moves) if untried_moves is not None else []
        self.visits = 0
        self.value_sum = 0.0  # absolute value accumulation: black=+, white=-

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def add_child(self, move, untried_moves):
        child = MCTSNode(self, move, untried_moves)
        self.children[move] = child
        return child

    def update(self, result_value):
        self.visits += 1
        self.value_sum += result_value

    def uct_score(self, child, c, player_to_move):
        if child.visits == 0:
            return float("inf")
        exploit = child.value_sum / child.visits
        if player_to_move == -1:
            exploit = -exploit
        explore = c * math.sqrt(math.log(max(1, self.visits)) / child.visits)
        return exploit + explore

    def best_child_uct(self, c, player_to_move):
        return max(self.children.values(), key=lambda ch: self.uct_score(ch, c, player_to_move))