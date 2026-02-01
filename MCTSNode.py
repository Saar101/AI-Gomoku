import math

class MCTSNode:
    def __init__(self, parent=None, move=None, untried_moves=None):
        self.parent = parent
        self.move = move
        self.children = {}
        self.untried_moves = list(untried_moves) if untried_moves is not None else []
        self.visits = 0
        self.value_sum = 0.0  # +1 RED, -1 YELLOW, 0 DRAW

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def add_child(self, move, untried_moves):
        child = MCTSNode(self, move, untried_moves)
        self.children[move] = child
        return child

    def update(self, result):
        self.visits += 1
        self.value_sum += result

    def uct_score(self, child, c):
        if child.visits == 0:
            return float("inf")
        # Each child stores results from its own perspective (1=win, 0=draw, -1=loss)
        # We want to pick the child with the HIGHEST value
        exploit = child.value_sum / child.visits
        explore = c * math.sqrt(math.log(max(1, self.visits)) / child.visits)
        return exploit + explore

    def best_child_uct(self, c):
        return max(self.children.values(), key=lambda ch: self.uct_score(ch, c))