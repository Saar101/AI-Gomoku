"""
PUCTNode - MCTS Node with Prior Probability from Neural Network
Enhanced version of MCTSNode for PUCT (Predictor + UCT) algorithm
"""

import math

class PUCTNode:
    def __init__(self, parent=None, move=None, prior_prob=0.0):
        """
        Initialize a PUCT node
        
        Args:
            parent: parent PUCTNode
            move: the move (row, col) that led to this state
            prior_prob: P(s,a) - prior probability from neural network policy
        """
        self.parent = parent
        self.move = move
        self.prior_prob = prior_prob  # P(s,a) from neural network
        
        self.children = {}  # dict: move -> PUCTNode
        self.untried_moves = []  # moves not yet expanded
        
        self.N = 0  # visit count
        self.W = 0.0  # total action value
        self.Q = 0.0  # mean action value (W/N)
    
    def is_fully_expanded(self):
        """Check if all legal moves have been tried"""
        return len(self.untried_moves) == 0
    
    def add_child(self, move, prior_prob):
        """
        Add a child node
        
        Args:
            move: the move (row, col) leading to this child
            prior_prob: P(s,a) from neural network
        
        Returns:
            The newly created child node
        """
        child = PUCTNode(parent=self, move=move, prior_prob=prior_prob)
        self.children[move] = child
        return child
    
    def update(self, value):
        """
        Update node statistics after a simulation
        
        Args:
            value: result from current node's perspective (+1 win, 0 draw, -1 loss)
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
    
    def puct_score(self, c_puct=1.0):
        """
        Calculate PUCT score for this node (called from parent)
        
        PUCT formula: U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: exploration constant (typically 1.0-5.0)
        
        Returns:
            PUCT score (higher is better)
        """
        if self.parent is None:
            return self.Q

        # Exploitation term: Q(s,a) - average value
        exploit = self.Q

        parent_visits = max(1, self.parent.N)
        if self.N == 0:
            return c_puct * self.prior_prob * math.sqrt(parent_visits)

        # Exploration term: c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
        explore = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.N)

        return exploit + explore
    
    def best_child_puct(self, c_puct=1.0):
        """
        Select child with highest PUCT score
        
        Args:
            c_puct: exploration constant
        
        Returns:
            Child node with highest PUCT score
        """
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.puct_score(c_puct))
    
    def best_child_visits(self):
        """
        Select child with most visits (for final move selection)
        
        Returns:
            Child node with highest visit count
        """
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda child: child.N)
    
    def get_visit_distribution(self):
        """
        Get normalized visit count distribution over all children
        
        Returns:
            dict: move -> normalized visit probability
        """
        if not self.children:
            return {}
        
        total_visits = sum(child.N for child in self.children.values())
        if total_visits == 0:
            return {}
        
        return {move: child.N / total_visits for move, child in self.children.items()}
    
    def __repr__(self):
        return (f"PUCTNode(move={self.move}, N={self.N}, Q={self.Q:.3f}, "
                f"P={self.prior_prob:.3f}, children={len(self.children)})")
