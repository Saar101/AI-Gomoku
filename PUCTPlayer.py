"""
PUCTPlayer - MCTS with Neural Network Guidance
Uses PUCT (Predictor + UCT) algorithm instead of random rollouts
"""

from PUCTNode import PUCTNode

class PUCTPlayer:
    def __init__(self, network, c_puct=1.0, num_simulations=800):
        """
        Initialize PUCT player
        
        Args:
            network: GameNetwork instance for evaluation
            c_puct: exploration constant (balance exploration/exploitation)
            num_simulations: number of MCTS simulations per move
        """
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def choose_move(self, game, temperature=1.0):
        """
        Choose a move using PUCT algorithm
        
        Args:
            game: Gomoku game instance
            temperature: controls randomness (0=deterministic, 1=proportional to visits)
        
        Returns:
            Selected move (row, col)
        """
        if game.status() is not None:
            return None
        
        legal_moves = game.legal_moves()
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Create root node
        root = PUCTNode()
        
        # Get initial policy from network
        value, policy_dict = self.network.predict(game, legal_moves)
        root.untried_moves = list(policy_dict.keys())
        
        # Store policy for expansion
        self._policy_cache = policy_dict
        
        # Run simulations
        for _ in range(self.num_simulations):
            # Clone game for simulation
            game_copy = game.clone()
            self._simulate(root, game_copy)
        
        # Select move based on visit counts
        if temperature == 0:
            # Deterministic: choose most visited
            best_child = root.best_child_visits()
            return best_child.move if best_child else legal_moves[0]
        else:
            # Stochastic: sample proportional to visits^(1/T)
            visit_dist = root.get_visit_distribution()
            if not visit_dist:
                return legal_moves[0]
            
            if temperature == 1.0:
                # Use visit counts directly as probabilities
                import random
                moves, probs = zip(*visit_dist.items())
                return random.choices(moves, weights=probs)[0]
            else:
                # Apply temperature
                import random
                temp_probs = {m: (p ** (1.0/temperature)) for m, p in visit_dist.items()}
                total = sum(temp_probs.values())
                temp_probs = {m: p/total for m, p in temp_probs.items()}
                moves, probs = zip(*temp_probs.items())
                return random.choices(moves, weights=probs)[0]
    
    def _simulate(self, node, game):
        """
        Run one PUCT simulation
        
        Steps:
        1. Selection: traverse tree using PUCT scores
        2. Expansion: add new node if not terminal
        3. Evaluation: use neural network instead of rollout
        4. Backpropagation: update all nodes in path
        
        Args:
            node: current PUCTNode
            game: current game state
        """
        # 1. SELECTION: traverse to leaf using PUCT
        path = [node]
        current = node
        
        while current.is_fully_expanded() and current.children:
            # Select best child according to PUCT
            current = current.best_child_puct(self.c_puct)
            path.append(current)
            
            # Apply move to game
            game.make_move(current.move)
        
        # Check if game is terminal
        status = game.status()
        
        if status is not None:
            # Terminal node: use actual game result
            # Convert result to current player's perspective
            if status == 0:
                value = 0.0  # Draw
            else:
                # status is winner (1 or -1)
                # If current player won, value = +1, else -1
                value = 1.0 if status == game.to_move else -1.0
        else:
            # 2. EXPANSION: add one child if possible
            if not current.is_fully_expanded():
                # Get legal moves and policy
                legal_moves = game.legal_moves()
                value, policy_dict = self.network.predict(game, legal_moves)
                
                # Set untried moves with their prior probabilities
                current.untried_moves = list(policy_dict.keys())
                current._policy_dict = policy_dict
                
                # Expand first untried move
                if current.untried_moves:
                    move = current.untried_moves.pop(0)
                    prior_prob = policy_dict[move]
                    
                    child = current.add_child(move, prior_prob)
                    path.append(child)
                    
                    # Apply move
                    game.make_move(move)
                    
                    # 3. EVALUATION: use neural network
                    value, _ = self.network.predict(game)
                    
                    # Value is from game.to_move perspective
                    # We need it from the child's perspective (who just moved)
                    # After the move, it's opponent's turn, so flip value
                    value = -value
            else:
                # Already expanded, use network evaluation
                value, _ = self.network.predict(game)
                # Value is from current player's perspective
        
        # 4. BACKPROPAGATION: update all nodes in path
        self._backpropagate(path, value)
    
    def _backpropagate(self, path, value):
        """
        Backpropagate value up the tree
        
        Args:
            path: list of nodes from root to leaf
            value: value from leaf node's perspective
        """
        # Value is from the last node's perspective
        # As we go up, alternate perspectives
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.update(value)
            value = -value  # Flip perspective for parent
    
    def get_action_probs(self, game, temperature=1.0):
        """
        Get policy (action probabilities) based on MCTS visit counts
        
        Args:
            game: Gomoku game instance
            temperature: controls randomness
        
        Returns:
            dict: move -> probability
        """
        if game.status() is not None:
            return {}
        
        legal_moves = game.legal_moves()
        if not legal_moves:
            return {}
        
        # Create root and run simulations
        root = PUCTNode()
        value, policy_dict = self.network.predict(game, legal_moves)
        root.untried_moves = list(policy_dict.keys())
        self._policy_cache = policy_dict
        
        for _ in range(self.num_simulations):
            game_copy = game.clone()
            self._simulate(root, game_copy)
        
        # Get visit distribution
        visit_dist = root.get_visit_distribution()
        
        if temperature == 0:
            # One-hot on most visited
            best_move = max(visit_dist.items(), key=lambda x: x[1])[0]
            return {best_move: 1.0}
        elif temperature == 1.0:
            return visit_dist
        else:
            # Apply temperature
            temp_probs = {m: (p ** (1.0/temperature)) for m, p in visit_dist.items()}
            total = sum(temp_probs.values())
            return {m: p/total for m, p in temp_probs.items()}
