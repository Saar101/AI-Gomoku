"""
PUCTPlayer - MCTS with Neural Network Guidance
Uses PUCT (Predictor + UCT) algorithm instead of random rollouts
"""

import random
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

        # Store policy for expansion
        self._policy_cache = policy_dict

        # Debug: show top policy moves from the network
        print(f"PUCTPlayer: Initial value={value:.4f}")
        if policy_dict:
            top_policy = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            top_policy_fmt = {str(move): round(prob, 4) for move, prob in top_policy}
            print(f"   Top 5 policy moves (network): {top_policy_fmt}")
        
        # Run simulations
        simulation_count = 0
        for sim in range(self.num_simulations):
            # Clone game for simulation
            game_copy = game.clone()
            self._simulate(root, game_copy)
            simulation_count += 1
            if (sim + 1) % 20 == 0:
                print(f"   Simulation {sim + 1}/{self.num_simulations} complete")
        
        print(f"   All {simulation_count} simulations complete")
        
        # Select move based on visit counts
        visit_dist = root.get_visit_distribution()
        top_moves = dict(sorted([(str(m), v) for m, v in visit_dist.items()], key=lambda x: x[1], reverse=True)[:5])
        top_q_moves = dict(sorted([(str(child.move), -child.Q) for child in root.children.values()], key=lambda x: x[1], reverse=True)[:5])
        print(f"   Top 5 moves: {top_moves}")
        print(f"   Top 5 Q-values: {top_q_moves}")
        
        if temperature == 0:
            # Deterministic: choose most visited
            best_child = root.best_child_visits()
            selected_move = best_child.move if best_child else legal_moves[0]
            print(f"   Selected (deterministic): {selected_move}")
            return selected_move
        else:
            # Stochastic: sample proportional to visits^(1/T)
            if not visit_dist:
                print(f"   No visit distribution, returning first legal move")
                return legal_moves[0]
            
            if temperature == 1.0:
                # Use visit counts directly as probabilities
                moves, probs = zip(*visit_dist.items())
                selected_move = random.choices(moves, weights=probs)[0]
                print(f"   Selected (temperature={temperature}): {selected_move}")
                return selected_move
            else:
                # Apply temperature
                temp_probs = {m: (p ** (1.0/temperature)) for m, p in visit_dist.items()}
                total = sum(temp_probs.values())
                temp_probs = {m: p/total for m, p in temp_probs.items()}
                moves, probs = zip(*temp_probs.items())
                selected_move = random.choices(moves, weights=probs)[0]
                print(f"   Selected (temperature={temperature}): {selected_move}")
                return selected_move
    
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
        
        while current.children:
            # Select best child according to PUCT
            current = current.best_child_puct(self.c_puct, game.to_move)
            path.append(current)
            
            # Apply move to game
            game.make_move(current.move)
        
        # Check if game is terminal
        status = game.status()
        
        if status is not None:
            # Terminal node: absolute scoring (black=+1, white=-1, draw=0)
            value = float(status)
        else:
            # 2. EXPANSION: add all children with priors if needed

            legal_moves = game.legal_moves()
            value, policy_dict = self.network.predict(game, legal_moves)

            current._policy_dict = policy_dict
            for move, prior_prob in policy_dict.items():
                current.add_child(move, prior_prob)


        # 3. BACKPROPAGATION: update all nodes in path
        self._backpropagate(path, value)
    
    def _backpropagate(self, path, value):
        """
        Backpropagate value up the tree
        
        Args:
            path: list of nodes from root to leaf
            value: absolute value (+1 black, 0 draw, -1 white)
        """
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.update(value)
    
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

    def _select_untried_move(self, node):
        """Select an untried move with prior-weighted sampling."""
        if not node.untried_moves:
            return None

        policy = getattr(node, "_policy_dict", {})
        moves = list(node.untried_moves)
        weights = [policy.get(m, 0.0) for m in moves]

        if sum(weights) == 0:
            move = random.choice(moves)
        else:
            move = random.choices(moves, weights=weights)[0]

        node.untried_moves.remove(move)
        return move
