"""
GameNetwork - Neural Network for Gomoku
Predicts game value and policy (move probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class GameNetwork(nn.Module):
    """
    Neural network for Gomoku that outputs:
    - Value head: probability that current player wins (-1 to +1)
    - Policy head: probability distribution over all 225 possible moves
    """
    
    def __init__(self, board_size=15, hidden_size=256):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size
        
        # Input size: 2 planes (black, white) + 1 (turn) = 3 channels flattened
        # Actually from Gomoku.encode(): 2 planes of board_size*board_size + 1 turn bit
        input_size = board_size * board_size * 2 + 1
        
        # Shared layers - process the board representation
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Value head - predicts win probability (-1 to +1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Policy head - predicts move probabilities (225 for 15x15 board)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_moves),
            # Softmax will be applied in forward for training
            # or during inference based on legal moves
        )
    
    def forward(self, encoded_state):
        """
        Forward pass
        
        Args:
            encoded_state: tensor of shape (batch_size, input_size) 
                          from Gomoku.encode()
        
        Returns:
            value: tensor of shape (batch_size, 1) in [-1, 1]
            policy: tensor of shape (batch_size, num_moves) - raw logits
        """
        # Ensure input is float tensor
        if not isinstance(encoded_state, torch.Tensor):
            encoded_state = torch.tensor(encoded_state, dtype=torch.float32)
        elif encoded_state.dtype != torch.float32:
            encoded_state = encoded_state.float()
        
        # Handle 1D input (single state)
        if encoded_state.dim() == 1:
            encoded_state = encoded_state.unsqueeze(0)
        
        # Shared representation
        shared_out = self.shared(encoded_state)
        
        # Value and policy heads
        value = self.value_head(shared_out)
        policy_logits = self.policy_head(shared_out)
        
        return value, policy_logits
    
    def forward_with_legal_moves(self, encoded_state, legal_moves_list=None):
        """
        Forward pass with masking illegal moves
        
        Args:
            encoded_state: tensor or list of tensors
            legal_moves_list: list of lists of legal moves
                             e.g., [[(0,0), (1,1), ...], ...]
        
        Returns:
            value: raw value outputs
            policy: policy logits with illegal moves masked to -inf
        """
        value, policy_logits = self.forward(encoded_state)
        
        # If no legal moves provided, return raw policy
        if legal_moves_list is None:
            return value, policy_logits
        
        # Mask illegal moves
        batch_size = policy_logits.shape[0]
        masked_policy = policy_logits.clone()
        
        for i in range(batch_size):
            # Set all moves to -inf initially
            masked_policy[i, :] = float('-inf')
            
            # Unmask legal moves
            if legal_moves_list[i]:
                for move in legal_moves_list[i]:
                    row, col = move
                    move_idx = row * self.board_size + col
                    masked_policy[i, move_idx] = policy_logits[i, move_idx]
        
        return value, masked_policy
    
    def get_device(self):
        """Get device where network is located"""
        return next(self.parameters()).device
    
    def to_device(self, device):
        """Move network to device"""
        return self.to(device)
    
    def save(self, filepath):
        """Save network weights to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        print(f"✓ Network saved to {filepath}")
    
    def load(self, filepath):
        """Load network weights from file"""
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            return False
        
        self.load_state_dict(torch.load(filepath, map_location=self.get_device()))
        print(f"✓ Network loaded from {filepath}")
        return True
    
    def predict(self, game, legal_moves=None):
        """
        Predict value and policy for a game state
        
        Args:
            game: Gomoku game object
            legal_moves: list of legal moves (will use game.legal_moves() if None)
        
        Returns:
            value: float in [-1, 1] (win probability from current player perspective)
            policy_dict: dict mapping moves to probabilities
        """
        self.eval()  # Set to evaluation mode (disables dropout)
        
        if legal_moves is None:
            legal_moves = game.legal_moves()
        
        # Encode game state
        encoded = game.encode()
        
        # Forward pass
        with torch.no_grad():
            encoded_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
            value, policy_logits = self.forward(encoded_tensor)
        
        value_scalar = value.squeeze().item()
        
        # Extract policy for legal moves
        policy_dict = {}
        if legal_moves:
            # Get softmax probabilities
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0)
            
            for move in legal_moves:
                row, col = move
                move_idx = row * game.size + col
                policy_dict[move] = policy_probs[move_idx].item()
            
            # Normalize to ensure they sum to 1 (for legal moves only)
            total = sum(policy_dict.values())
            if total > 0:
                policy_dict = {m: p/total for m, p in policy_dict.items()}
        
        return value_scalar, policy_dict


class GameNetworkOptimizer:
    """Helper class for training GameNetwork"""
    
    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self.device = network.get_device()
    
    def train_step(self, encoded_states, target_values, target_policies):
        """
        Single training step
        
        Args:
            encoded_states: list or tensor of encoded game states
            target_values: list or tensor of target values (-1 to 1)
            target_policies: list or tensor of target policy distributions
        
        Returns:
            loss_dict: dict with loss components
        """
        # Convert to tensors if needed
        if not isinstance(encoded_states, torch.Tensor):
            encoded_states = torch.tensor(encoded_states, dtype=torch.float32)
        if not isinstance(target_values, torch.Tensor):
            target_values = torch.tensor(target_values, dtype=torch.float32)
        if not isinstance(target_policies, torch.Tensor):
            target_policies = torch.tensor(target_policies, dtype=torch.float32)
        
        # Move to device
        encoded_states = encoded_states.to(self.device)
        target_values = target_values.to(self.device)
        target_policies = target_policies.to(self.device)
        
        # Forward pass
        values, policy_logits = self.network(encoded_states)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), target_values)
        
        # Policy loss (cross entropy)
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        
        # Combined loss
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item()
        }
