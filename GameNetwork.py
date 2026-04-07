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
    - Value head: absolute game value (+1 black win, 0 draw, -1 white win)
    - Policy head: probability distribution over all 225 possible moves

    Input encoding for CNN uses 3 planes of shape (3, N, N):
    - black stones plane
    - white stones plane
    - side-to-move plane

    Legacy flattened 3-plane inputs are still supported for compatibility.
    """
    
    def __init__(self, board_size=15, hidden_size=256):
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size
        
        # Shared CNN trunk for board processing
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Value head - predicts absolute value in [-1, 1]
        self.value_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(32 * self.num_moves, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        # Policy head - one logit per board cell
        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, encoded_state):
        """
        Forward pass
        
        Args:
            encoded_state:
                - single CNN state: (3, board_size, board_size)
                - batch of CNN states: (batch_size, 3, board_size, board_size)
                - flattened legacy states are accepted for backward compatibility
        
        Returns:
            value: tensor of shape (batch_size, 1) in [-1, 1]
            policy: tensor of shape (batch_size, num_moves) - raw logits
        """
        # Ensure input is float tensor
        if not isinstance(encoded_state, torch.Tensor):
            encoded_state = torch.tensor(encoded_state, dtype=torch.float32)
        elif encoded_state.dtype != torch.float32:
            encoded_state = encoded_state.float()
        
        encoded_state = self._normalize_to_cnn_input(encoded_state)

        # Shared representation
        shared_out = self.shared(encoded_state)

        # Value head
        value_features = self.value_conv(shared_out)
        value = self.value_head(value_features.flatten(start_dim=1))

        # Policy head
        policy_logits = self.policy_head(shared_out).flatten(start_dim=1)
        
        return value, policy_logits

    def _legacy_flat_to_planes(self, legacy_flat_state):
        """Convert one legacy flattened state into CNN 3-plane format."""
        n = self.board_size
        num_moves = self.num_moves
        black_plane = legacy_flat_state[:num_moves].reshape(n, n)
        white_plane = legacy_flat_state[num_moves:num_moves * 2].reshape(n, n)
        turn_plane = legacy_flat_state[num_moves * 2:].reshape(n, n)
        return torch.stack([black_plane, white_plane, turn_plane], dim=0)

    def _matrix_to_planes(self, board_matrix):
        """Convert one NxN signed board matrix into CNN 3-plane format."""
        black_plane = (board_matrix > 0).float()
        white_plane = (board_matrix < 0).float()
        turn_plane = torch.ones_like(black_plane)
        return torch.stack([black_plane, white_plane, turn_plane], dim=0)

    def _normalize_to_cnn_input(self, encoded_state):
        """Normalize supported input variants into (batch_size, 3, N, N)."""
        n = self.board_size
        num_moves = self.num_moves
        legacy_input_size = num_moves * 3

        if encoded_state.dim() == 1:
            if encoded_state.shape[0] != legacy_input_size:
                raise ValueError(f"Unexpected 1D input length {encoded_state.shape[0]}")
            planes = self._legacy_flat_to_planes(encoded_state)
            return planes.unsqueeze(0)

        if encoded_state.dim() == 2:
            if encoded_state.shape == (n, n):
                planes = self._matrix_to_planes(encoded_state)
                return planes.unsqueeze(0)
            if encoded_state.shape[1] == legacy_input_size:
                batch_planes = [self._legacy_flat_to_planes(state) for state in encoded_state]
                return torch.stack(batch_planes, dim=0)
            raise ValueError(f"Unexpected 2D input shape {tuple(encoded_state.shape)}")

        if encoded_state.dim() == 3:
            if encoded_state.shape[0] == 3 and encoded_state.shape[1:] == (n, n):
                return encoded_state.unsqueeze(0)
            if encoded_state.shape[1:] == (n, n):
                batch_planes = [self._matrix_to_planes(state) for state in encoded_state]
                return torch.stack(batch_planes, dim=0)
            raise ValueError(f"Unexpected 3D input shape {tuple(encoded_state.shape)}")

        if encoded_state.dim() == 4 and encoded_state.shape[1:] == (3, n, n):
            return encoded_state

        raise ValueError(f"Unexpected input shape {tuple(encoded_state.shape)}")
    
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
            print(f"ERROR: File not found: {filepath}")
            return False
        
        try:
            state_dict = torch.load(filepath, map_location=self.get_device())
            self.load_state_dict(state_dict)
            print(f"OK: Network loaded successfully from {filepath}")
            print(f"    Device: {self.get_device()}")
            
            # Verify weights are loaded
            total_params = sum(p.numel() for p in self.parameters())
            print(f"    Total parameters: {total_params}")
            
            return True
        except Exception as e:
            print(f"ERROR: Error loading network: {e}")
            return False
    
    def predict(self, game, legal_moves=None):
        """
        Predict value and policy for a game state
        
        Args:
            game: Gomoku game object
            legal_moves: list of legal moves (will use game.legal_moves() if None)
        
        Returns:
            value: float in [-1, 1] as absolute value (+black, -white)
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
        
        # Policy loss (soft-target cross entropy)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
        
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
