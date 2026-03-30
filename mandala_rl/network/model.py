"""
Policy-Value neural network for Mandala.

Architecture: ResNet-style with shared trunk + policy/value heads.
Supports optional phase-aware routing and factored action policy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..game.state import GameState

# Dominion action space layout
_DOM_PLAY_OFFSET = 3
_DOM_BUY_OFFSET = 34
_DOM_SELECT_OFFSET = 65
_DOM_GAIN_OFFSET = 96
_DOM_NUM_CARD_TYPES = 31
_DOM_CONTROL_INDICES = [0, 1, 2, 127, 128, 129, 130]  # END_ACTIONS, END_BUYS, DONE_SELECTING, REVEAL_MOAT, DECLINE_REACTION, ACCEPT, DECLINE

# Phase values encoded in channel 129 of the tensor
_PHASE_ACTION = 0.0
_PHASE_BUY = 0.33
_PHASE_SELECT = 0.67
_PHASE_REACT = 1.0
_PHASE_VALS = [_PHASE_ACTION, _PHASE_BUY, _PHASE_SELECT, _PHASE_REACT]
_PHASE_NAMES = ['action', 'buy', 'select', 'react']


class ResBlock(nn.Module):
    """Residual block with batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class FactoredPolicyHead(nn.Module):
    """Factored policy: control + type + card logits → 131 actions."""

    def __init__(self, in_features: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.fc_control = nn.Linear(in_features, len(_DOM_CONTROL_INDICES))
        self.fc_type = nn.Linear(in_features, 4)  # PLAY, BUY, SELECT, GAIN
        self.fc_card = nn.Linear(in_features, _DOM_NUM_CARD_TYPES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B = features.size(0)
        control = self.fc_control(features)  # (B, 7)
        type_logits = self.fc_type(features)  # (B, 4)
        card_logits = self.fc_card(features)  # (B, 31)

        policy = torch.zeros(B, self.num_actions, device=features.device, dtype=features.dtype)

        # Control actions at fixed positions
        for i, idx in enumerate(_DOM_CONTROL_INDICES):
            policy[:, idx] = control[:, i]

        # Card-based actions: type + card (broadcasting)
        for t, offset in enumerate([_DOM_PLAY_OFFSET, _DOM_BUY_OFFSET, _DOM_SELECT_OFFSET, _DOM_GAIN_OFFSET]):
            policy[:, offset:offset + _DOM_NUM_CARD_TYPES] = type_logits[:, t:t + 1] + card_logits

        return policy


class MandalaNet(nn.Module):
    """
    Policy-Value network for Mandala.

    Input: State tensor (channels x height x width)
    Output:
        - Policy: Probability distribution over actions
        - Value: Expected outcome for current player [-1, 1]
    """

    def __init__(
        self,
        input_channels: int = 50,
        num_actions: int = 256,
        num_res_blocks: int = 10,
        channels: int = 128,
        belief_size: int = 12,
        phase_aware_policy: bool = False,
        factored_policy: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.phase_aware = phase_aware_policy
        self.factored = factored_policy

        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])

        # Policy head (shared conv)
        self.conv_policy = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        policy_in = 32 * 8 * 8

        if phase_aware_policy and factored_policy:
            # 4 phase-routed factored heads
            self.phase_policy_heads = nn.ModuleDict({
                name: FactoredPolicyHead(policy_in, num_actions)
                for name in _PHASE_NAMES
            })
        elif phase_aware_policy:
            # 4 phase-routed flat FCs
            self.phase_policy_heads = nn.ModuleDict({
                name: nn.Linear(policy_in, num_actions)
                for name in _PHASE_NAMES
            })
        elif factored_policy:
            # Single factored head
            self.fc_policy = FactoredPolicyHead(policy_in, num_actions)
        else:
            # Original single flat FC
            self.fc_policy = nn.Linear(policy_in, num_actions)

        # Value head
        self.conv_value = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_value2 = nn.Linear(128, 1)

        # Score head (auxiliary: predicts normalized score margin, no activation)
        self.conv_score = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_score = nn.BatchNorm2d(32)
        self.fc_score1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_score2 = nn.Linear(128, 1)

        # Belief head (predicts opponent hand/cup contents)
        self.conv_belief = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_belief = nn.BatchNorm2d(32)
        self.fc_belief = nn.Linear(32 * 8 * 8, belief_size)

    def _compute_policy(self, policy_features: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        """Compute policy logits with optional phase routing and factoring."""
        if self.phase_aware:
            B = policy_features.size(0)
            phase_vals = x_input[:, 129, 0, 0]  # (B,)
            policy = torch.zeros(B, self.num_actions, device=policy_features.device, dtype=policy_features.dtype)

            for phase_val, name in zip(_PHASE_VALS, _PHASE_NAMES):
                mask = (phase_vals - phase_val).abs() < 0.1
                if mask.any():
                    head = self.phase_policy_heads[name]
                    policy[mask] = head(policy_features[mask])

            # Safety: if any example didn't match a phase, use action head as fallback
            unmatched = policy.sum(dim=1) == 0
            if unmatched.any():
                head = self.phase_policy_heads['action']
                policy[unmatched] = head(policy_features[unmatched])

            return policy
        else:
            return self.fc_policy(policy_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch, channels, height, width)

        Returns:
            (policy_logits, value, score, belief) tuple
                - policy_logits: (batch, num_actions)
                - value: (batch, 1) in range [-1, 1]
                - score: (batch, 1) unbounded score margin prediction
                - belief: (batch, belief_size) opponent hand/cup predictions (logits)
        """
        x_input = x  # Save for phase routing

        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy_features = F.relu(self.bn_policy(self.conv_policy(x)))
        policy_features = policy_features.view(policy_features.size(0), -1)
        policy = self._compute_policy(policy_features, x_input)

        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        # Score head (auxiliary: normalized score margin)
        score = F.relu(self.bn_score(self.conv_score(x)))
        score = score.view(score.size(0), -1)
        score = F.relu(self.fc_score1(score))
        score = self.fc_score2(score)

        # Belief head (opponent hand/cup color predictions)
        belief = F.relu(self.bn_belief(self.conv_belief(x)))
        belief = belief.view(belief.size(0), -1)
        belief = self.fc_belief(belief)

        return policy, value, score, belief

    def predict(self, state: GameState) -> tuple[np.ndarray, float]:
        """
        Predict policy and value for a single state.

        Args:
            state: Game state

        Returns:
            (policy, value) tuple
                - policy: numpy array of action probabilities
                - value: scalar value in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0)

            # Move to device
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)

            # Forward pass
            policy_logits, value, _score, _belief = self.forward(state_tensor)

            # Convert to numpy
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]

        return policy, float(value)

    def predict_batch(self, states: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict policy and value for a batch of states.

        Args:
            states: List of GameState objects

        Returns:
            (policies, values) tuple
                - policies: numpy array of shape (batch, num_actions)
                - values: numpy array of shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            tensors = np.stack([s.to_tensor() for s in states]).astype(np.float32)
            device = next(self.parameters()).device
            state_batch = torch.from_numpy(tensors).to(device)
            policy_logits, values, _scores, _beliefs = self.forward(state_batch)
            policies = F.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()[:, 0]
        return policies, values.astype(np.float64)

    def get_loss(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
        target_scores: torch.Tensor = None,
        target_beliefs: torch.Tensor = None,
        entropy_weight: float = 0.0,
        policy_weight: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.

        L = policy_weight * policy_loss + value_loss + 1.0 * score_loss
            + 0.5 * belief_loss - entropy_weight * entropy

        Args:
            states: Batch of state tensors
            target_policies: Target policy distributions from MCTS
            target_values: Target values (game outcomes)
            target_scores: Target score margins (normalized), optional
            target_beliefs: Target belief labels (binary), optional
            entropy_weight: Weight for entropy regularization (0 = disabled)
            policy_weight: Weight for policy loss (decays from 3.0 to 1.0)

        Returns:
            (total_loss, policy_loss, value_loss, score_loss, belief_loss) tuple
        """
        policy_logits, values, scores, belief_logits = self.forward(states)

        # Policy loss (cross-entropy)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), target_values)

        # Score loss (MSE on normalized score margin)
        if target_scores is not None:
            score_loss = F.mse_loss(scores.squeeze(), target_scores)
        else:
            score_loss = torch.tensor(0.0, device=states.device)

        # Belief loss (BCE on opponent hand/cup predictions)
        if target_beliefs is not None:
            belief_loss = F.binary_cross_entropy_with_logits(belief_logits, target_beliefs)
        else:
            belief_loss = torch.tensor(0.0, device=states.device)

        # Total loss
        total_loss = policy_weight * policy_loss + value_loss + 1.0 * score_loss + 0.5 * belief_loss

        # Entropy regularization: maximize entropy to prevent policy collapse
        if entropy_weight > 0:
            probs = F.softmax(policy_logits, dim=1)
            entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))
            total_loss = total_loss - entropy_weight * entropy

        return total_loss, policy_loss, value_loss, score_loss, belief_loss
