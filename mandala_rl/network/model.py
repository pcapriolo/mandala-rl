"""
Policy-Value neural network for Mandala.

Architecture: ResNet-style with shared trunk + policy/value heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..game.state import GameState


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


class MandalaNet(nn.Module):
    """
    Policy-Value network for Mandala.

    Input: State tensor (channels × height × width)
    Output:
        - Policy: Probability distribution over actions
        - Value: Expected outcome for current player [-1, 1]
    """

    def __init__(
        self,
        input_channels: int = 50,
        num_actions: int = 256,
        num_res_blocks: int = 10,
        channels: int = 128
    ):
        """
        Args:
            input_channels: Number of input planes in state tensor
            num_actions: Size of action space
            num_res_blocks: Number of residual blocks in trunk
            channels: Number of channels in residual blocks
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.conv_policy = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, num_actions)  # Assuming 8×8 board

        # Value head
        self.conv_value = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(32)
        self.fc_value1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_value2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch, channels, height, width)

        Returns:
            (policy_logits, value) tuple
                - policy_logits: (batch, num_actions)
                - value: (batch, 1) in range [-1, 1]
        """
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.bn_policy(self.conv_policy(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)

        # Value head
        value = F.relu(self.bn_value(self.conv_value(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value))

        return policy, value

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
            policy_logits, value = self.forward(state_tensor)

            # Convert to numpy
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0, 0]

        return policy, float(value)

    def get_loss(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute training loss.

        Args:
            states: Batch of state tensors
            target_policies: Target policy distributions from MCTS
            target_values: Target values (game outcomes)

        Returns:
            (total_loss, policy_loss, value_loss) tuple
        """
        policy_logits, values = self.forward(states)

        # Policy loss (cross-entropy)
        policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1))

        # Value loss (MSE)
        value_loss = F.mse_loss(values.squeeze(), target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        return total_loss, policy_loss, value_loss
