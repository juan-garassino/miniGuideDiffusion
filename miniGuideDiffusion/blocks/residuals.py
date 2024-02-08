import torch.nn as nn
import torch

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        """
        Standard ResNet style convolutional block.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - is_res (bool): Whether the block should be a residual block (default: False).

        """
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        # Convolutional layers with batch normalization and GELU activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResidualConvBlock.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.

        """
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # Adding the correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414  # Normalization factor for preserving variance
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
