from miniGuideDiffusion.blocks.residuals import ResidualConvBlock
import torch.nn as nn
import torch

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Down-sampling block for U-Net architecture.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.

        """
        super(UnetDown, self).__init__()
        # Layers for down-sampling
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the UnetDown block.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after down-sampling.

        """
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Up-sampling block for U-Net architecture.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.

        """
        super(UnetUp, self).__init__()
        # Layers for up-sampling
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        Forward pass of the UnetUp block.

        Args:
        - x (torch.Tensor): Input tensor.
        - skip (torch.Tensor): Skip connection tensor from the corresponding down-sampling block.

        Returns:
        - torch.Tensor: Output tensor after up-sampling and processing.

        """
        x = torch.cat((x, skip), 1)  # Concatenate skip connection with up-sampled tensor
        x = self.model(x)  # Process the concatenated tensor
        return x
