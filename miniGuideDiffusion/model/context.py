from miniGuideDiffusion.blocks.residuals import ResidualConvBlock
from miniGuideDiffusion.blocks.unet import UnetDown, UnetUp
from miniGuideDiffusion.layers.embedding import EmbedFC

import torch.nn as nn
import torch

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        """
        Contextual U-Net model for image processing tasks.

        Args:
        - in_channels (int): Number of input channels.
        - n_feat (int): Number of features.
        - n_classes (int): Number of classes in the dataset.

        """
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        # Initial convolutional block
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Down-sampling blocks
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # Global pooling layer
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        # Embedding layers for time and context
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        # Up-sampling blocks
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        """
        Forward pass of the ContextUnet model.

        Args:
        - x (torch.Tensor): Input image tensor.
        - c (torch.Tensor): Context label tensor.
        - t (torch.Tensor): Time step tensor.
        - context_mask (torch.Tensor): Tensor indicating which samples to block the context on.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Initial convolutional block
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # Convert context to one-hot encoding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # Mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = -1 * (1 - context_mask)
        c = c * context_mask

        # Embed context and time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Up-sampling blocks
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)

        # Output layer
        out = self.out(torch.cat((up3, x), 1))
        return out
