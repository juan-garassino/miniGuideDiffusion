from miniGuideDiffusion.residuals import ResidualConvBlock
import torch
import torch.nn as nn

class UnetDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        self.model = nn.Sequential(
            *[ResidualConvBlock(in_channels, out_channels),
              nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
