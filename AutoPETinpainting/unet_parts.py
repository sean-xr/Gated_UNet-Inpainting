""" Parts of the U-Net model """

import torch
import torch.nn as nn


class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=3):
        super(GatedConvolution, self).__init__()
        # Standard convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Gating convolution layer
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply standard convolution
        conv_out = self.conv(x)
        # Apply gating mechanism
        gate_out = self.sigmoid(self.gate_conv(x))
        # Multiply the convolutional output by the gating output
        gated_output = conv_out * gate_out

        return gated_output


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernal_size, padding, batch_norm=True):
        super().__init__()
        self.gated_conv = GatedConvolution(in_channels, out_channels, kernal_size, padding=padding)
        self.BN_layer = nn.BatchNorm2d(num_features=out_channels)
        self.batch_norm = batch_norm  # indicator to see whether bn is activated
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.gated_conv(x)
        if self.batch_norm:
            x1 = self.BN_layer(x1)
        x1 = self.relu(x1)
        return x1


class Up(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernal_size, padding, batch_norm=True):
        super().__init__()
        self.gated_conv = GatedConvolution(in_channels, out_channels, kernal_size, stride=1, padding=padding)
        self.batch_norm = batch_norm  # indicator to see whether bn is activated
        self.BN_layer = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.up_sampling = nn.UpsamplingNearest2d(scale_factor=2.)

    def forward(self, x1, x2):
        x1 = self.up_sampling(x1)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.gated_conv(x3)
        if self.batch_norm:
            x3 = self.BN_layer(x3)
        x3 = self.leaky_relu(x3)
        return x3


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.up_sampling = nn.UpsamplingNearest2d(scale_factor=2.)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        x1 = self.up_sampling(x1)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.conv(x3)
        x3 = self.sigmoid(x3)
        return x3
