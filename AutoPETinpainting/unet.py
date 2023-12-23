""" Full assembly of the parts to form the complete network """

from unet_parts import *

'''
Implementation of Unet-like archottecture, which replace all the normal convolution with gated convolution, the implementation follows.

'''


class GatedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(GatedUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.down1 = Down(in_channels=self.in_channels, out_channels=64, kernal_size=7, padding=3,  batch_norm=True)
        self.down2 = Down(in_channels=64, out_channels=128, kernal_size=5, padding=2, batch_norm=True)
        self.down3 = Down(in_channels=128, out_channels=256, kernal_size=3, padding=1, batch_norm=True)
        self.down4 = Down(in_channels=256, out_channels=512, kernal_size=3, padding=1, batch_norm=True)
        self.down5 = Down(in_channels=512, out_channels=512, kernal_size=3, padding=1, batch_norm=True)
        self.down6 = Down(in_channels=512, out_channels=512, kernal_size=3, padding=1, batch_norm=True)

        self.bottom = Down(in_channels=512, out_channels=512, kernal_size=3, padding=1, batch_norm=True)

  
        self.up1 = Up(in_channels=1024, out_channels=512, kernal_size=3, padding=1, batch_norm=True)
        self.up2 = Up(in_channels=1024, out_channels=512, kernal_size=3, padding=1, batch_norm=True)
        self.up3 = Up(in_channels=1024, out_channels=512, kernal_size=3, padding=1, batch_norm=True)
        self.up4 = Up(in_channels=768, out_channels=256, kernal_size=3, padding=1, batch_norm=True)
        self.up5 = Up(in_channels=384, out_channels=128, kernal_size=3, padding=1, batch_norm=True)
        self.up6 = Up(in_channels=192, out_channels=64, kernal_size=3, padding=1, batch_norm=True)

        self.conv_out = OutConv(in_channels=self.in_channels+64, out_channels=self.out_channels)

    def forward(self, x_in):
        x1 = self.down1(x_in)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        xb = self.bottom(x6)

        x = self.up1(xb, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)


        logits = self.conv_out(x, x_in)

        return logits
