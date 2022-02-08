""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Layers for multi-scale loss compute
        self.flow1 = predict_flow(1024 // factor)
        self.flow2 = predict_flow(512 // factor)
        self.flow3 = predict_flow(256 // factor)
        self.flow4 = predict_flow(128 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        flow1 = self.flow1(x5)
        x = self.up1(x5, x4)
        flow2 = self.flow2(x)
        x = self.up2(x, x3)
        flow3 = self.flow3(x)
        x = self.up3(x, x2)
        flow4 = self.flow4(x)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, flow4, flow3, flow2, flow1

