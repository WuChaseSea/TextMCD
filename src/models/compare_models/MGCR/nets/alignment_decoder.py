import torch
from torch import nn
import torch.nn.functional as F

# Bi-temporal RS Image fusion
class BIFusion(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()

        self.in_channel = in_channel_low + in_channel_high
        self.out_channel = out_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, diff, cat):
        fusion = torch.cat([diff, cat], dim=1)
        fusion = self.conv(fusion)
        fusion = self.dropout(fusion)
        return fusion

# Up sampling
class Decoder(nn.Module):
    def __init__(self, in_channel_high, in_channel_low, out_channel):
        super().__init__()
        self.in_channel = in_channel_low + in_channel_high
        self.out_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(0.1)
    def forward(self, low, high):
        low = F.interpolate(low, scale_factor=2, mode='bilinear')
        fusion = torch.cat([low, high], dim=1)
        out = self.conv(fusion)
        out = self.dropout(out)
        return out


