import torch.nn as nn

import torch
import torch.nn as nn

# Original implementation from ConfidNet:
# https://github.com/valeoai/ConfidNet/blob/master/confidnet/models/small_convnet_svhn.py 
class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = act

    def forward(self, x):
      return self.act(self.bn(self.conv(x)))
