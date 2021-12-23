from torch import nn
import torch.nn.functional as F
import math


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 16, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(16, num_channels, kernel_size=3, padding=3 // 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv_block = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2) for _ in range(8)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(0.5 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.leaky_relu(self.conv1(x))
        for conv in self.conv_block:
            h = self.leaky_relu(conv(h)+h)
        h = self.leaky_relu(self.conv2(h))
        output = self.conv3(h)
        return output
