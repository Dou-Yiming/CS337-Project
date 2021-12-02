from torch import nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, num_channels=1,
                 kernel_sizes=[9, 5, 3],
                 filters=[64, 32]):
        super(SRCNN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_channels, filters[0],
                      kernel_size=kernel_sizes[0],
                      padding=kernel_sizes[0]//2),
            nn.LeakyReLU(inplace=True),
        )
        self.backbone = nn.ModuleList(
            [nn.Conv2d(filters[0], filters[0],
                       kernel_size=kernel_sizes[1],
                       padding=kernel_sizes[1]//2)
             for _ in range(5)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(filters[0], filters[1],
                      kernel_size=kernel_sizes[2],
                      padding=kernel_sizes[2]//2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(filters[1], num_channels,
                      kernel_size=kernel_sizes[2],
                      padding=kernel_sizes[2]//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.conv0(x)
        for conv in self.backbone:
            h = F.leaky_relu_(h+conv(h))
        return self.head(h)
