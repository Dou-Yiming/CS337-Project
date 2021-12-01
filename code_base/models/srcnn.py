from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1,
                 kernel_sizes=[9, 5, 5],
                 filters=[64, 32]):
        super(SRCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, filters[0],
                      kernel_size=kernel_sizes[0],
                      padding=kernel_sizes[0]//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[1],
                      kernel_size=kernel_sizes[1],
                      padding=kernel_sizes[1]//2),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(filters[1], num_channels,
                      kernel_size=kernel_sizes[2],
                      padding=kernel_sizes[2]//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.conv(x)
        return self.head(h)
