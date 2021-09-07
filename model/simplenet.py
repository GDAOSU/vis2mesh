import torch
import torch.nn as nn

class Conv4(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Conv1(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Linear(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__(**kwargs)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == '__main__':
    from torchsummary import *
    summary(Conv1(input_channels=3),(3,64,64))