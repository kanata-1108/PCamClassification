import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:
            self.identity = nn.Identity()
        else:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        idnetity = self.identity(x)
        out = self.layer(x)
        out += idnetity
        out = self.ReLU(out)

        return out

class My_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.25)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(in_channels = 16, out_channels = 16),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(in_channels = 32, out_channels = 32),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.25)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(in_channels = 64, out_channels = 64),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.25),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)
        x = self.linear_layer(x)
        x = self.Sigmoid(x)

        return x