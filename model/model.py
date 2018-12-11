import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, strides=(1, 1, 1), padding=(1, 1, 1), downsample=False, activation=nn.ReLU, normalization=nn.BatchNorm3d, bias=False):
        super().__init__()
        self.downsample = downsample
        
        self.out_channels = out_channels
        if self.downsample:
            self.d1 = nn.Conv3d(in_channels, out_channels, 1, stride=(1, 2, 2), padding=padding, bias=False)
            self.dbn1 = nn.BatchNorm3d(out_channels)
            in_channels = out_channels
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel, strides, padding, bias=bias)
        self.bn1 = normalization(out_channels)
        self.activation = activation(inplace=True)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel, (1, 1, 1), padding, bias=bias)
        self.bn2 = normalization(out_channels)
    
    def forward(self, x):
        residual = x
        x = self.c1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.c2(x)
        x = self.bn2(x)
        print("SHAPE", x.size(), residual.size())
        if self.downsample:
            residual = self.dbn1(self.d1(x))
        x += residual
        x = self.activation(x)

        return x

class ResNet3D(BaseModel):
    def __init__(self, num_classes=26):
        super().__init__()
        layers = [3, 4, 6, 3]
        self.network = nn.ModuleList()
        # First conv
        self.network.append(nn.Conv3d(3, 64, 7, (1, 2, 2), (3, 3, 3)))
        self.network.append(nn.BatchNorm3d(64))
        self.network.append(nn.ReLU())
        self.network.append(nn.MaxPool3d(3, 2, 1))
        input = 64
        # group 1 is missing a block without this
        self.network.append(ResNetBlock(input, input, 3, (1, 1, 1)))
        # add other layers
        for i, group in enumerate(layers):
            section = nn.ModuleList()
            for _ in range(1, group):
                section.append(ResNetBlock(input, input))
            # don't upsample on last layer
            if i != len(layers)-1:
                section.append(ResNetBlock(input, input*2, strides=(2, 2, 2)))
                input *= 2
            self.network.extend(section)
        
        out_features = self.network[-1].out_channels
        last_section = nn.ModuleList()
        last_section.append(nn.AvgPool3d((1, 4, 4), stride=1))
        last_section.append(nn.Linear(out_features, num_classes))
        self.network.extend(last_section)

    def forward(self, x):
        for i, m in enumerate(self.network):
            print("Layer: ", i, self.network[i])
            x = m(x)
        return x


class LSTM(BaseModel):
    def __init__(self, num_classes=26):
        super().__init__()
        self.l1 = nn.LSTM(3, 512, 5)

        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
    