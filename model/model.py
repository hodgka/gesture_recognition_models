import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, strides=(1, 1, 1), padding=(1, 1, 1), downsample=False, activation=nn.ReLU, normalization=nn.BatchNorm3d, bias=False):
        super().__init__()
        self.downsample = downsample
        
        self.out_channels = out_channels
        if self.downsample:
            self.d1 = nn.Conv3d(in_channels, out_channels, 1, stride=(2, 2, 2), padding=padding, bias=False)
            self.dbn1 = nn.BatchNorm3d(out_channels)
            in_channels = out_channels
        self.c1 = nn.Conv3d(in_channels, out_channels, kernel, strides, padding, bias=bias)
        self.bn1 = normalization(out_channels)
        self.activation = activation(inplace=True)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel, (1, 1, 1), padding, bias=bias)
        self.bn2 = normalization(out_channels)
    
    def forward(self, x):
        if self.downsample:
            x = self.dbn1(self.d1(x))
        input_shape = x.size()
        residual = x
        x = self.c1(x)
        x = self.bn1(x)
        x = self.activation(x)
        first_layer_shape = x.size()
        x = self.c2(x)
        x = self.bn2(x)
        # if self.downsample:
        #     residual = self.dbn1(self.d1(x))
        second_layer_shape = x.size()
        # print("SHAPES: ", input_shape, first_layer_shape, second_layer_shape)
        x += residual
        x = self.activation(x)
        # output_shape = x.size()
        

        return x

class ResNet3D(BaseModel):
    def __init__(self, num_classes=26, input_shape=(18, 176, 100)):
        super().__init__()
        #layers = [3, 4, 6, 3]
        layers = [2, 2, 2, 2]
        self.network = nn.ModuleList()
        self.last = nn.ModuleList()
        # First conv
        self.network.append(nn.Conv3d(3, 64, 7, (1, 2, 2), (3, 3, 3)))
        self.network.append(nn.BatchNorm3d(64))
        self.network.append(nn.ReLU())
        self.network.append(nn.MaxPool3d(3, 2, 1))
        self.last.append(nn.Linear(64*21*21*9, 10))
        input = 64
        # group 1 is missing a block without this
        self.network.append(ResNetBlock(input, input, 3, (1, 1, 1)))
        # # add other layers
        # for i, group in enumerate(layers):
        #     section = nn.ModuleList()
        #     for _ in range(1, group):
        #         section.append(ResNetBlock(input, input))
        #     # don't upsample on last layer
        #     if i != len(layers)-1:
        #         section.append(ResNetBlock(input, input*2, downsample=True))
        #         input *= 2
        #     self.network.extend(section)
        
        # self.network.append(nn.AvgPool3d((1, 4, 4), stride=1))
        # out_features = self.network[-1].out_channels
        # last_section = nn.ModuleList()
        # last_section.append(nn.Linear(6144, num_classes))
        # self.network.extend(last_section)

    def forward(self, x):
        for i, m in enumerate(self.network):
            input_size = x.size()
            x = m(x)
            print("Layer: ", i, self.network[i], input_size, x.size())
        x = x.view(10, -1)
        x = self.last[0](x)
        return x
