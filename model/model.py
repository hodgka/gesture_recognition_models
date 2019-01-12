import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math


# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel=3, strides=(1, 1, 1), padding=(1, 1, 1), downsample=False, activation=nn.ReLU, normalization=nn.BatchNorm3d, bias=False):
#         super().__init__()
#         self.downsample = downsample
        
#         self.out_channels = out_channels
#         if self.downsample:
#             self.d1 = nn.Conv3d(in_channels, out_channels, 1, stride=(2, 2, 2), padding=padding, bias=False)
#             self.dbn1 = nn.BatchNorm3d(out_channels)
#             in_channels = out_channels
#         self.c1 = nn.Conv3d(in_channels, out_channels, kernel, strides, padding, bias=bias)
#         self.bn1 = normalization(out_channels)
#         self.activation = activation(inplace=True)
#         self.c2 = nn.Conv3d(out_channels, out_channels, kernel, (1, 1, 1), padding, bias=bias)
#         self.bn2 = normalization(out_channels)
    
#     def forward(self, x):
#         if self.downsample:
#             x = self.dbn1(self.d1(x))
#         input_shape = x.size()
#         residual = x
#         x = self.c1(x)
#         x = self.bn1(x)
#         x = self.activation(x)
#         first_layer_shape = x.size()
#         x = self.c2(x)
#         x = self.bn2(x)
#         # if self.downsample:
#         #     residual = self.dbn1(self.d1(x))
#         second_layer_shape = x.size()
#         # print("SHAPES: ", input_shape, first_layer_shape, second_layer_shape)
#         x += residual
#         x = self.activation(x)
#         # output_shape = x.size()
        
#         return x


# class ResNet3D(BaseModel):
#     def __init__(self, batch_size=10, num_classes=27, input_shape=(18, 176, 100)):
#         super().__init__()
#         self.batch_size=batch_size
#         self.num_classes=num_classes
#         #layers = [3, 4, 6, 3]
#         layers = [2, 2, 2, 2]
#         self.network = nn.ModuleList()
#         self.last = nn.ModuleList()
#         # First conv
#         self.network.append(nn.Conv3d(3, 64, 7, (1, 2, 2), (3, 3, 3)))
#         self.network.append(nn.BatchNorm3d(64))
#         self.network.append(nn.ReLU())
#         self.network.append(nn.MaxPool3d(3, 2, 1))
#         # self.last.append(nn.Linear(64*21*21*9, num_classes))
        
#         input = 64
#         # group 1 is missing a block without this
#         self.network.append(ResNetBlock(input, input, 3, (1, 1, 1)))
#         # # add other layers
#         for i, group in enumerate(layers):
#             section = nn.ModuleList()
#             for _ in range(1, group):
#                 section.append(ResNetBlock(input, input))
#             # don't upsample on last layer
#             if i != len(layers)-1:
#                 section.append(ResNetBlock(input, input*2, downsample=True))
#                 input *= 2
#             self.network.extend(section)
        
#         # self.network.append(nn.)
#         self.last.append(nn.Conv1d(512*3*5*5, 512*5, kernel_size=1, stride=1, padding=0))
#         self.last.append(nn.ReLU(True))
#         self.last.append(nn.Conv1d(512*5, 512, kernel_size=1, stride=1, padding=0))
#         self.last.append(nn.ReLU(True))
#         self.fc =nn.Linear(512, num_classes)

#         # self.network.append(nn.AvgPool3d((1, 4, 4), stride=1))
#         # out_features = self.network[-1].out_channels
#         # last_section = nn.ModuleList()
#         # last_section.append(nn.Linear(6144, num_classes))
#         # self.network.extend(last_section)

#     def forward(self, x):
#         print()
#         print("START FORWARD")
#         for i, m in enumerate(self.network):
#             input_size = x.size()
#             print(input_size)
#             x = m(x)
        
#         input_size = x.size()
#         print(input_size)
#         x = x.view(self.batch_size, -1, 1)



#         for i, m in enumerate(self.last):
#             input_size = x.size()
#             print(input_size)
#             x = m(x)
#         x = x.view(self.batch_size, -1)
#         x = self.fc(x)

#         print("END FORWARD")
#         print()
#         return F.log_softmax(x, dim=1)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BaseModel):

    def __init__(self,
                 block,
                 layers,
                 sample_size=84,
                 sample_duration=16,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


# class TSN(BaseModel):
#     def __init__(self):
#         super().__init__()

#         self.spatial_net = self.make_spatial()
#         self.temporal_net = self.make_temporal()


#         self.inplanes = 64
#         self.conv1 = nn.Conv3d(
#             3,
#             64,
#             kernel_size=7,
#             stride=(1, 2, 2),
#             padding=(3, 3, 3),
#             bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
#         self.layer2 = self._make_layer(
#             block, 128, layers[1], shortcut_type, stride=2)
#         self.layer3 = self._make_layer(
#             block, 256, layers[2], shortcut_type, stride=2)
#         self.layer4 = self._make_layer(
#             block, 512, layers[3], shortcut_type, stride=2)
#         last_duration = int(math.ceil(sample_duration / 16))
#         last_size = int(math.ceil(sample_size / 32))
#         self.avgpool = nn.AvgPool3d(
#             (last_duration, last_size, last_size), stride=1)
#         self.fc = nn.Linear(1024 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if shortcut_type == 'A':
#                 downsample = partial(
#                     downsample_basic_block,
#                     planes=planes * block.expansion,
#                     stride=stride)
#             else:
#                 downsample = nn.Sequential(
#                     nn.Conv3d(
#                         self.inplanes,
#                         planes * block.expansion,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False), nn.BatchNorm3d(planes * block.expansion))

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)

#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return F.log_softmax(x, dim=1)

#     def make_spatial(self):
#         pass
    
#     def make_temporal(self):
