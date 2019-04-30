import ipdb
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

downsampling_config_dict = {'layer1': {"ksize": 1, "padding": 0}, 'layer2': {
    "ksize": 3, "padding": 0}, 'layer3': {"ksize": 1, "padding": 0}, 'layer4': {"ksize": 1, "padding": 0}}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv3x3_with_dilation(in_planes, out_planes, stride=1, padding=2, dilation_ratio=2, groups=1):
    """3x3 convolution with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation_ratio, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, padding=1, dilation_ratio=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        # TODO
        if padding == 1 and dilation_ratio == 1:
            self.conv2 = conv3x3(planes, planes, stride, groups)
        else:
            self.conv2 = conv3x3_with_dilation(
                planes, planes, stride, padding, dilation_ratio, groups)

        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        # ipdb.set_trace()
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=0,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, planes[0], layers[0], groups=groups, norm_layer=norm_layer, padding=1, dilation_ratio=1, layer_name='layer1')
        self.layer2 = self._make_layer(
            block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, padding=0, dilation_ratio=1, layer_name='layer2')
        # TODO
        self.extra_1x1_conv3 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer3 = self._make_layer(
            block, planes[2], layers[2], stride=1, groups=groups, norm_layer=norm_layer, padding=2, dilation_ratio=2, layer_name='layer3')
        # TODO
        self.extra_1x1_conv4 = nn.Conv2d(
            1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer4 = self._make_layer(
            block, planes[3], layers[3], stride=1, groups=groups, norm_layer=norm_layer, padding=4, dilation_ratio=4, layer_name='layer4')
        # TODO
        self.extra_1x1_conv5 = nn.Conv2d(
            2048, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, padding=1, dilation_ratio=1, layer_name=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layer_downsampling_config_dict = downsampling_config_dict[layer_name]
            ksize = layer_downsampling_config_dict["ksize"]

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          stride=stride, kernel_size=ksize),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, groups, norm_layer, padding, dilation_ratio))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        #ipdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        conv_3_output = self.extra_1x1_conv3(x)
        x = self.layer3(x)
        conv_4_output = self.extra_1x1_conv4(x)
        x = self.layer4(x)
        conv_5_output = self.extra_1x1_conv5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, conv_3_output, conv_4_output, conv_5_output


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # ipdb.set_trace()
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    # if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
