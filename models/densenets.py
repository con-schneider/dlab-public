import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Densenet model used in the DLAB-VS training.

This file contains code from https://github.com/kenshohara/video-classification-3d-cnn-pytorch,
used under the following license:

Copyright (c) 2017 Kensho Hara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def densenet_dlab_vs(**kwargs):
    """Densenet architecture used in the paper.

    Returns:
        DenseNet: Instance of the DenseNet class
    """
    model = DenseNet(
        num_init_features=32, growth_rate=16, block_config=(4, 4, 4), **kwargs
    )
    return model


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append("denseblock{}".format(i))
        ft_module_names.append("transition{}".format(i))
    ft_module_names.append("norm5")
    ft_module_names.append("classifier")

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({"params": v})
                break
        else:
            parameters.append({"params": v, "lr": 0.0})

    return parameters


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm3d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv3d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv3d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class

    Args:
        grid_in: Input atom type density grid.
        in_channels (int, optional): Amount of channels in the input grid.
            Defaults to 32.
        growth_rate (int, optional): how many filters to add each layer (k in paper).
            Defaults to 32.
        block_config (tuple, optional): how many layers in each pooling block.
            Defaults to (6, 12, 24, 16).
        num_init_features (int, optional): number of filters in the first convolution layer
            Defaults to 64.
        bn_size (int, optional):  multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer). Defaults to 4.
        drop_rate (int, optional): dropout rate after each dense layer.
            Defaults to 0.
        out_classes (int, optional): number of classification classes.
            Defaults to 1.
        out_layer (str, optional): Layer type of the output layer.
            Defaults to "sigmoid". Options: "sigmoid", "linear".
        first_conv (int, optional): Size of the convolution kernel of the first Conv3d layer.
            Defaults to 7.
    """

    def __init__(
        self,
        grid_in,
        in_channels=32,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        out_classes=1,
        out_layer="sigmoid",
        first_conv=7,
    ):

        super(DenseNet, self).__init__()

        self.block_config = block_config

        self.sample_size = grid_in
        self.sample_duration = grid_in
        num_classes = out_classes

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=num_init_features,
                            kernel_size=first_conv,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm3d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        self.fc = nn.Linear(num_features, num_classes)

        if out_layer == "sigmoid":
            self.out_layer = nn.Sigmoid()
        elif out_layer == "linear":
            self.out_layer = nn.Identity()
        else:
            self.out_layer = nn.Identity()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)

        last_duration = int(
            math.floor(self.sample_duration / 2 ** (len(self.block_config) + 1))
        )
        last_size = int(
            math.floor(self.sample_size / 2 ** (len(self.block_config) + 1))
        )

        out = F.avg_pool3d(out, kernel_size=(last_duration, last_size, last_size)).view(
            features.size(0), -1
        )
        out = self.fc(out)
        out = self.out_layer(out)
        return out
