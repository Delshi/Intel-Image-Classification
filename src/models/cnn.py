import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualBlock
from .bottleneck import BottleneckBlock


class CNN(nn.Module):
    def __init__(self, num_classes=6, use_bottleneck=False, dropout_rate=0.3):
        super(CNN, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.dropout = nn.Dropout(0.1)

        # if use_bottleneck:
        #     self.layer1 = self._make_bottleneck_layer(64, 64, blocks=2, stride=1)
        #     self.layer2 = self._make_bottleneck_layer(
        #         256, 128, blocks=2, stride=2
        #     )  # 64*4=256
        #     self.layer3 = self._make_bottleneck_layer(
        #         512, 256, blocks=2, stride=2
        #     )  # 128*4=512
        #     self.layer4 = self._make_bottleneck_layer(
        #         1024, 512, blocks=2, stride=2
        #     )  # 256*4=1024

        #     self.classifier = nn.Sequential(
        #         nn.Dropout(dropout_rate),
        #         nn.Linear(2048, 1024),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(dropout_rate / 2),
        #         nn.Linear(1024, num_classes),
        #     )
        # else:
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        #!
        self.layer5 = self._make_layer(512, 1024, blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(2048, num_classes),
        )
        # end of else
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Для обычных Residual блоков"""
        downsample = None

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    # def _make_bottleneck_layer(self, in_channels, out_channels, blocks, stride):
    #     """Для Bottleneck блоков"""
    #     downsample = None
    #     expansion = 4

    #     if stride != 1 or in_channels != out_channels * expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(
    #                 in_channels,
    #                 out_channels * expansion,
    #                 kernel_size=1,
    #                 stride=stride,
    #                 bias=False,
    #             ),
    #             nn.BatchNorm2d(out_channels * expansion),
    #         )

    #     layers = []
    #     layers.append(BottleneckBlock(in_channels, out_channels, stride, downsample))

    #     for _ in range(1, blocks):
    #         layers.append(BottleneckBlock(out_channels * expansion, out_channels))

    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
