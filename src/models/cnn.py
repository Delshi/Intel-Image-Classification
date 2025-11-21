import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualBlock


class CNN(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(CNN, self).__init__()

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

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.layer5 = self._make_layer(512, 1024, blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 1.75),
            nn.Linear(2048, num_classes),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(512)

        # Добавляем downsample слои для приведения размерностей
        self.downsample_x3_to_x4 = self._create_downsample(256, 512, stride=2)
        self.downsample_x3_to_x5 = self._create_downsample(
            256, 1024, stride=4
        )  # layer4 + layer5 оба с stride=2
        self.downsample_x4_to_x5 = self._create_downsample(512, 1024, stride=2)

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

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.layer5(x)
    #     x = self.dropout(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)

    #     return x

    def _create_downsample(self, in_channels, out_channels, stride):
        """Создает downsample слой для приведения размерностей"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)  # (1)
        x2 = self.layer2(x1)  # (2)
        x3 = self.layer3(x2)  # (3)

        # (4) = (4 полученный из 3) + (3)
        x4_output = self.layer4(x3)
        x3_downsampled = self.downsample_x3_to_x4(x3)
        x4 = x4_output + x3_downsampled
        x4 = self.bn2(x4)

        # (5) = (5 полученный из 4) + (3) + (4)
        x5_output = self.layer5(x4)
        x3_downsampled_to_x5 = self.downsample_x3_to_x5(x3)
        x4_downsampled_to_x5 = self.downsample_x4_to_x5(x4)
        x5 = x5_output + x3_downsampled_to_x5 + x4_downsampled_to_x5

        x = self.dropout(x5)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = self.classifier(x)

        return x
