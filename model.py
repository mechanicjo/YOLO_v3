"""
python :
pytorch :
torchinfo :
추후에 version 작성하기
"""

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary


# Defining CNN Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()

        # **kwargs에는 kernel_size,  stride, padding 등 포함
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)  # param : negative slope
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        # Applying convolution only
        x = self.conv(x)
        # Applying BatchNorm and activation if needed
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


# Defining Residual Block
class ResidualBlock(nn.Module):
    # residual block은 input channels 수와 output channels 수가 같다.
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()

        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    # Conv2d 이후에 BatchNorm layer => bias = False
                    nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False, stride=1, padding=0),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, bias=False, stride=1, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1),
                )
            ]

        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            # save the input for the skip connection
            skip_connection = x
            # 1x1 conv -> bn -> leakyRelu -> 3x3 conv -> bn -> leakyRelu
            x = layer(x)
            if self.use_residual:  # use_residual이 과연 필요한가? -> 나중에 skip_connection 없을때 퍼포먼스 비교해보자!
                x = x + skip_connection
        return x


# DarkNet53 정의
class Darknet53(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Darknet 구조를 확인해보면
        # first_conv_layer ->
        # (conv_layer + res_block * repeat) * 5

        self.first_conv = CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)

        self.residual_block1 = nn.Sequential(
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
        )

        self.residual_block2 = nn.Sequential(
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
        )

        self.residual_block3 = nn.Sequential(
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
        )

        self.residual_block4 = nn.Sequential(
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
        )

        self.residual_block5 = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
        )

    def forward(self, x):
        # TODO : Darknet53에서 output으로 나오는 세가지 feature map 생산
        x = self.first_conv(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        feature_map_01 = self.residual_block3(x)
        feature_map_02 = self.residual_block4(feature_map_01)
        feature_map_03 = self.residual_block5(feature_map_02)
        return feature_map_01, feature_map_02, feature_map_03


class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            # TODO : YOLO Network Architecture에서 Upsampling에 사용
            CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        return self.upsample(x)


class YoloBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.route_conv = nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            CNNBlock(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            CNNBlock(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.output_conv = nn.Sequential(
            CNNBlock(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        route = self.route_conv(x)
        output = self.output_conv(route)
        return route, output  # route의 경우 upsampling을 거쳐 다음 yolo block으로 전달되고 output의 경우 DetectionLayer로 전달


# YOLO Network에서 output 된 결과를 이용하여 prediction
class DetectionLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # out_channels : ((c, x, y, w, h) + class_prob) per bounding box
        # 3 bbox per each cell
        self.pred = CNNBlock(in_channels=2 * in_channels, out_channels=3 * (1 + 4 + num_classes), kernel_size=1)

    # format: (batch_size, 3, grid_size, grid_size, num_classes + 5)
    def forward(self, x):
        output = self.pred(x)
        # x.size(0) : batch size / x.size(1) : RGB channels * ((c, x, y, w, h) + class_prob)
        # x.size(2) : grid_size
        # x.size(3) : grid_size
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


class YOLOv3(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.num_classes = num_classes

        self.darknet53 = Darknet53()

        # input_channels : darknet53 feature map 03 채널(1024)
        self.yolo_block_01 = YoloBlock(1024, 512)
        self.detectlayer_01 = DetectionLayer(512, num_classes)
        self.upsample_01 = Upsampling(512, 256)

        # input_channels : darknet53 feature map 02 채널(512) + upsampling 채널(256)
        self.yolo_block_02 = YoloBlock(512 + 256, 256)
        self.detectlayer_02 = DetectionLayer(256, num_classes)
        self.upsample_02 = Upsampling(256, 128)

        # input_channels : darknet53 feature map 01 채널(256) + upsampling 채널(128)
        self.yolo_block_03 = YoloBlock(256 + 128, 128)
        self.detectlayer_03 = DetectionLayer(128, num_classes)

    def forward(self, x):
        self.feature_map_01, self.feature_map_02, self.feature_map_03 = self.darknet53(x)

        x, output_01 = self.yolo_block_01(self.feature_map_03)
        output_01 = self.detectlayer_01(output_01)
        x = self.upsample_01(x)

        x, output_02 = self.yolo_block_02(torch.cat([x, self.feature_map_02], dim=1))
        output_02 = self.detectlayer_02(output_02)
        x = self.upsample_02(x)

        x, output_03 = self.yolo_block_03(torch.cat([x, self.feature_map_01], dim=1))
        output_03 = self.detectlayer_03(output_03)

        return output_01, output_02, output_03


# Testing YOLO v3 model
if __name__ == "__main__":
    # Setting number of classes and image size
    num_classes = 3
    IMAGE_SIZE = 640

    # Creating model and testing output shapes
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Asserting output shapes
    assert model(x)[0].shape == (1, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32,
                                 num_classes + 5)  # B, RGB, cell size, cell size, (c, x, y, w, h) + classes_prob
    assert model(x)[1].shape == (1, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert model(x)[2].shape == (1, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    print("Output shapes are correct!")

    # torch summary
    summary(model, input_size=(2, 3, 640, 640), device="cpu")

