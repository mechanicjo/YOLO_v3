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
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1) # param : negative slope
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
                    nn.Conv2d(channels, channels // 2, kernel_size=1, stride = 1, padding = 0),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, stride = 1, padding = 1),
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
            if self.use_residual: # use_residual이 과연 필요한가? -> 나중에 skip_connection 없을때 퍼포먼스 비교해보자!
                x = x + skip_connection
        return x


# DarkNet53 정의
class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()

        # Darknet 구조를 확인해보면
        # first_conv_layer ->
        # (conv_layer + res_block * repeat) * 5

        self.first_conv = CNNBlock(3, 32, kernel_size=3, stride=1, padding=1)

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
        return route, output # route의 경우 upsampling을 거쳐 다음 yolo block으로 전달되고 output의 경우 DetectionLayer로 전달


# YOLO Network에서 output 된 결과를 이용하여 prediction
class DetectionLayer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # out_channels : ((c, x, y, w, h) + class_prob) per bounding box
        # 3 bbox per each cell
        self.pred = CNNBlock(in_channels = 2 * in_channels, out_channels = 3 * (1 + 4 + num_classes), kernel_size=1)


    def forward(self, x):
        output = self.pred(x)
        return output



class Yolov3(nn.Module):
    def __init__(self, num_classes = 3):
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


# 모델 확인 코드
x = torch.randn((1, 3, 416, 416)) # RGB format의 640 x 640 랜덤 이미지
model = Yolov3(num_classes = 3)
out = model(x)
print(out[0].shape) # torch.Size([1, 3, 13, 13, 8]) / B, RGB, cell size, cell size, (c, x, y, w, h) + classes_prob
print(out[1].shape) # torch.Size([1, 3, 26, 26, 8])
print(out[2].shape) # torch.Size([1, 3, 52, 52, 8])

# torch summary
summary(model, input_size = (2, 3, 416, 416), device = "cpu")

# # Anchors
# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]
#
# GRID_SIZE = [13, 26, 52]
#
# # Define Util & Loss function
# # 참고 자료 : https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
# def iou(box1, box2, is_pred = True):
#
#     # TODO
#     iou_score = None
#
#     return iou_score
#
#
#
# def nms(bboxes, iou_threshold, threshold):
#     # TODO
#     bboxes_nms = None
#     return bboxes_nms
#
#
# def convert_cells_to_bboxes():
#     # TODO
#     converted_bboxes = None
#     return converted_bboxes.tolist()
#
#
# def plot_image(image, boxes):
#
#     plt.show()
#
#
# def save_checkpoint(model, optimizer, filename = "dr_bee_checkpoint.ptr.tar"):
#     print("==> Saving checkpoint")
#     checkpoint = {
#         "state_dict": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(checkpoint, filename)
#
#
#
# # Function to load checkpoint
# def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
#     print("==> Loading checkpoint")
#     checkpoint = torch.load(checkpoint_file, map_location=device)
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#
#
#
#
# class YoloLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.ce = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, pred, target, anchors):
#
#
#         # TODO
#         box_loss = 0
#         class_loss = 0
#         object_loss = 0
#         no_object_loss = 0
#
#         return(
#             box_loss
#             + object_loss
#             + no_object_loss
#             + class_loss
#         )
#
