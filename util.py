import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
import numpy as np

# Anchors
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

GRID_SIZE = [20, 40, 60]


# Define Util & Loss function
# 참고 자료 : https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/
def iou(box1, box2):
    # box1 and box2 are both in [x, y, width, height] format

    # Box coordinates of prediction
    b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2  # x - width/2
    b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2  # y - height/2
    b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2  # x + width/2
    b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2  # y + height/2

    # Box coordinates of ground truth
    b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
    b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
    b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
    b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

    # Get the coordinates of the intersection rectangle
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    # Make sure the intersection is at least 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the union area
    box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    union = box1_area + box2_area - intersection

    # Calculate the IoU score
    epsilon = 1e-6
    iou_score = intersection / (union + epsilon)

    # Return IoU score
    return iou_score


# Test IoU function
def test_iou():
    print("Testing IoU function...")

    # Case 1: Perfect match
    box1 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    box2 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    result = iou(box1, box2)
    print(f"Case 1 (Perfect match): IoU = {result.item()}")

    # Case 2: Partial overlap
    box1 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    box2 = torch.tensor([[60, 60, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    result = iou(box1, box2)
    print(f"Case 2 (Partial overlap): IoU = {result.item()}")

    # Case 3: No overlap
    box1 = torch.tensor([[10, 10, 20, 20]], dtype=torch.float32)  # [x, y, width, height]
    box2 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    result = iou(box1, box2)
    print(f"Case 3 (No overlap): IoU = {result.item()}")

    # Case 4: One box inside another
    box1 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    box2 = torch.tensor([[50, 50, 20, 20]], dtype=torch.float32)  # [x, y, width, height]
    result = iou(box1, box2)
    print(f"Case 4 (One box inside another): IoU = {result.item()}")

    # Case 5: Identical centers, different sizes
    box1 = torch.tensor([[50, 50, 60, 60]], dtype=torch.float32)  # [x, y, width, height]
    box2 = torch.tensor([[50, 50, 40, 40]], dtype=torch.float32)  # [x, y, width, height]
    result = iou(box1, box2)
    print(f"Case 5 (Identical centers, different sizes): IoU = {result.item()}")


# Non-maximum suppression function to remove overlapping bounding boxes
def nms(bboxes, iou_threshold, threshold):
    # bboxes in [best_class, confidence score, x, y, width, height] format
    # 설정한 threshold 보다 높은 신뢰도(confidence score)를 가진 bbox만 남기기
    bboxes = [box for box in bboxes if box[1] > threshold]

    # confidence score 순으로 정렬
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    # non-maximum suppression 이후 return할 bounding boxes list 선언
    bboxes_nms = []

    # 아직 처리되지 않은 bboxes가 있다면 계속 반복
    while bboxes:
        # 신뢰도가 가장 높은 박스를 pop (bboxes 리스트에서 제거됨) / 비교기준으로 설정
        first_box = bboxes.pop(0)

        # bboxes_nms에 first_box 추가
        bboxes_nms.append(first_box)

        next_bboxes = []
        # 남아 있는 모든 박스들과 신뢰도가 가장 높았던 박스 (first_box)와 하나씩 비교
        for curr_box in bboxes[:]:
            discard = False
            # 기준 박스 (first_box)랑 class가 같고 iou가 iou_threshold 보다 높으면 제거
            if curr_box[0] == first_box[0] \
                    and iou(torch.tensor(first_box[2:]), torch.tensor(curr_box[2:])) >= iou_threshold:
                discard = True

            if not discard:
                next_bboxes.append(curr_box)

        bboxes = next_bboxes

    # Return bounding boxes after non-maximum suppression.
    return bboxes_nms


# soft_nms(2017)
def soft_nms(bboxes, iou_threshold, threshold, sigma = 0.5):

    # B in original paper
    bboxes = [box for box in bboxes if box[1] > threshold]

    # D in original paper
    bboxes_nms = []

    while bboxes:
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

        # M in original paper
        first_box = bboxes.pop(0)

        bboxes_nms.append(first_box)

        for curr_box in bboxes:
            # calculate iou(M, b_i)
            ovr = iou(torch.tensor(first_box[2:]), torch.tensor(curr_box[2:]))

            # Gaussian decay
            weight = np.exp(-(ovr * ovr) / sigma)
            curr_box[1] *= weight.item()

        # discard bbox with low s_i score
        # bboxes = [box for box in bboxes if box[1] > threshold]


    return bboxes_nms





def test_nms():
    # Test bounding boxes in [class, confidence score, x, y, width, height] format
    bboxes = [
        [1, 0.9, 50, 50, 40, 40],  # High confidence, class 1
        [1, 0.85, 55, 55, 40, 40],  # Overlapping with box 1, class 1
        [2, 0.8, 100, 100, 30, 30],  # Class 2
        [1, 0.75, 50, 50, 20, 20],  # Smaller box inside box 1
        [1, 0.6, 10, 10, 20, 20],  # Low confidence, should be filtered
    ]

    # Parameters for NMS
    iou_threshold = 0.5
    confidence_threshold = 0.7

    # Perform NMS
    # result = nms(bboxes, iou_threshold, confidence_threshold)
    result = soft_nms(bboxes, iou_threshold, confidence_threshold)

    # Print results
    print("Bounding boxes after NMS:")
    for box in result:
        print(f"Class: {box[0]}, Confidence: {box[1]:.4f}, Coordinates: {box[2:]}")


if __name__ == "__main__":
    print("\n")
    test_nms()
#
# def convert_cells_to_bboxes():
#     # TODO
#     converted_bboxes = None
#     return converted_bboxes.tolist()
#
#
# def plot_image(image, boxes):
#     plt.show()
#
#
# def save_checkpoint(model, optimizer, filename="dr_bee_checkpoint.ptr.tar"):
#     print("==> Saving checkpoint")
#     checkpoint = {
#         "state_dict": model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#     }
#     torch.save(checkpoint, filename)
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
# class YoloLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.mse = nn.MSELoss()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.ce = nn.CrossEntropyLoss()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, pred, target, anchors):
#         # TODO
#         box_loss = 0
#         class_loss = 0
#         object_loss = 0
#         no_object_loss = 0
#
#         return (
#                 box_loss
#                 + object_loss
#                 + no_object_loss
#                 + class_loss
#         )
