import os
import torch
import torch.utils.data
import torchvision
import numpy as np

from data.apple_dataset import AppleDataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utility.utils as utils
import utility.transforms as T
import cv2
from torchvision.ops import nms

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_maskrcnn_model_instance(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_frcnn_model_instance(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # 参数直接写在代码中
    data_path = "E:/courses/MV/apple/MinneApple/detection/test"
    output_file = "E:/courses/MV/apple/MinneApple/outcome.txt"
    weight_file = "E:/courses/MV/apple/MinneApple/outputs/model_69.pth"
    output_dir = "E:/courses/MV/apple/MinneApple/outcome"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_mask_rcnn = False  # 设置为 True 使用 Mask-RCNN，False 使用 Faster-RCNN

    num_classes = 2

    print("Loading model")
    if use_mask_rcnn:
        model = get_maskrcnn_model_instance(num_classes)
    else:
        model = get_frcnn_model_instance(num_classes)

    checkpoint = torch.load(weight_file, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    print("Creating data loaders")
    dataset_test = AppleDataset(data_path, get_transform(train=False))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   shuffle=False, num_workers=1,
                                                   collate_fn=utils.collate_fn)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    score_threshold = 0.5
    nms_iou_threshold = 0.4

    total_images = 0
    correct_predictions = 0

    with open(output_file, 'a') as f:
        for image, targets in data_loader_test:
            image = list(img.to(device) for img in image)
            outputs = model(image)
            for ii, output in enumerate(outputs):
                total_images += 1

                # 获取真实苹果数量
                real_count = len(targets[ii]['boxes'])

                # 获取预测结果
                boxes = output['boxes'].cpu().detach().numpy()
                scores = output['scores'].cpu().detach().numpy()

                mask = scores >= score_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                if len(boxes) == 0:
                    predicted_count = 0
                else:
                    keep = nms(torch.tensor(boxes), torch.tensor(scores), nms_iou_threshold)
                    boxes = boxes[keep]
                    predicted_count = len(boxes)

                # 判断预测是否正确
                if abs(predicted_count - real_count) <= 5:
                    correct_predictions += 1

                # 输出可视化结果
                img_id = targets[ii]['image_id']
                img_name = data_loader_test.dataset.get_img_name(img_id)
                img_path = os.path.join(data_path, 'images', img_name)
                img = cv2.imread(img_path)
                boxes = boxes.astype(int)

                for i in range(len(boxes)):
                    [x1, y1, x2, y2] = boxes[i]
                    score = scores[i]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                count_text = f"apple counts: {predicted_count} (real: {real_count})"
                cv2.putText(img, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                output_path = os.path.join(output_dir, f"predicted_{img_name}")
                cv2.imwrite(output_path, img)

                print(f"Saved visualization to: {output_path}")

    # 计算准确率
    accuracy = correct_predictions / total_images
    print(f"Total images: {total_images}, Correct predictions: {correct_predictions}, Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
