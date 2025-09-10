import os
import numpy as np
import torch
from PIL import Image

class AppleDataset(object):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load all image and label files, ensuring they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root_dir, "labels"))))

    def __getitem__(self, idx):
        # Load images and labels
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        label_path = os.path.join(self.root_dir, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")

        # Read label file (Yolo format)
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                elements = line.strip().split()
                class_id = int(elements[0])  # First column is class ID
                x_center, y_center, width, height = map(float, elements[1:])

                # Convert to [xmin, ymin, xmax, ymax]
                img_w, img_h = img.size
                xmin = (x_center - width / 2) * img_w
                xmax = (x_center + width / 2) * img_w
                ymin = (y_center - height / 2) * img_h
                ymax = (y_center + height / 2) * img_h
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id + 1)  # Shift class IDs to start from 1

        # Convert everything into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]
