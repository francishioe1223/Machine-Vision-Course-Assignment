import datetime
import os
import time
import csv

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.apple_dataset import AppleDataset
from utility.engine import train_one_epoch, evaluate

import utility.utils as utils
import utility.transforms as T

##################################################
# Train a Faster-RCNN predictor using the MinneApple dataset
##################################################

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_frcnn_model_instance(num_classes):
    # Load a pre-trained Faster-RCNN model (COCO)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one for the custom dataset
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def early_stop_monitor(patience, best_val_acc, val_acc, epoch, best_epoch):
    # If validation accuracy improves, reset patience
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        patience = 5  # Reset patience if improvement occurs
    else:
        patience -= 1

    # If patience reaches 0, stop training
    stop = patience <= 0
    return stop, best_val_acc, best_epoch


def main():
    # Set training parameters
    data_path = r"E:\\courses\\MV\\apple\\MinneApple\\detection"
    model_name = 'frcnn'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    epochs = 70
    workers = 8
    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-4
    lr_steps = [8, 11]
    lr_gamma = 0.1
    print_freq = 20
    output_dir = r'E:\\courses\\MV\\apple\\MinneApple\\outputs'
    
    csv_file = os.path.join(output_dir, "outcome2.csv")

    resume = ''  # Resume from checkpoint (optional)

    print("Loading data")
    num_classes = 2  # 1 class for apple, 1 for background
    dataset = AppleDataset(os.path.join(data_path, 'train'), get_transform(train=True))
    dataset_test = AppleDataset(os.path.join(data_path, 'val'), get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers, collate_fn=utils.collate_fn)

    # Initialize model
    model = get_frcnn_model_instance(num_classes)
    model.to(device)

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()

    # Early stopping variables
    best_val_acc = 0
    best_epoch = 0
    patience = 5  # Patience parameter

    # Open the CSV file for writing
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Epoch", "val_acc", "mAP@0.5", "Recall", "F1"])

        for epoch in range(epochs):
            # Train for one epoch
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
            lr_scheduler.step()

            if output_dir:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, os.path.join(output_dir, f"model_{epoch}.pth"))

            # Evaluate after each epoch and get metrics
            evaluator = evaluate(model, data_loader_test, device=device)

            # Extract metrics
            val_acc = evaluator.coco_eval['bbox'].stats[0]  # Mean Average Precision
            map_50 = evaluator.coco_eval['bbox'].stats[1]  # mAP@0.5 (adjust index if needed)
            recall = evaluator.coco_eval['bbox'].stats[8]  # Recall (adjust index if needed)
            precision = evaluator.coco_eval['bbox'].stats[1]  # Precision (adjust index if needed)
            
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0

            # Write metrics to CSV file
            csvwriter.writerow([epoch, val_acc, map_50, recall, f1_score])

            # Monitor for early stopping based on mAP
            stop, best_val_acc, best_epoch = early_stop_monitor(patience, best_val_acc, val_acc, epoch, best_epoch)
            if stop:
                print(f"Early stopping at epoch {epoch}. Best mAP: {best_val_acc}")
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    main()
