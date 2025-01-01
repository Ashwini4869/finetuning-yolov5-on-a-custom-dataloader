# Necessary Imports
import torch
from torch.utils.data import Dataset, dataloader, DataLoader
from PIL import Image
from pathlib import Path
import glob
import os
import cv2
import numpy as np
import torchvision.transforms.v2 as transform
import configparser

# Custom Imports
from helpers import read_labels, collate_fn

# Read constants from config file
config = configparser.ConfigParser()
config.read("config.cfg")

TRAIN_PATH = config["DATASET"]["train_path"]
VALID_PATH = config["DATASET"]["valid_path"]
num_batch = int(config["HYPERPARAMETERS"]["batch_size"])


# CONSTANTS
IMG_FORMATS = [".jpg", ".png"]


# Custom Dataset Class
class WIDERFaceDataset(Dataset):
    def __init__(self, path, img_size=640, transforms=None):
        self.img_size = img_size
        self.img_files = []
        self.label_files = []
        self.transforms = transforms

        self.img_dir = Path(path) / "images"

        if self.img_dir.is_dir():
            for img_file in self.img_dir.glob("*.*"):
                self.img_files.append(str(img_file))

        for image_file in self.img_files:
            label_file = image_file.replace("/images/", "/labels/")
            label_file = label_file.replace(".jpg", ".txt")
            self.label_files.append(label_file)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        image_file = self.img_files[index]
        im = cv2.imread(image_file)
        resized_image = cv2.resize(im, (640, 640))
        img = resized_image.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        tensor_img = torch.from_numpy(img)

        labels = read_labels(self.label_files[index])
        nl = len(labels)
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        bboxes = labels_out[:, 2:]
        labels_cls = labels_out[:, 1]
        transformed_labels_out = torch.zeros((nl, 6))
        if self.transforms:
            transformed_img, transformed_bboxes, transformed_labels_cls = (
                self.transforms(tensor_img, bboxes, labels_cls)
            )

            transformed_labels_out[:, 2:] = transformed_bboxes
            transformed_labels_out[:, 1] = transformed_labels_cls

        else:
            transformed_img = tensor_img
            transformed_labels_out = labels_out
        return transformed_img, transformed_labels_out, image_file


# Transformation Pipeline
transforms = transform.Compose(
    [
        transform.ColorJitter(brightness=0.1, hue=0.1),
        transform.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    ]
)

print("Loading dataset!")

# Load Train and Valid dataset
print("Loading training dataset...")
train_dataset = WIDERFaceDataset(path=TRAIN_PATH, transforms=transforms)
print("Loading validation dataset...")
valid_dataset = WIDERFaceDataset(path=VALID_PATH, transforms=transforms)

print("Dataset Loading Complete!")

print(
    f"Creating dataloaders with batch size {num_batch} for train and {num_batch} for validation"
)

# Dataloader for train and valid dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size=num_batch,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=num_batch, collate_fn=collate_fn, num_workers=8
)

print("Dataloaders Created!\n\n")
