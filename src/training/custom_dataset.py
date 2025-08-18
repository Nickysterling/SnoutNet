# custom_dataset.py

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PetNoseDataset(Dataset):
    def __init__(self, annotations_file, img_dir, apply_flip=False, apply_color_jitter=False):
        self.img_labels = pd.read_csv(annotations_file, names=["filename", "coordinates"])
        self.img_dir = img_dir
        self.apply_flip = apply_flip
        self.apply_color_jitter = apply_color_jitter
        self.resize_transform = transforms.Resize((227, 227))  # Resize to the correct size for SnoutNet

        # Define individual transforms (but they will only be applied based on flags)
        self.flip_transform = transforms.Compose([
            transforms.functional.hflip  # Full horizontal flip
        ])

        self.color_jitter_transform = transforms.Compose([
            transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)

        # Ensure the image has 3 channels (convert to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = transforms.ToTensor()(image)

        # Store the original width and height of the image
        orig_height, orig_width = image.shape[1], image.shape[2]

        # Resize the image to 227x227
        image = self.resize_transform(image)

        # Get label (coordinates), parse the string, and convert to a list
        label_str = self.img_labels.iloc[idx, 1]
        label_list = label_str.strip("()").split(", ")
        label_list = [int(coord) for coord in label_list]

        # Scale the label coordinates according to the new image size
        x_scale = 227 / orig_width
        y_scale = 227 / orig_height
        label_list[0] = label_list[0] * x_scale  # Scale x-coordinate
        label_list[1] = label_list[1] * y_scale  # Scale y-coordinate

        # Apply the transformations based on the flags set in the constructor
        if self.apply_flip:
            image = self.flip_transform(image)
            label_list[0] = 227 - label_list[0]  # Adjust x-coordinate for flip
        if self.apply_color_jitter:
            image = self.color_jitter_transform(image)

        # Convert label list to a tensor
        label = torch.tensor(label_list, dtype=torch.float32)

        return image, label
