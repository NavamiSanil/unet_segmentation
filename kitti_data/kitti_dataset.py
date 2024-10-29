import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset


class KittiDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Kitti dataset for semantic segmentation.

        Args:
            data_dir (str): Directory where the images and labels are stored.
            transform (callable, optional): Optional transform to be applied on an image and its corresponding mask.
        """
        self.data_dir = data_dir
        self.images = sorted(glob.glob(os.path.join(data_dir, 'images', '*.png')))
        self.masks = sorted(glob.glob(os.path.join(data_dir, 'labels', '*.png')))
        self.transform = transform

        assert len(self.images) == len(self.masks), "Number of images must match number of masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
