import os
import torch
from pytorch_lightning import LightningDataModule
from kitti_data.kitti_dataset import KittiDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split


class KittiDataModule(LightningDataModule):

    name = 'kitti'

    def __init__(
            self,
            data_dir: str = '/kaggle/input/data-semantics',
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        Kitti train, validation, and test dataloaders.

        Note: You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
        You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        Specs:
            - 200 samples
            - Each image is (3 x 1242 x 376)

        In total, there are 34 classes but some of these are not useful so by default we use only 19 of the classes
        specified by the `valid_labels` parameter.

        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            val_split: size of validation set (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
        """
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])

        # Split into train, val, test
        kitti_dataset = KittiDataset(self.data_dir, transform=self.default_transforms)

        val_len = round(val_split * len(kitti_dataset))
        test_len = round(test_split * len(kitti_dataset))
        train_len = len(kitti_dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            kitti_dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.testset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader
