import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from kitti_data.kitti_datamodule import KittiDataModule
from unet import UNet  # Replace with your actual model class


class SemSegment(pl.LightningModule):
    def __init__(self, datamodule: KittiDataModule, lr: float = 0.01, num_classes: int = 19, num_layers: int = 5,
                 features_start: int = 64, bilinear: bool = False):
        super().__init__()
        self.datamodule = datamodule
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(num_classes=self.num_classes,
                        num_layers=self.num_layers,
                        features_start=self.features_start,
                        bilinear=self.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


def cli_main():
    pl.seed_everything(1234)

    # Data Module
    data_dir = '/kaggle/input/data-semantics'  # Update this path
    kitti_data_module = KittiDataModule(data_dir=data_dir, batch_size=32, num_workers=4)

    # Model
    model = SemSegment(datamodule=kitti_data_module)

    # Trainer
    trainer = Trainer(accelerator='gpu', devices=1)  # Use GPU if available

    # Train
    trainer.fit(model, kitti_data_module)


if __name__ == '__main__':
    cli_main()

