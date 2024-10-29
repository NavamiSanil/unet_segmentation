from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from unet import UNet
from kitti_data.kitti_datamodule import KittiDataModule


class SemSegment(pl.LightningModule):

    def __init__(self,
                 datamodule: pl.LightningDataModule = None,
                 lr: float = 0.01,
                 num_classes: int = 19,
                 num_layers: int = 5,
                 features_start: int = 64,
                 bilinear: bool = False,
                 network: str = 'unet',
                 **kwargs
                 ):
        super().__init__()

        assert datamodule
        self.datamodule = datamodule

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(num_classes=num_classes,
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir", type=str, default="/kaggle/input/data-semantics", help="Path to the dataset directory")
        parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=int, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")

        return parser


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()

    # Trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # Model args
    parser = SemSegment.add_model_specific_args(parser)
    args = parser.parse_args()

    # Data
    dm = KittiDataModule(data_dir=args.data_dir, batch_size=args.batch_size)  # Pass data_dir and batch_size directly

    # Model
    model = SemSegment(**args.__dict__, datamodule=dm)

    # Train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
