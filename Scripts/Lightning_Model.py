import os
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.optim import Adam
import torch.nn.functional as f
import pytorch_lightning as pl
from typing import List, Tuple
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import accuracy
import timm


class SubvolumeDataset(Dataset):
    def __init__(self, volume: np.ndarray, inklabels: np.ndarray, pixels: List[Tuple], subvolume_size: int):
        self.volume = volume
        self.inklabels = inklabels
        # pixels in train or validation mask
        self.pixels = pixels
        self.subvolume_size = subvolume_size  # is this 30? Then same as buffer in tutorial

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        y, x = self.pixels[idx]
        # no .view like tutorial
        subvolume = self.volume[:,
                    y - self.subvolume_size: y + self.subvolume_size,
                    x - self.subvolume_size: x + self.subvolume_size]
        # 3d notebook had  subvolume = subvolume[np.newaxis, ...] here. Debug with ctrl U
        subvolume = torch.from_numpy(subvolume).to(torch.float32)
        inklabel = torch.tensor(self.inklabels[y, x], dtype=torch.float32)
        return subvolume, inklabel

    # From 3D notebook. has both train and valid pixels input and returns a tuple of dataloaders.
    @staticmethod
    def create_data_loader(batch_size: int, pixels: List[Tuple], dataset_kwargs: dict, shuffle: bool) -> DataLoader:
        dataset = SubvolumeDataset(pixels=pixels, **dataset_kwargs)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return data_loader


class LightningInkModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(config["model_name"], pretrained=True, in_chans=10, num_classes=0)  # 10 here being z depth of images? todo use config["z_dim"] instead
        self.backbone_dim = self.backbone(torch.rand(1, config["z_dim"], 2 * config["subvolume_size"], 2 * config["subvolume_size"])).shape[-1]
        self.classifier = nn.Linear(in_features=self.backbone_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = config["learning_rate"]
        self.loss_fn = nn.BCEWithLogitsLoss  # nn.BCELoss()
        self.save_hyperparameters()  # To save hyperparms to WandB

    def forward(self, x):
        x = self.backbone(x)  # Pretrained efficientnet_b0 model
        logits = self.classifier(x)  # Classifies as ink/no_ink
        out = self.sigmoid(logits)  # Sigmoid for some reason
        return out.flatten()  # Flatten that bitch?

    def training_step(self, batch, batch_idx):
        y_hat, loss, acc, dice = self._get_preds_loss_accuracy_dice_auc(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', acc)
        self.log('train_dice', dice)
        # self.log('train_auc', auc)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss, acc, dice = self._get_preds_loss_accuracy_dice_auc(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc)
        self.log('val_dice', dice)
        # self.log('val_auc', auc)
        return loss  # Should this return y_hat also?

    def test_step(self, batch, batch_idx):
        y_hat, loss, acc, dice = self._get_preds_loss_accuracy_dice_auc(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_accuracy', acc)
        self.log("train_dice", dice)
        # self.log('test_auc', auc)
        return loss  # Should this return y_hat also?

    def _get_preds_loss_accuracy_dice_auc(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = f.binary_cross_entropy_with_logits(y_hat, y)
        # loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y, task='binary')  # acc = (y_hat.argmax(dim=-1) == y).float().mean()
        dice = self.calc_dice_coef(y_hat, y)
        # auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        return y_hat, loss, acc, dice

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def calc_dice_coef(preds: Tensor, targets: Tensor, beta=0.5, smooth=1e-5) -> float:
        # https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
        #comment out if your model contains a sigmoid or equivalent activation layer
        #preds = torch.sigmoid(preds)

        # flatten label and prediction tensors
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()

        y_true_count = targets.sum()
        ctp = preds[targets == 1].sum()
        cfp = preds[targets == 0].sum()
        beta_squared = beta * beta

        c_precision = ctp / (ctp + cfp + smooth)
        c_recall = ctp / (y_true_count + smooth)
        dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)
        return dice
