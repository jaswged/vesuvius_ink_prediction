import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple
import timm

IMAGE_SHAPE = 8181, 6330


class AverageCalc:  # Similar to 2-5d `AverageMeter` method
    """
    Calculates and stores the average and current value.
    Used to update the loss.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, size):
        self.value = value
        self.sum += value * size
        self.count += size
        self.avg = self.sum/self.count


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
                    y - self.subvolume_size : y + self.subvolume_size,
                    x - self.subvolume_size : x + self.subvolume_size]
        # 3d notebook had  subvolume = subvolume[np.newaxis, ...] here. Debug with ctrl U
        subvolume = torch.from_numpy(subvolume).to(torch.float32)
        inklabel = torch.tensor(self.inklabels[y, x], dtype=torch.float32)
        return subvolume, inklabel


def create_data_loader(batch_size: int, pixels: List[Tuple], dataset_kwargs: dict, shuffle: bool) -> DataLoader:
    dataset = SubvolumeDataset(pixels=pixels, **dataset_kwargs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# From 3D notebook. has both train and valid pixels input and returns a tuple of dataloaders.
def create_data_loaders(batch_size: int, train_pixels: List[Tuple], val_pixels: List[Tuple], dataset_kwargs: dict) -> Tuple[DataLoader]:
    train_ds = SubvolumeDataset(pixels=train_pixels, **dataset_kwargs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    val_ds = SubvolumeDataset(pixels=val_pixels, **dataset_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    return train_loader, val_loader


def dice_coef_torch(preds: Tensor, targets: Tensor, beta=0.5, smooth=1e-5) -> float:
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    # comment out if your model contains a sigmoid or equivalent activation layer. Ie You have already sigmoid(ed)
    #preds = torch.sigmoid(preds)

    # flatten label and prediction tensors
    preds = preds.view(-1).float()
    targets = targets.view(-1).float()

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)
    return dice


# Like val but also has train_steps, optimizer, logger params
def train_fn(train_steps: int, train_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer, logger):
    model.train()  # sets model into train mode.

    for i, (x, y) in enumerate(train_loader):  # x, y are (volume np.ndarray, inklabels: np.ndarray)
        x = x.to("cuda")
        y = y.to("cuda")

        if i > train_steps:  # Only do so many train steps.
            break

        optimizer.zero_grad()  # Resets the gradients
        yhat = model(x)  # Run the model predictions
        loss = loss_fn(yhat, y)  # Calculate loss
        loss.backward()
        optimizer.step()  # update dem weights
        logger.log({"train_loss": loss.item(), "train_dice": dice_coef_torch(yhat, y)})
        try:  # .detach() Returns a new Tensor, detached from the current graph.
            train_auc = roc_auc_score(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
            logger.log({"train_auc": train_auc})
        except ValueError:
            pass


@torch.no_grad()
def val_fn(val_loader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> dict:
    run_loss = AverageCalc()
    model.eval()  # Sets the model into evaluation mode
    y_val = []
    yhat_val = []

    for i, (x, y) in enumerate(val_loader):
        x = x.to("cuda")
        y = y.to("cuda")
        batch_size = x.shape[0]

        yhat = model(x)  # .squeeze(1) 3D
        loss = loss_fn(yhat, y)
        run_loss.update(loss.item(), x.shape[0])

        y_val.append(y.cpu())
        yhat_val.append(yhat.cpu())

    y_val = torch.cat(y_val)  # Concatenates the given sequence of seq tensors in the given dimension
    yhat_val = torch.cat(yhat_val)
    df_val = pd.DataFrame({"y_val": y_val, "yhat_val": yhat_val})

    try:
        val_auc = roc_auc_score(y_val.numpy(), yhat_val.numpy())
    except ValueError:
        val_auc = np.nan

    return {"val_loss": run_loss.avg, "val_dice": dice_coef_torch(yhat_val, y_val),
            "val_auc": val_auc, "df_val": df_val}


@torch.no_grad()
def generate_ink_pred(val_loader: DataLoader, model: nn.Module, val_pixels: List[Tuple[int, int]]) -> Tensor:
    model.eval()  # Sets the model into evaluation mode
    output = torch.zeros(IMAGE_SHAPE, dtype=torch.float32)
    for i, (x, y) in enumerate(val_loader):  # Todo add tqdm?
        x = x.to("cuda")
        y = y.to("cuda")
        batch_size = x.shape[0]

        yhat = model(x)
        for j, pred in enumerate(yhat):   # Todo add tqdm?
            output[val_pixels[i * batch_size + j]] = pred
    return output


class InkClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(config["model_name"], pretrained=True, in_chans=10, num_classes=0)  # only 10 in channels?
        self.backbone_dim = self.backbone(torch.rand(1, config["z_dim"], 2 * config["subvolume_size"], 2 * config["subvolume_size"])).shape[-1]
        self.classifier = nn.Linear(in_features=self.backbone_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)  # Pretrained efficientnet_b0 model
        logits = self.classifier(x)  # Classifies as ink/no_ink
        out = self.sigmoid(logits)  # Sigmoid for some reason
        return out.flatten()  # Flatten that bitch?