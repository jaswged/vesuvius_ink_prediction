import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn
from torch import Tensor
import torch.nn.functional as f
from torchmetrics.functional import accuracy
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from typing import List, Tuple
import timm

IMAGE_SHAPE = 14830, 9506  # 1: 8181, 6330    3: 7606, 5249


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


class SubvolumeDatasetTest(Dataset):
    def __init__(self, volume: np.ndarray, pixels: List[Tuple], subvolume_size: int):
        self.volume = volume
        # pixels in test mask
        self.pixels = pixels
        self.subvolume_size = subvolume_size

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        y, x = self.pixels[idx]
        subvolume = self.volume[:,
                    y - self.subvolume_size: y + self.subvolume_size,
                    x - self.subvolume_size: x + self.subvolume_size]
        subvolume = torch.from_numpy(subvolume).to(torch.float32)
        return subvolume


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
        subvolume = torch.from_numpy(subvolume).to(torch.float32)
        inklabel = torch.tensor(self.inklabels[y, x], dtype=torch.float32)
        return subvolume, inklabel


def create_data_loader(batch_size: int, pixels: List[Tuple], dataset_kwargs: dict, shuffle: bool) -> DataLoader:
    # dataset_kwargs {"volume": volume, "inklabels": inklabels, "subvolume_size": config["subvolume_size"]}
    dataset = SubvolumeDataset(pixels=pixels, **dataset_kwargs)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # pin_memory=True
    return data_loader


def create_test_data_loader(batch_size: int, volume: np.ndarray, pixels: List[Tuple], subvolume_size: int):
    dataset = SubvolumeDatasetTest(volume, pixels, subvolume_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def dice_coef_torch(preds: Tensor, targets: Tensor, beta=0.5, smooth=1e-5) -> float:
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    # comment out if your model contains a sigmoid or equivalent activation layer. Ie You have already sigmoid(ed)
    # preds = torch.sigmoid(preds)

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


# Like val but also has train_steps, optimizer, logger params
def train_fn(train_steps: int, data_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer, logger, epoch):
    model.train()  # sets model into train mode.

    example_ct = 0  # number of examples seen
    train_loss = 0
    train_total = 0

    for batch_idx, (x, y) in tqdm(enumerate(data_loader), total=train_steps, desc='Train Batches', leave=False):
        # x, y are (volume np.ndarray, inklabels: np.ndarray)
        x = x.to("cuda")
        y = y.to("cuda")

        if batch_idx > train_steps:  # Only do so many train steps.
            break

        optimizer.zero_grad()  # Resets the gradients
        y_hat = model(x)  # Run the model predictions

        # Calculate loss
        example_ct += len(x)  # Total number of examples the model has seen this epoch
        loss = loss_fn(y_hat, y)
        loss2 = f.binary_cross_entropy_with_logits(y_hat, y)
        dice = dice_coef_torch(y_hat, y)
# Todo do i need all of these fuckers?
#         acc = accuracy(y_hat, y, task='binary')
        # acc2 = (y_hat.argmax(dim=-1) == y).float().mean()

        train_loss += loss.item()
        train_total += y_hat.size(0)  # is this always = to example_ct above?

        # Backward pass ⬅
        loss.backward()

        # Step with optimizer and update dem weights
        optimizer.step()

        # Report metrics every 25th batch/step
        if batch_idx % 50 == 0:  # todo only calculate the extra losses when its a modulo run. save compute
            train_auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
            logger.log({"train_loss":  loss.item(), "train_dice": dice, "train_auc": train_auc})  # 'train_accuracy': acc

    avg_loss = round((train_loss / train_total) * 100, 2)
    print(f"Average loss for this epoch was:{avg_loss}")
    # logger.log({"Train Loss": train_loss/train_total, "Train Accuracy": acc, "epoch": epoch})
    # Todo return average losses?
    return None


@torch.no_grad()
def val_fn(val_steps: int, val_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, logger) -> dict:
    run_loss = AverageCalc()
    dice_loss = AverageCalc()
    model.eval()  # Sets the model into evaluation mode

    for batch_idx, (x, y) in tqdm(enumerate(val_loader), total=val_steps, desc='Valid Batches', leave=False):
        x = x.to("cuda")
        y = y.to("cuda")

        if batch_idx > val_steps:  # Only do so many train steps.
            break

        y_hat = model(x)  # .squeeze(1) 3D
        loss = loss_fn(y_hat, y)
        dice = dice_coef_torch(y_hat, y)

        run_loss.update(loss.item(), x.shape[0])
        dice_loss.update(dice, x.shape[0])

        if batch_idx % 50 == 0:
            # val_auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
            logger.log({"val_loss": run_loss.avg, "val_dice": dice})  # Todo check wandb for these. losses the same?

    # Val_dict is what is returned below v
    # logger.log({"val_loss": run_loss.avg, "val_dice": dice_coef_torch(yhat_val, y_val), "val_auc": val_auc})
    return {"val_loss": run_loss.avg, "val_dice": dice_loss.avg}


@torch.no_grad()
def generate_ink_pred(val_loader: DataLoader, model: nn.Module, val_pixels, image_shape) -> Tensor:
    model.eval()  # Sets the model into evaluation mode
    output = torch.zeros(image_shape, dtype=torch.float32)
    for i, (x, y) in tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Inference Batches', leave=False):
        x = x.to("cuda")
        y = y.to("cuda")
        batch_size = x.shape[0]  # 400, 10, 0, 60

        yhat = model(x)
        for j, pred in enumerate(yhat):
            output[val_pixels[i * batch_size + j]] = pred
    return output


@torch.no_grad()
def generate_ink_pred_test(test_loader: DataLoader, model: nn.Module, test_pixels, image_shape) -> Tensor:
    model.eval()  # Sets the model into evaluation mode
    output = torch.zeros(image_shape, dtype=torch.float32)
    for i, x in tqdm(enumerate(test_loader), total=len(test_loader), desc='Test Inference Batches', leave=True):
        x = x.to("cuda")
        batch_size = x.shape[0]

        yhat = model(x)  # b x.shape [400, 10, 0 , 60] | a x.shape [400, 10, 60, 60]
        for j, pred in enumerate(yhat):
            output[test_pixels[i * batch_size + j]] = pred
    return output


class InkClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(config["model_name"], pretrained=True, in_chans=config['z_dim'], num_classes=0)
        self.backbone_dim = self.backbone(torch.rand(1, config["z_dim"], 2 * config["subvolume_size"], 2 * config["subvolume_size"])).shape[-1]
        self.classifier = nn.Linear(in_features=self.backbone_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)  # Pretrained efficientnet_b0 model
        logits = self.classifier(x)  # Classifies as ink/no_ink
        out = self.sigmoid(logits)  # Sigmoid for some reason
        return out.flatten()  # Flatten that bitch?
