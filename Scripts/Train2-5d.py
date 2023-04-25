import os
import sys
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from celluloid import Camera
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import time
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import wandb
from random import random
import timm
from subvolume_dataloader import create_data_loader, InkClassifier, train_fn, val_fn, generate_ink_pred

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# region Config
# NUM_SLICES = 65
# LOG_TABLE = True
rect = {"x": 1100, "y": 3400, "width": 1600, "height": 1300}  # Todo rect per fragment
config = {"model_name": "efficientnet_b0",
          "subvolume_size": 30,
          "z_start": 27,
          "z_dim": 10,
          "batch_size": 100,
          "epochs": 3,
          "train_steps": 4000,
          "threshold": 0.6,
          "seed": 1337}

EXPERIMENT_NAME = f"{config['model_name']}-subvolume_{config['subvolume_size']}-zstart_{config['z_start']}-zdim_{config['z_dim']}-round_{config['epochs']}-step_{config['train_steps']}-thr_{config['threshold']}"
# endregion


# region Methods
def seed_all_the_things(seed):
    """
    Sets the seed of the entire notebook for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def dice_coef_torch(preds: Tensor, targets: Tensor, beta=0.5, smooth=1e-5) -> float:
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    #comment out if your model contains a sigmoid or equivalent activation layer
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


# Finds batches where there is always ink in it.
def find_batches_both_classes(data_loader: DataLoader) -> Tuple[Tuple]:
    batch_inklabel_0 = None
    batch_inklabel_1 = None
    for subvolume, inklabel in data_loader:  # 3D used train_loader_check instead of data_loader. bug in their code?
        if inklabel.item() == 0 and batch_inklabel_0 is None:
            batch_inklabel_0 = subvolume, inklabel
        if inklabel.item() == 1 and batch_inklabel_1 is None:
            batch_inklabel_1 = subvolume, inklabel
        if batch_inklabel_0 is not None and batch_inklabel_1 is not None:
            break
    return batch_inklabel_0, batch_inklabel_1


def load_png(fragment_id: str, png_name: str) -> np.ndarray:
    fragment_dir = os.path.join(TRAIN_DIR, fragment_id)
    path = os.path.join(fragment_dir, f"{png_name}.png")
    image = Image.open(path)
    return np.array(image)


def load_volume(fragment_id: str, z_start: int, z_dim: int) -> np.ndarray:
    volume_dir = os.path.join(TRAIN_DIR, fragment_id, "surface_volume")
    print(f"Volume directory: {volume_dir}")
    volume = []

    for i in range(z_start, z_start + z_dim):
        slice_path = os.path.join(volume_dir, f"{i:02d}.tif")
        slice_png = Image.open(slice_path)
        # normalize pixel intesity values into [0,1]
        slice_array = np.array(slice_png, dtype=np.float32) / 65535.0
        volume.append(slice_array)
    return np.stack(volume, axis=0)


def get_train_val_masks(mask: np.ndarray, rect: dict, subvolume_size: int) -> Tuple[np.ndarray]:
    # erode mask so that subvolumes will be fully within the mask
    eroded_mask = binary_erosion(mask, structure=np.ones((subvolume_size+10, subvolume_size+10)))
    # binary mask of the rectangle
    rect_mask = np.zeros((mask.shape), dtype=np.uint8)
    rect_mask[rect["y"] : rect["y"] + rect["height"], rect["x"] : rect["x"] + rect["width"]] = 1
    # validation set contains pixels inside the rectangle
    val_mask = eroded_mask * rect_mask
    # dilate rectangle mask so that training subvolumes will have no overlap with rectangle
    dilated_rect_mask = binary_dilation(rect_mask, structure=np.ones((subvolume_size, subvolume_size)))
    train_mask = eroded_mask * (1 - dilated_rect_mask)
    return train_mask, val_mask
# endregion


seed_all_the_things(config["seed"])
logger = wandb.init(project="Vesuvius", name=EXPERIMENT_NAME, config=config)

# Setup the data
mask = load_png(fragment_id="1", png_name="mask")
inklabels = load_png(fragment_id="1", png_name="inklabels")

volume = load_volume(fragment_id="1", z_start=config["z_start"], z_dim=config["z_dim"])
dataset_kwargs = {"volume": volume, "inklabels": inklabels, "subvolume_size": config["subvolume_size"]}

train_mask, val_mask = get_train_val_masks(mask, rect, config["subvolume_size"])  # Need more values to unpack warning
train_pixels = list(zip(*np.where(train_mask == 1)))
val_pixels = list(zip(*np.where(val_mask == 1)))
del train_mask, val_mask
gc.collect();

# Actually train the model
model = InkClassifier(config).to(DEVICE)
print(f'Model is: {model}')
loss_fn = nn.BCELoss()

# train_loader, val_loader = create_data_loaders(config["batch_size"], train_pixels, val_pixels, dataset_kwargs) 3D
train_loader = create_data_loader(config["batch_size"], train_pixels, dataset_kwargs, shuffle=True)
val_loader = create_data_loader(config["batch_size"], val_pixels, dataset_kwargs, shuffle=False)
optimizer = Adam(model.parameters())
best_val_dice = 0

print(f"Train the model for {config['epochs']} rounds")
initial = time.time()
for rnd in tqdm(range(config["epochs"])):  # TODO add tqdm here or in train function itself?
    train_fn(config["train_steps"], train_loader, model, loss_fn, optimizer, logger)
    val_dict = val_fn(val_loader, model, loss_fn)  # Dict of {"val_loss", "val_dice", "val_auc", "df_val"}

    logger.log({"val_loss": val_dict["val_loss"], "val_dice": val_dict["val_dice"], "val_auc": val_dict["val_auc"]})

    if val_dict["val_dice"] > best_val_dice:
        best_val_dice = val_dict["val_dice"]
        torch.save(model.state_dict(), f"../model_checkpoints/{EXPERIMENT_NAME}.pt")
        print(f"Model saved at round {rnd} with val_dice {val_dict['val_dice']:.4f}.")
        # if LOG_TABLE:
        #     logger.log({"table": wandb.Table(dataframe=val_dict["df_val"])})
final = time.time()
total = final - initial
logger.finish()
print(f"Total training took {total:.4} seconds or {total/60:.4} mins")

print("Model is trained. Show the output")
ink_pred = generate_ink_pred(val_loader, model, val_pixels)

del train_pixels, val_pixels
gc.collect()

while True:
    try:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        axs.flatten()

        # Show the output images
        axs[0].imshow(ink_pred.gt(config["threshold"]).cpu().numpy(), cmap="gray")
        # axs.axis("off")
        # axs.set_title(f"{'ink_pred'}")
        axs[1].imshow(inklabels, cmap="gray")
        # axs.axis("off")
        # axs.set_title(f"{'inklabels'}")

        plt.savefig(f'../model_checkpoints/{EXPERIMENT_NAME}_transparent.png', transparent=True)
        plt.savefig(f'../model_checkpoints/{EXPERIMENT_NAME}.png', transparent=False)  # Compare after this is done.
        print("Graph has saved")
        break
    except Exception as e:
        print(f"Fudge. No graph. {e}")
