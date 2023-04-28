import os
import sys
import gc
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from time import time
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import wandb
from random import random
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
import traceback
from Lightning_Model import LightningInkModel, SubvolumeDataset

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# region Config
IMAGE_SHAPE = 8181, 6330
learning_rate = 0.0022
epochs = 1
rect = {"x": 1100, "y": 3400, "width": 1600, "height": 1300}  # Todo rect per fragment

config = {"model_name": "efficientnet_b0",
          "subvolume_size": 30,
          "z_start": 27,  # halfway - 5.
          "z_dim": 10,
          "batch_size": 100,
          "epochs": epochs,
          "train_steps": 5000,
          "threshold": 0.6,  # Todo play with this
          "seed": 1337,
          "learning_rate": learning_rate}

EXPERIMENT_NAME = f"lightning_{config['model_name']}-epochs_{config['epochs']}-thr_{config['threshold']}"
# endregion


# region Methods
def seed_all_the_things(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed_everything(seed=seed)  # From pytorch lightning


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
    rect_mask[rect["y"]: rect["y"] + rect["height"], rect["x"]: rect["x"] + rect["width"]] = 1
    # validation set contains pixels inside the rectangle
    val_mask = eroded_mask * rect_mask

    # dilate rectangle mask so that training subvolumes will have no overlap with rectangle
    dilated_rect_mask = binary_dilation(rect_mask, structure=np.ones((subvolume_size, subvolume_size)))
    train_mask = eroded_mask * (1 - dilated_rect_mask)
    return train_mask, val_mask


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
# endregion


try:
    seed_all_the_things(config["seed"])
    # logger = wandb.init(project="Vesuvius", name=EXPERIMENT_NAME, config=config)  # this or the wandb logger?
    wandb_logger = WandbLogger(project='Vesuvius', name=EXPERIMENT_NAME)

    print("Setup the data")
    mask = load_png(fragment_id="1", png_name="mask")
    ink_labels = load_png(fragment_id="1", png_name="inklabels")
    volume = load_volume(fragment_id="1", z_start=config["z_start"], z_dim=config["z_dim"])
    dataset_kwargs = {"volume": volume, "inklabels": ink_labels, "subvolume_size": config["subvolume_size"]}

    print("Get the train and validation masks")
    train_mask, val_mask = get_train_val_masks(mask, rect, config["subvolume_size"])  # Need more values to unpack warning
    train_pixels = list(zip(*np.where(train_mask == 1)))
    val_pixels = list(zip(*np.where(val_mask == 1)))
    del train_mask, val_mask
    gc.collect()

    print("Create the data loaders and datasets.")
    train_loader = SubvolumeDataset.create_data_loader(config["batch_size"], train_pixels, dataset_kwargs, shuffle=True)
    val_loader = SubvolumeDataset.create_data_loader(config["batch_size"], val_pixels, dataset_kwargs, shuffle=False)
    # loader = LSTMDataLoader(dataset_train, dataset_valid, batch_size)  # Todo remove

    # Setup the Model Trainer
    print('create the model')
    model = LightningInkModel(config)
    print(f'Model is: {model}')
    # model.predict_step()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='val_loss',
        mode="min",
        dirpath='checkpoints',
        filename='epoch_{epoch:02d}-val_loss_{val_loss:.3f}'
    )  #     auto_insert_metric_name=False, mode="min",

    start = time()
    #  precision='16-mixed', callbacks=[checkpoint_callback]   # default log stpes is 50
    trainer = Trainer(accelerator='gpu', devices=1, precision=16, max_epochs=epochs, val_check_interval=0.1, logger=wandb_logger, log_every_n_steps=1, num_nodes=1)
    print("Trainer.fit")
    trainer.fit(model, train_loader, val_loader)
    end = time()
    total = end - start
    trainer.save_checkpoint("../checkpoints/final.ckpt")
    print(F"Finished training all epochs in {total:.5} seconds or {total/60} minutes")

    print("Model is trained. Show the output")
    ink_pred = generate_ink_pred(val_loader, model, val_pixels)
    del train_pixels, val_pixels
    gc.collect()

    print("Save an image of the predicted area.")
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axs.flatten()

    # Show the output images
    axs[0].imshow(ink_pred.gt(config["threshold"]).cpu().numpy(), cmap="gray")
    # axs.axis("off")
    # axs.set_title(f"{'ink_pred'}")
    axs[1].imshow(ink_labels, cmap="gray")
    # axs.axis("off")
    # axs.set_title(f"{'inklabels'}")

    # plt.savefig(f'../model_checkpoints/{EXPERIMENT_NAME}_transparent.png', transparent=True)
    plt.savefig(f'../model_checkpoints/{EXPERIMENT_NAME}.png', transparent=False)  # Compare after this is done.
    print("Graph has saved")

    # trainer.test(model, test_loader)
    # Get back best model for inference
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"Best checkpoint was: {best_checkpoint}")
    best_model = LightningInkModel.load_from_checkpoint(best_checkpoint)
    print(best_model)
except Exception as e:
    print(f"Fudge. effin broke. {e}")
    print(traceback.format_exc())
