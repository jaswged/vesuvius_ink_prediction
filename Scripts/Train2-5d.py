import traceback

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
from subvolume_dataloader import create_data_loader, InkClassifier, train_fn, val_fn, generate_ink_pred

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
model_to_load = None  # '../model_checkpoints/efficientnet_b0-subvolume_30-zstart_27-zdim_10-round_8-step_4000-thr_0.6.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# region Config
rect = {"x": 200, "y": 6700, "width": 1400, "height": 1000}  # Todo rect per fragment
config = {"model_name": "efficientnet_b0",
          "subvolume_size": 30,
          "z_start": 28,  # Test different start point and depths z_dims
          "z_dim": 9,
          "batch_size": 200,  # 100 was 3.7 GB; 150 was 4.1
          "epochs": 2,
          "train_steps": 1000,  # was 5000
          "threshold": 0.6,
          "seed": 1337}

EXPERIMENT_NAME = f"{config['model_name']}-zdim_{config['z_dim']}-epoch_{config['epochs']}-step_{config['train_steps']}-thr_{config['threshold']}"
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
        # normalize pixel intensity values into [0,1]
        slice_array = np.array(slice_png, dtype=np.float32) / 65535.0
        volume.append(slice_array)
    return np.stack(volume, axis=0)


def get_train_val_masks(mask: np.ndarray, rect: dict, subvolume_size: int) -> Tuple[np.ndarray]:
    # erode mask so that subvolumes will be fully within the mask
    eroded_mask = binary_erosion(mask, structure=np.ones((subvolume_size+10, subvolume_size+10)))
    # binary mask of the rectangle TODo fro test
    rect_mask = np.zeros((mask.shape), dtype=np.uint8)
    rect_mask[rect["y"]: rect["y"] + rect["height"], rect["x"]: rect["x"] + rect["width"]] = 1
    # validation set contains pixels inside the rectangle
    val_mask = eroded_mask * rect_mask
    # dilate rectangle mask so that training subvolumes will have no overlap with rectangle
    dilated_rect_mask = binary_dilation(rect_mask, structure=np.ones((subvolume_size, subvolume_size)))
    train_mask = eroded_mask * (1 - dilated_rect_mask)
    return train_mask, val_mask
# endregion


def save_predictions_image(ink_pred: Tensor, file_name: str) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    axs.flatten()

    axs[0][0].imshow(inklabels, cmap="gray")
    axs[0][0].set_title("Labels")

    # Show the output images at different thresholds
    axs[0][1].imshow(ink_pred.gt(0.4).cpu().numpy(), cmap="gray")
    axs[0][1].set_title("@ .4")

    axs[0][2].imshow(ink_pred.gt(0.5).cpu().numpy(), cmap="gray")
    axs[0][2].set_title("@ .5")

    axs[1][0].imshow(ink_pred.gt(0.6).cpu().numpy(), cmap="gray")
    axs[1][0].set_title("@ .6")

    axs[1][1].imshow(ink_pred.gt(0.7).cpu().numpy(), cmap="gray")
    axs[1][1].set_title("@ .7")

    axs[1][2].imshow(ink_pred.gt(0.8).cpu().numpy(), cmap="gray")
    axs[1][2].set_title("@ .8")

    [axi.set_axis_off() for axi in axs.ravel()]  # Turn off the axes on all the sub plots
    plt.savefig(file_name, transparent=False)
    print("Graph has saved")


try:
    # Begin the training code
    seed_all_the_things(config["seed"])
    logger = wandb.init(project="Vesuvius", name=EXPERIMENT_NAME, config=config)

    print("Setup the data")  # Todo all 3 fragments pl0x
    mask = None
    inklabels = None
    volume = None
    for i in range(1, 4):
        print(i)
        # mask = load_png(fragment_id="1", png_name="mask")
        # inklabels = load_png(fragment_id="1", png_name="inklabels")
        # volume = load_volume(fragment_id="1", z_start=config["z_start"], z_dim=config["z_dim"])
    mask = load_png(fragment_id="1", png_name="mask")
    inklabels = load_png(fragment_id="1", png_name="inklabels")
    volume = load_volume(fragment_id="1", z_start=config["z_start"], z_dim=config["z_dim"])
    dataset_kwargs = {"volume": volume, "inklabels": inklabels, "subvolume_size": config["subvolume_size"]}

    print("Get the train and validation masks")  # todo tqdm somehow?
    train_mask, val_mask = get_train_val_masks(mask, rect, config["subvolume_size"])  # Need more values to unpack warning
    train_pixels = list(zip(*np.where(train_mask == 1)))
    val_pixels = list(zip(*np.where(val_mask == 1)))
    # test_pixels = list(zip(*np.where(test_mask == 1)))  # Todo setup hold out test set
    del train_mask, val_mask
    gc.collect()

    print("Create the data loaders and datasets.")
    train_loader = create_data_loader(config["batch_size"], train_pixels, dataset_kwargs, shuffle=True)
    val_loader = create_data_loader(config["batch_size"], val_pixels, dataset_kwargs, shuffle=False)
    # test_loader = create_data_loader(config["batch_size"], test_pixels, dataset_kwargs, shuffle=False)
    best_val_dice = 0

    print('Create the model')
    model = InkClassifier(config).to(DEVICE)
    if model_to_load:
        model.load_state_dict(torch.load(model_to_load))  # loads model for inference?
    print(f'Model is: {model}')
    optimizer = Adam(model.parameters())

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = 'GradualWarmupSchedulerV2' vs = 'CosineAnnealingLR'
    loss_fn = nn.BCELoss()

    print(f"Train the model for {config['epochs']} epochs")
    initial = time.time()
    for epoch in tqdm(range(config["epochs"]), desc='Epochs'):  # TODO add tqdm here or in train function itself?
        start = time.time()
        train_fn(config["train_steps"], train_loader, model, loss_fn, optimizer, logger, epoch)

        val_dict = val_fn(val_loader, model, loss_fn, logger)  # Dict of {"val_loss", "val_dice", "val_auc"}

        if val_dict["val_dice"] > best_val_dice:
            best_val_dice = val_dict["val_dice"]
            EPOCH_NAME = f"{config['model_name']}-zdim_{config['z_dim']}-epoch_{epoch}-dice_{best_val_dice}-thr_{config['threshold']}"

            # Save model for inference
            torch.save(model.state_dict(), f"../model_checkpoints/{EPOCH_NAME}.pt")
            print(f"Model saved at round {epoch} with val_dice {best_val_dice:.4f}.")
            print("Since dice is better predict and save an image for Validation set")
            # Todo add a threshold sweep for multiple images?
            ink_pred = generate_ink_pred(val_loader, model, val_pixels)  # Switch to Test pixels and loader
            save_predictions_image(ink_pred, f'../model_images/{EPOCH_NAME}.png')

        if epoch % 5 == 0:
            # Every X epochs log out an image of pred on the holdout set.
            print("Log out an image of a prediction on the holdout set with epoch + model/fold name.")

        # todo log epoch time?
        end = time.time()
        total = start - end
        print(f"Epoch {epoch} took {total:.4} seconds or {total/60:.3} minutes")
    final = time.time()
    total = final - initial
    logger.finish()
    print(f"Total epoch training took {total:.4} seconds or {total/60:.4} minutes")

    print("Save final model")
    torch.save(model, f"../model_checkpoints/{EXPERIMENT_NAME}_final.pt")

    print("Model is trained. Show the output predicted on the hold out Test set")
    # Todo load the best dice model before generating this prediction!
    ink_pred = generate_ink_pred(val_loader, model, val_pixels)  # Switch to Test pixels and loader
    # Todo add a threshold sweep for multiple images?
    save_predictions_image(ink_pred, f'../model_images/{EXPERIMENT_NAME}_holdout.png')

except Exception as e:
    print(f"Fudge. effin broke. {e}")
    print(traceback.format_exc())
