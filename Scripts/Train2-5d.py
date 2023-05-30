from copy import deepcopy
import traceback
import os
from os.path import join
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
from time import time
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import wandb
from Scripts.utils import Timer, save_predictions_image
from subvolume_dataloader import create_data_loader, InkClassifier, train_fn, val_fn, generate_ink_pred, \
    SubvolumeDatasetTest, create_test_data_loader, generate_ink_pred_test

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

DATA_DIR = "../data"
TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
model_to_load = "../model_checkpoints/efficientnet_b0-subvolume_30-zstart_27-zdim_10-round_8-step_4000-thr_0.6.pt"  # best_dice_epoch-efficientnet_b0-zdim_9-epoch_11-dice_0.4967948794364929-thr_0.6.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
fragment_id = "1"  # Todo put into config dict
train = False

# region Config
# Frag 1
# Frag 2 {"x": 700, "y": 9450, "width": 3300, "height": 2300}
rect = {"x": 2750, "y": 300, "width": 1500, "height": 1000}  # Todo rect per fragment
config = {"model_name": "efficientnet_b0",
          "subvolume_size": 30,
          "z_start": 28,  # Test different start point and depths z_dims
          "z_dim": 10,
          "batch_size": 400,  # 100 was 3.7 GB; 150 was 4.1
          "epochs": 1,
          "train_steps": 1000,  # was 5000
          "val_steps": 500,
          "threshold": 0.6,
          "seed": 1337}

EXPERIMENT_NAME = f"{config['model_name']}-zdim_{config['z_dim']}-epoch_{config['epochs']}"  # -step_{config['train_steps']}"
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


def load_png(fragment_id: str, png_name: str, is_train: bool = True) -> np.ndarray:
    fragment_dir = join(TRAIN_DIR, fragment_id) if is_train else join(TEST_DIR, fragment_id)
    path = join(fragment_dir, f"{png_name}.png")
    image = Image.open(path)  # returns PIL.Image.Image
    return np.array(image)


def load_volume(fragment_id: str, z_start: int, z_dim: int, is_train: bool = True) -> np.ndarray:
    volume_dir = join(TRAIN_DIR, fragment_id, "surface_volume") if is_train else join(TEST_DIR, fragment_id, "surface_volume")
    print(f"Volume directory: {volume_dir}")
    _volume = []

    for i in range(z_start, z_start + z_dim):
        slice_path = join(volume_dir, f"{i:02d}.tif")
        slice_png = Image.open(slice_path)
        # normalize pixel intensity values into [0,1]
        slice_array = np.array(slice_png, dtype=np.float32) / 65535.0
        _volume.append(slice_array)
    return np.stack(_volume, axis=0)


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


try:
    # Begin the training code
    seed_all_the_things(config["seed"])

    print("Setup the data")  # Todo all 3 fragments pl0x
    mask = load_png(fragment_id=fragment_id, png_name="mask")
    inklabels = load_png(fragment_id=fragment_id, png_name="inklabels")
    volume = load_volume(fragment_id=fragment_id, z_start=config["z_start"], z_dim=config["z_dim"])
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

    initial = time()
    best_model_state = model.state_dict()
    if train:  # Bool for whether to train or not.
        print(f"Train the model for {config['epochs']} epochs")
        logger = wandb.init(project="Vesuvius", name=EXPERIMENT_NAME, config=config)
        for epoch in range(config["epochs"]):  # TODO add tqdm here or in train function itself?
            print(f"Begin epoch: {epoch+1:02d}/{config['epochs']:02d}")
            start = time()
            # Could do a different train loader ever x epochs to switch up the fragments?
            train_fn(config["train_steps"], train_loader, model, loss_fn, optimizer, logger, epoch)

            val_dict = val_fn(config["val_steps"], val_loader, model, loss_fn, logger)  # Dict of {"val_loss", "val_dice", "val_auc"}

            if val_dict["val_dice"] > best_val_dice:
                best_val_dice = val_dict["val_dice"]
                EPOCH_NAME = f"{config['model_name']}-zdim_{config['z_dim']}-epoch_{epoch}-dice_{best_val_dice:3}-thr_{config['threshold']}"

                # Save model for inference
                epoch_name = f"{config['model_name']}-best_dice-{best_val_dice}_epoch-{EPOCH_NAME}"
                torch.save(model.state_dict(), f"../model_checkpoints/{epoch_name}.pt")
                best_model_state = deepcopy(model.state_dict())  # Save a copy of the best model state for later inference
                print(f"Model saved at round {epoch} with val_dice {best_val_dice:.4f}.")

                print("Save an image of new best preds")
                ink_pred = generate_ink_pred(val_loader, model, val_pixels, mask.shape)  # Switch to Test pixels and loader
                save_predictions_image(ink_pred, f'../model_checkpoints/images/{epoch_name}.png')

            if epoch % 5 == 0:
                # Every X epochs log out an image of pred on the holdout set.
                print("Log out an image of a prediction on the holdout set with epoch + model/fold name.")

            # todo log epoch time?
            end = time()
            total = start - end
            print(f"Epoch {epoch} took {total:.4} seconds or {total/60:.3} minutes")
        final = time()
        total = final - initial
        logger.finish()
        print(f"Total epoch training took {total:.4} seconds or {total/60:.4} minutes")

        print("Save final models")
        torch.save(model, f"../model_checkpoints/{EXPERIMENT_NAME}_final.pt")
        model.load_state_dict(best_model_state)
        torch.save(model, f"../model_checkpoints/best_{EXPERIMENT_NAME}_final.pt")

    # print("Model is trained. Show the output predicted on the hold out Test set img_val")
    # ink_pred = generate_ink_pred(val_loader, model, val_pixels, mask.shape)  # Switch to Test pixels and loader
    # print("Got preds. save image")
    # save_predictions_image(ink_pred, f'../model_checkpoints/images/{EXPERIMENT_NAME}_val.png')

    # Inference
    print("\n\nTry and do own inference. Load fragment 'A' as test frag")
    start = time()
    try:
        # model.load_state_dict(best_model_state)  # Todo load model from file?
        print("Try inference again with the binary dilation added.")
        mask_a = load_png(fragment_id="3", png_name="mask", is_train=True)
        eroded_mask = binary_erosion(mask_a, structure=np.ones((config["subvolume_size"]+10, config["subvolume_size"]+10)))
        rect_mask = np.zeros((mask_a.shape), dtype=np.uint8)
        dilated_rect_mask = binary_dilation(rect_mask, structure=np.ones((config["subvolume_size"], config["subvolume_size"])))
        test_mask = eroded_mask * (1 - dilated_rect_mask)

        test_pixels = list(zip(*np.where(eroded_mask == 1)))
        test_volume = load_volume(fragment_id="3", z_start=config["z_start"], z_dim=config["z_dim"], is_train=True)
        test_data_loader = create_test_data_loader(config["batch_size"], test_volume, test_pixels, config["subvolume_size"])
        print("Their inference?")

        with Timer("Their inference took"):
            test_preds = generate_ink_pred_test(test_data_loader, model, test_pixels, eroded_mask.shape)
        print("Save image to disk")
        save_predictions_image(test_preds, f'../model_checkpoints/images/final_pred_theirs_try.png')
    except Exception as e:
        final = time()
        total = final - start
        print(f"Failed after {total:.4} seconds or {total/60:.4} minutes.")
        print(f"Fudge. effin broke1 . {e}")
        print(traceback.format_exc())

    try:
        print("Try inference again with whole first fragment. 0 size rect.")
        mask = load_png(fragment_id="3", png_name="mask", is_train=True)
        inklabels = load_png(fragment_id="3", png_name="inklabels")  # second frag to make sure its big enough?
        volume = load_volume(fragment_id="3", z_start=config["z_start"], z_dim=config["z_dim"], is_train=True)
        dataset_kwargs = {"volume": volume, "inklabels": inklabels, "subvolume_size": config["subvolume_size"]}
        train_mask, val_mask = get_train_val_masks(mask, {"x": 0, "y": 0, "width": 0, "height": 0}, config["subvolume_size"])  # Need more values to unpack warning
        train_pixels = list(zip(*np.where(train_mask == 1)))
        train_loader = create_data_loader(config["batch_size"], train_pixels, dataset_kwargs, shuffle=False)

        start = time()
        test_preds = generate_ink_pred(train_loader, model, train_pixels, mask.shape)
        final = time()
        total = final - start
        print(f"Inference took {total:.4} seconds or {total/60:.4} minutes")
        print("After Test inference. Save preds to image")
        save_predictions_image(test_preds, f'../model_checkpoints/images/final_pred_theirs_whole.png')
    except Exception as e:
        final = time()
        total = final - start
        print(f"Failed after {total:.4} seconds or {total/60:.4} minutes.")
        print(f"Fudge. effin broke 2. {e}")
        print(traceback.format_exc())
except Exception as e:
    print(f"Fudge. effin broke final. {e}")
    print(traceback.format_exc())
