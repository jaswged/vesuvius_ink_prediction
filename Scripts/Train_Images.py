from copy import deepcopy

import numpy as np
import os
import sys
import gc
import pandas as pd
from PIL import Image

from Scripts.CompactConvolutionalTransformer import CCT
from Scripts.Compact_Transformer import CVT
from Scripts.FCT import FCT
from Scripts.FullyConvolutionalTransformer import FullyConvolutionalTransformer

Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import random
import torch
from torch import nn
import cv2
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import traceback
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
import wandb
from albumentations import ImageOnlyTransform

sys.path.append(os.path.abspath("scripts"))
from segmentation_model import ImageDataset, TestImageDataset, build_model, GradualWarmupSchedulerV2, calc_fbeta
from utils import Timer, dice_coef_torch, rle_fast, save_predictions_image, AverageMeter

import segmentation_models_pytorch as smp


class CFG:
    seed = 1337
    comp_name = 'vesuvius'
    mode = "train"  # 'test'  # "train"

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'  # 'se_resnext50_32x4d'
    model_to_load = None  # '../model_checkpoints/vesuvius_notebook_clone_exp_holdout_3/models/Unet-zdim_6-epochs_30-step_15000-validId_3-epoch_9-dice_0.5195_dict.pt'
    target_size = 1
    in_chans = 4  # 8  # 6
    pretrained = True
    inf_weight = 'best'

    # ============== training cfg =============
    epochs = 50  # 15 # 30
    train_steps = 15000
    size = 224  # Size to shrink image to
    tile_size = 224
    stride = tile_size // 2

    train_batch_size = 1
    valid_batch_size = train_batch_size * 2
    valid_id = 4
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'  # 'CosineAnnealingLR'
    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000
    num_workers = 0

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # adamW warmup
    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== Experiment cfg =============
    # ToDO consolidate these names into one
    # exp_name = f'vesuvius_notebook_clone_exp_holdout_{valid_id}'
    EXPERIMENT_NAME = f"{model_name}-zdim_{in_chans}-epochs_{epochs}-validId_{valid_id}"

    # ============== Inference cfg =============
    THRESHOLD = 0.3  # .52 score had a different value of .25

    # ============== set dataset paths =============
    comp_dir_path = '../'
    comp_dataset_path = comp_dir_path + 'data/'
    outputs_path = comp_dir_path + f'model_checkpoints/{EXPERIMENT_NAME}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{EXPERIMENT_NAME}.csv'
    model_dir = outputs_path + 'models/'
    figures_dir = outputs_path + 'figures/'

    # ============== Augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


# region Functions
def seed_all_the_things(seed=1337):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # when running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_image_and_labels(fragment_id: str, is_train: bool = True, mode: str = "train"):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        image = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)  # Shape: (8288, 6496, 6)

    print(f"Length of image stack: {images.size}")
    if is_train:
        labels = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
        labels = np.pad(labels, [(0, pad0), (0, pad1)], constant_values=0)

        labels = labels.astype('float32')
        labels /= 255.0  # Normalizing?
    else:
        labels = None

    return images, labels


def get_train_valid_dataset():
    train_images = []
    train_labels = []

    valid_images = []
    valid_labels = []
    valid_xyxys = []

    for frag_id in range(3, 7):
        print(f"Load images for fragment: {frag_id}")
        image, label = read_image_and_labels(frag_id)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            y2 = y1 + CFG.tile_size
            for x1 in x1_list:
                x2 = x1 + CFG.tile_size

                if frag_id == CFG.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_labels.append(label[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_labels.append(label[y1:y2, x1:x2, None])

    return train_images, train_labels, valid_images, valid_labels, valid_xyxys


def make_test_dataset(frag_id: str):
    test_images, _ = read_image_and_labels(frag_id, is_train=False, mode='test')

    x1_list = list(range(0, test_images.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, test_images.shape[0]-CFG.tile_size+1, CFG.stride))

    test_images_list = []
    xyxys = []
    for y1 in y1_list:
        y2 = y1 + CFG.tile_size
        for x1 in x1_list:
            x2 = x1 + CFG.tile_size

            test_images_list.append(test_images[y1:y2, x1:x2])
            xyxys.append((x1, y1, x2, y2))
    xyxys = np.stack(xyxys)

    test_dataset = TestImageDataset(test_images_list, transform=get_transforms(data='valid', cfg=CFG))

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.valid_batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    return test_loader, xyxys


def get_transforms(data, cfg):
    return A.Compose(cfg.train_aug_list) if data == 'train' else A.Compose(cfg.valid_aug_list)


def make_dirs(cfg):
    for dir in [cfg.outputs_path, cfg.model_dir, cfg.figures_dir, cfg.submission_dir]:
        os.makedirs(dir, exist_ok=True)
# endregion


########################################################################
# #############              Begin Script                ###############
########################################################################
# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

if not torch.cuda.is_available():
    print('CUDA is not available. Exiting...')
    exit()

seed_all_the_things(CFG.seed)

print("Lets do this!")
make_dirs(CFG)

config = {'model_name': CFG.model_name,
          'backbone': CFG.backbone,
          "epochs": CFG.epochs,
          "seed": CFG.seed,
          "z_dim": CFG.in_chans,
          }

########################################################################
# #############           Load up the data               ###############
########################################################################
print("Load images")
with Timer("Loading images took"):
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    valid_xyxys = np.stack(valid_xyxys)

print("Size of returned images")
print(len(train_images))
print(len(valid_images))

print("Create datasets and loaders")
train_dataset = ImageDataset(train_images, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = ImageDataset(valid_images, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

print(f"The length of train set is: {len(train_loader)}")
print(f"The length of valid set is: {len(valid_loader)}")

# Plot the dataset from the train notebook
"""
plot_dataset = CustomDataset(train_images, CFG, labels=train_masks)

transform = CFG.train_aug_list
transform = A.Compose([t for t in transform if not isinstance(t, (A.Normalize, ToTensorV2))])

plot_count = 0
for i in range(1000):
    image, mask = plot_dataset[i]
    data = transform(image=image, mask=mask)
    aug_image = data['image']
    aug_mask = data['mask']

    if mask.sum() == 0:
        continue

    fig, axes = plt.subplots(1, 4, figsize=(15, 8))
    axes[0].imshow(image[..., 0], cmap="gray")
    axes[1].imshow(mask, cmap="gray")
    axes[2].imshow(aug_image[..., 0], cmap="gray")
    axes[3].imshow(aug_mask, cmap="gray")
    
    plt.savefig(CFG.figures_dir + f'aug_fold_{CFG.valid_id}_{plot_count}.png')

    plot_count += 1
    if plot_count == 5:
        break
"""

# Create model and setup params
print('Create the model')
# model = InkClassifier(config)
# model = build_model(CFG)
model2 = FullyConvolutionalTransformer()
model = FCT(CFG.size, CFG.lr, CFG.weight_decay, CFG.min_lr)
# model = CVT(img_size=CFG.size,  # 224
#             embedding_dim=384,  # 16 * 16 * 3 = 768; 8 * 8 * 6
#             n_input_channels=6,  # was 3 put to my 6/8?
#             kernel_size=8,  # 16
#             dropout=0.1,
#             attention_dropout=0.1,
#             stochastic_depth=0.1,
#             num_layers=14,
#             num_heads=6,
#             mlp_ratio=4.0,
#             num_classes=2,
#             positional_embedding='learnable',
#            )  # patch-size=4,   *args, **kwargs
# model = CCT(
#     img_size=224,
#     embedding_dim=768,   # 16 * 16 * 3 = 768
#     n_input_channels=4,  # 3
#     n_conv_layers=1,
#     kernel_size=7,
#     stride=2,
#     padding=3,
#     pooling_kernel_size=3,
#     pooling_stride=2,
#     pooling_padding=1,
#     dropout=0.1,
#     attention_dropout=0.1,
#     stochastic_depth=0.1,
#     num_layers=6,  # 14,
#     num_heads=4,  # 6,
#     mlp_ratio=2.0,  # 4.0
#     num_classes=1,
#     positional_embedding='learnable'
# )
# Investigate. num_layers, num_heads and num_classes
model.to(DEVICE)

# Number of params.
# FCTor:   1,945,214
# FCT:    51,060,940
# TrimFCT 27,694,924
# Effnet:  6,252,909
# CVt:   101,300,201   25,277,187
# CCT:   101,836,035   181,342,212
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")

# ################### TEMP try out 1 batch of model ###################
for images, labels in train_loader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    break

# returned = model(images)
model2.to(DEVICE)
returned2 = model2(images)

# ################### Remove this ###################

optimizer = AdamW(model.parameters(), lr=CFG.lr)
# Setup Scheduler
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.epochs, eta_min=1e-7)
scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

# Loss Functions
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

# Setup custom loss function as average between dice and Binary Cross entropy
criterion = lambda y_pred, y_true: (BCELoss(y_pred, y_true) + DiceLoss(y_pred, y_true)) / 2

if CFG.model_to_load:
    print("Loading model from disk")
    model.load_state_dict(torch.load(CFG.model_to_load), )  # loads model for inference or continue training

fragment_id = CFG.valid_id

valid_labels = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
labels = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
valid_labels = valid_labels / 255
pad0 = (CFG.tile_size - valid_labels.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_labels.shape[1] % CFG.tile_size)
valid_labels = np.pad(valid_labels, [(0, pad0), (0, pad1)], constant_values=0)

fold = CFG.valid_id
best_score = -1

########################################################################
# #############             Train the model              ###############
########################################################################
print(f"Train the model for {CFG.epochs} epochs")
best_loss = np.inf
best_model_state = None
# logger = wandb.init(project="Vesuvius", name=CFG.EXPERIMENT_NAME, config=config)
initial = time()

for epoch in range(CFG.epochs):
    start = time()
    print(f"Begin epoch: {epoch+1:02d}/{config['epochs']:02d}")
    model.train()
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    # Training Loop
    train_loss = 0
    train_total = 0
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train Batches', leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        batch_size = y.size(0)

        with autocast(CFG.use_amp):
            y_hat = model(x)   # Run the model predictions
            loss = criterion(y_hat, y)  # Loss lambda

        # Calculate the loss
        losses.update(loss.item(), batch_size)
        train_loss += loss.item()
        train_total += batch_size

        # Backward pass
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()  # Resets the gradients

        # Report metrics every 50th batch/step
        if batch_idx % 50 == 0:  # todo only calculate the extra losses when its a modulo run. save compute?
            # train_auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
            dice = dice_coef_torch(y_hat, y)
            logger.log({"train_loss": losses.avg, "train_loss_mine": train_loss/train_total, "train_dice": dice})  #, "train_auc": train_auc})

    print(f"Average training loss for this epoch was:{losses.avg} or mine: {train_loss/train_total}. In theory these number should match!")
    # End Train loop

    # Begin Validation Loop
    model.eval()  # Sets the model into evaluation mode
    mask_pred = np.zeros(valid_labels.shape)
    mask_count = np.zeros(valid_labels.shape)

    valid_loss = 0
    dice_loss = 0
    valid_total = 0
    valid_losses = AverageMeter()

    for batch_idx, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid Batches', leave=False):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        batch_size = y.size(0)

        with torch.no_grad():
            y_hat = model(x)
            loss = criterion(y_hat, y)

        valid_losses.update(loss.item(), batch_size)
        valid_loss += loss.item()
        valid_total += batch_size
        dice_coef = dice_coef_torch(y_hat, y)
        dice_loss += dice_coef

        # make whole mask
        y_hat = torch.sigmoid(y_hat).to('cpu').numpy()
        start_idx = batch_idx * CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_hat[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

        if batch_idx % 50 == 0:
            logger.log({"val_loss": valid_losses.avg, "val_loss_mine": valid_loss/valid_total, "val_dice": dice_loss/valid_total})

    mask_pred /= mask_count
    scheduler.step(epoch)

    # Calculate CV for model epoch comparisons.
    best_dice, best_th = calc_fbeta(valid_labels, mask_pred)
    print(f'best_th: {best_th}, fbeta: {best_dice}')

    print("Save image of this epochs validation.")
    # TODO every nth epoch save out an image of our preds
    save_predictions_image(mask_pred, ink_labels=labels, file_name=f'{CFG.figures_dir}/validation_step_epoch_{epoch}_preds_best_th_{best_th}.png')

    if best_dice > best_score:
        best_loss = valid_losses.avg
        best_score = best_dice
        print(f"Epoch {epoch:02d} has better dice score: {dice_loss:.4f}. Saving model checkpoint")
        epoch_name = f"{CFG.EXPERIMENT_NAME}-epoch_{epoch}-dice_{best_score:.4f}"
        best_model_state = deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"{CFG.model_dir}/{epoch_name}_dict.pt")
        torch.save(model, f"{CFG.model_dir}/{epoch_name}.pt")

    end = time()
    total = end - start
    logger.log({"epoch_loss": valid_losses.avg, "epoch_dice": best_dice, "lr": scheduler.get_lr(), "best_th": best_th})
    print(f'Epoch {epoch+1:02d}/{config["epochs"]:02d} - avg_train_loss: {losses.avg:.4f}  avg_val_loss: {valid_losses.avg:.4f} time: {total:.5}s or {total/60:.3} mins at lr {scheduler.get_lr()}')
    print(f'Epoch {epoch+1} - avgScore: {best_dice:.4f}')

logger.finish()  # Close logger
final = time()
total = final - initial
logger.finish()
print(f"Total training took {total:.5} seconds or {total/60:.4} minutes for {CFG.epochs:02d} epochs")

print("Training over: Save final models")
torch.save(model, CFG.model_dir + f"{CFG.EXPERIMENT_NAME}_final.pt")
torch.save(model.state_dict(), CFG.model_dir + f"{CFG.EXPERIMENT_NAME}_dict_final.pt")

# Save as Open Neural Network eXchange format
torch.onnx.export(model, CFG.model_dir + "model.onnx")
logger.save(CFG.model_dir + "model.onnx")
# todo gc col and del

########################################################################
# #############           Generate Validation            ###############
########################################################################
# region Validation Prediction
print("Save image of validation predictions")
model.eval()  # Sets the model into evaluation mode
mask_pred = np.zeros(valid_labels.shape, dtype=np.float32)
mask_count = np.zeros(valid_labels.shape, dtype=np.float32)
for batch_idx, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Inference Batches', leave=False):
    x = x.to(DEVICE)
    batch_size = y.size(0)

    with torch.no_grad():
        y_hat = model(x)

    y_preds = torch.sigmoid(y_hat).to('cpu').numpy()
    start_idx = batch_idx * CFG.valid_batch_size
    end_idx = start_idx + batch_size

    for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
        mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)  # dtype float32
        mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

save_predictions_image(mask_pred, file_name=f'{CFG.figures_dir}/{CFG.EXPERIMENT_NAME}_preds.png')  # plt.imshow(mask_pred); plt.savefig(f'{CFG.figures_dir}/{CFG.EXPERIMENT_NAME}_preds.png')
# endregion

# todo gc col and del

########################################################################
# #############             Final Inference              ###############
########################################################################
# region Final Inference
print("Do Inference")
print("Would do the model ensemble here!")

if CFG.mode == 'test':
    print("Below is straight from the .41 submission notebook!")
    fragment_ids = sorted(os.listdir(CFG.comp_dataset_path + 'test'))
    print(f"Testing fragment Ids: {fragment_ids}")
    results = []
    for fragment_id in fragment_ids:
        print(f"Inferring for {fragment_id}")
        test_loader, xyxys = make_test_dataset(fragment_id)
        binary_mask = cv2.imread(CFG.comp_dataset_path + f"{CFG.mode}/{fragment_id}/mask.png", 0)
        binary_mask = (binary_mask / 255).astype(int)

        ori_h = binary_mask.shape[0]
        ori_w = binary_mask.shape[1]
        # mask = mask / 255

        pad0 = (CFG.tile_size - binary_mask.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - binary_mask.shape[1] % CFG.tile_size)

        binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

        mask_pred = np.zeros(binary_mask.shape, dtype=float)  # Kaggle didn't have the dtype arg.
        mask_count = np.zeros(binary_mask.shape, dtype=float)

        with Timer(f"Doing inference for fragment: {fragment_id} took"):
            for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
                images = images.to(DEVICE)
                batch_size = images.size(0)

                with torch.no_grad():
                    y_preds = model(images)
                    y_preds = torch.sigmoid(y_hat).to('cpu').numpy()  # Wasn't in the original code here.
                    # y_pred = TTA(images,model).cpu().numpy()  From TTA

                start_idx = step*CFG.valid_batch_size
                end_idx = start_idx + batch_size
                for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
                    # mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
                    mask_pred[y1:y2, x1:x2] += y_preds[i].reshape(mask_pred[y1:y2, x1:x2].shape)  # From TTA
                    mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

        print(f'mask_count_min: {mask_count.min()}')
        mask_pred /= mask_count

        # Setup plot for the graphs
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        axes[0].imshow(mask_count)
        axes[1].set_title("Mask")
        axes[1].imshow(mask_pred.copy())
        axes[1].set_title("Raw predictions")

        mask_pred = mask_pred[:ori_h, :ori_w]
        binary_mask = binary_mask[:ori_h, :ori_w]

        mask_pred = (mask_pred >= CFG.THRESHOLD).astype(int)
        mask_pred *= binary_mask

        axes[2].imshow(mask_pred)
        axes[2].set_title('Final Predictions')
        # plt.show()
        plt.savefig(f'{CFG.figures_dir}/Fragment_{fragment_id}_Final_Preds.png', transparent=False)

        inklabels_rle = rle_fast(mask_pred, CFG.THRESHOLD)

        results.append((fragment_id, inklabels_rle))

        del mask_pred, mask_count
        del test_loader

        gc.collect()
        torch.cuda.empty_cache()
        break

    print("Inference over. Assemble submission dataframe")
    sub = pd.DataFrame(results, columns=['Id', 'Predicted'])

    sample_sub = pd.read_csv(CFG.comp_dataset_path + 'sample_submission.csv')
    sample_sub = pd.merge(sample_sub[['Id']], sub, on='Id', how='left')
    print(sample_sub.head())

    # Save out the csv for kaggle to score it
    sample_sub.to_csv("submission.csv", index=False)
# endregion

print("End script")
