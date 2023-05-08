import albumentations as A
import cv2
import numpy as np
import os
import random
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.cuda.amp import autocast, GradScaler
from time import time
import matplotlib.pyplot as plt
import wandb
from torchmetrics.functional import accuracy
from Scripts.segmentation_model import CustomDataset, CustomModel


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    exp_name = 'vesuvius_2d_slide_exp_holdout_3'
    comp_dir_path = '../data/'
    comp_dataset_path = comp_dir_path

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    target_size = 1
    in_chans = 6  # 65

    # ============== training cfg =============
    size = 224  # Size to shrink image to
    tile_size = 224
    stride = tile_size // 2

    train_batch_size = 46
    valid_batch_size = train_batch_size * 2
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30  # 15 # 30

    # adamW warmup
    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 3

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000
    num_workers = 0

    seed = 1337

    # ============== set dataset path =============
    print('set dataset path')
    outputs_path = f'/kaggle/working/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + f'{comp_name}-models/'
    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def read_image_mask(fragment_id):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)  # idxs = range(65)

    for i in tqdm(idxs):
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    print(f"Length of image stack: {images.size}")

    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0

    return images, mask


def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):
        image, mask = read_image_mask(fragment_id)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))

                if fragment_id == CFG.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None])

                    valid_xyxys.append([x1, y1, x2, y2])
                else:
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    print(aug)
    return aug


def build_model(cfg, weight="imagenet"):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    return CustomModel(cfg, weight)


def save_predictions_image(ink_pred: Tensor, inklabels, file_name: str) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 30))
    axs.flatten()

    axs[0][0].imshow(inklabels, cmap="gray")
    axs[0][0].set_title("Labels")

    # Show the output images at different thresholds
    axs[0][1].imshow(ink_pred >= 0.4, cmap="gray")
    axs[0][1]. set_title("@ .4")

    axs[0][2].imshow(ink_pred >= 0.5, cmap="gray")
    axs[0][2].set_title("@ .5")

    axs[1][0].imshow(ink_pred >= 0.6, cmap="gray")
    axs[1][0].set_title("@ .6")

    axs[1][1].imshow(ink_pred >= 0.7, cmap="gray")
    axs[1][1].set_title("@ .7")

    axs[1][2].imshow(ink_pred >= 0.8, cmap="gray")
    axs[1][2].set_title("@ .8")

    [axi.set_axis_off() for axi in axs.ravel()]  # Turn off the axes on all the sub plots
    plt.savefig(file_name, transparent=False)
    print("Graph has saved")


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
# endregion


# region Warmup Scheduler
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler
# endregion


# Torch Functions
def train_fn(train_loader, model, optimizer, device, logger):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_hat = model(images)
            loss = BCELoss(y_hat, labels)  # Todo define this inside the model itself
            acc = accuracy(y_hat, labels, task='binary')
            dice = dice_coef_torch(y_hat, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    logger.log({"train_loss": losses.avg, "epoch": epoch, "train_dice": dice, 'train_accuracy': acc})
    return losses.avg


def valid_fn(valid_loader, model, device, valid_xyxys, valid_mask_gt, logger):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_hat = model(images)
            loss = BCELoss(y_hat, labels)
            acc = accuracy(y_hat, labels, task='binary')
            dice = dice_coef_torch(y_hat, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_hat = torch.sigmoid(y_hat).to('cpu').numpy()
        start_idx = step*CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_hat[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    logger.log({"val_loss": losses.avg})
    return losses.avg, mask_pred


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(10, 50+1, 5)) / 100:
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th

    print(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)
    return best_dice, best_th
# endregion


make_dirs(CFG)
seed_all_the_things(CFG.seed)

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)
config = {'model_name': CFG.model_name,
          'backbone': CFG.backbone,
          "epochs": CFG.epochs,
          "seed": CFG.seed,
          "z_dim": CFG.in_chans,
          }
Logger = init_logger(log_file=CFG.log_path)
logger = wandb.init(project="Vesuvius", name=CFG.exp_name, config=config)  # init_logger(CFG.log_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print("Load images")
train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()

print("Create datasets and loaders")
train_dataset = CustomDataset(train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
valid_dataset = CustomDataset(valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

print('train_dataset[0][0].shape')
print(train_dataset[0][0].shape)

print("Create the model")
model = build_model(CFG)
model.to(device)

optimizer = AdamW(model.parameters(), lr=CFG.lr)
scheduler = get_scheduler(CFG, optimizer)

print("Setup the loss functions")
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()

alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False, alpha=alpha, beta=beta)

# BCELoss(y_pred, y_true)
print("Train the model here.")

fragment_id = CFG.valid_id

valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
valid_mask_gt = valid_mask_gt / 255
pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

fold = CFG.valid_id

best_score = -1
best_loss = np.inf

print("Begin epoch training")
initial = time()
for epoch in range(CFG.epochs):
    start_time = time()

    avg_loss = train_fn(train_loader, model, optimizer, device, logger)
    avg_val_loss, mask_pred = valid_fn(valid_loader, model, device, valid_xyxys, valid_mask_gt, logger)
    scheduler.step()  # get_last_lr()

    best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

    # score = avg_val_loss
    score = best_dice

    elapsed = time() - start_time

    Logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
    Logger.info(f'Epoch {epoch+1} - avgScore: {score:.4f}')

    if score > best_score:
        best_loss = avg_val_loss
        best_score = score

        Logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
        Logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

        # ToDO Save the model separate from the preds for smaller kaggle model upload.
        torch.save({'model': model.state_dict(),
                    'preds': mask_pred},
                   CFG.model_dir + f'{CFG.model_name}_fold{fold}_best.pth')

final = time()
total = final - initial
print(f"Total epoch training took {total:.4} seconds or {total/60:.4} minutes")
logger.finish()

print("Checkpoint notebook section")
check_point = torch.load(CFG.model_dir + f'{CFG.model_name}_fold{fold}_{CFG.inf_weight}.pth', map_location=torch.device('cpu'))
mask_pred = check_point['preds']
best_dice, best_th = calc_fbeta(valid_mask_gt, mask_pred)
print(f'Best dice and th scores: {best_dice} : {best_score}')

save_predictions_image(mask_pred, valid_mask_gt, f'{CFG.figures_dir}{CFG.exp_name}_holdout.png')
