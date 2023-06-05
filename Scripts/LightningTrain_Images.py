import numpy as np
import os
import sys
import gc
import pandas as pd
from PIL import Image
import cv2
import argparse
from Scripts import LightningFCT
import argparse
import cv2
import gc
import numpy as np
import os
import pandas as pd
import sys
from PIL import Image

from Scripts import LightningFCT

Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from pytorch_lightning.loggers import WandbLogger

sys.path.append(os.path.abspath("scripts"))
from segmentation_model import ImageDataset, TestImageDataset
from utils import Timer, rle_fast, save_predictions_image


class CFG:
    seed = 1337
    comp_name = 'vesuvius'
    mode = "test"  # 'test'  # "train"

    # ============== model cfg =============
    model_name = 'lightning_FCT'  #'Unet'
    backbone = 'efficientnet-b0'  # 'se_resnext50_32x4d'
    model_to_load = None  # '../model_checkpoints/lightning_FCT-zdim_6-epochs_3-validId_3/models/lightning_FCT-zdim_6-epochs_3-validId_3_dict_final.pt'
    target_size = 1
    in_chans = 6  # 8  # 6
    pretrained = True
    inf_weight = 'best'

    # ============== training cfg =============
    epochs = 1  # 15 # 30 50
    train_steps = 15000
    size = 224  # Size to shrink image to
    tile_size = 224
    stride = tile_size // 2

    train_batch_size = 1
    valid_batch_size = train_batch_size  # * 2
    valid_id = 3  # 3-6
    use_amp = True  # True  False

    scheduler = 'GradualWarmupSchedulerV2'  # 'CosineAnnealingLR'
    weight_decay = 1e-6
    max_grad_norm = 1000
    num_workers = 0

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # adamW warmup
    warmup_factor = 10
    lr = 1e-4  # / warmup_factor

    # From lightning args
    lr_factor = 0.5
    min_lr = 1e-6
    lr_scheduler = 'ReduceLROnPlateau'
    decay = 0.00

    # ============== Experiment cfg =============
    EXPERIMENT_NAME = f"{model_name}-zdim_{in_chans}-epochs_{epochs}-validId_{valid_id}-reduced-6chan"

    # ============== Inference cfg =============
    THRESHOLD = 0.35  # .52 score had a different value of .25

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
    L.seed_everything(seed=seed)  # From pytorch lightning


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


def get_lr_scheduler(CFG, optimizer):
    if CFG.lr_scheduler == 'none':
        return None
    if CFG.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=CFG.lr_factor,
            verbose=True,
            threshold=1e-6,
            patience=5,
            min_lr=CFG.min_lr)
        return scheduler
    if CFG.lr_scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=500
        )
        return scheduler


@torch.no_grad()
def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
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

config = {  # Config dict for Wandb
    'model_name': CFG.model_name,
    "epochs": CFG.epochs,
    "seed": CFG.seed,
    "z_dim": CFG.in_chans,
    "valid_id": CFG.valid_id
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

########################################################################
# #########         Create the model and Trainer             ###########
########################################################################
# Create model and setup params
print('Create the model')

model = LightningFCT.FCT(CFG)
# model.apply(init_weights)  # ToDo Has .zeros. maybe don't use if not learning?

if CFG.model_to_load:
    print("Loading model from disk")
    # parser = argparse.ArgumentParser(description='FCT for medical image')
    # model = LightningFCT.FCT.load_from_checkpoint('lightning_logs/version_2/checkpoints/epoch=74-step=4500.ckpt',
    #                                               args=parser.parse_args())
    model.load_state_dict(torch.load(CFG.model_to_load))  # loads model for inference or continue training

# FCT Default: 51,060,326
# FCT with 12 based filters & 2 classes:  11,997,513
# FCT with 2 attn heads:
#
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of Lightning model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Lightning FCT Parameters in FCT: %.3fM' % parameters)

precision = '16-mixed' if CFG.use_amp else 32  # Currently true
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# logger = wandb.init(project="Vesuvius", name=CFG.EXPERIMENT_NAME, config=config)
print("Create Trainer")
wandb_logger = WandbLogger(project='Vesuvius', name=CFG.EXPERIMENT_NAME)
trainer = L.Trainer(precision=precision, max_epochs=CFG.epochs, callbacks=[lr_monitor], logger=wandb_logger, log_every_n_steps=25)

print("Fit the model with Lightning")
with Timer("Fitting the model took"):
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.save_checkpoint(CFG.model_dir + f"{CFG.EXPERIMENT_NAME}_final.ckpt")
print("Training over.")

print("Do image generation here to gauge if model learned anything!")

print("Training over: Save final models")
torch.save(model, CFG.model_dir + f"{CFG.EXPERIMENT_NAME}_final.pt")
torch.save(model.state_dict(), CFG.model_dir + f"{CFG.EXPERIMENT_NAME}_dict_final.pt")

########################################################################
# #############           Generate Validation            ###############
########################################################################
# region Validation Prediction
print("Save image of validation predictions")

valid_labels = cv2.imread(CFG.comp_dataset_path + f"train/{CFG.valid_id}/inklabels.png", 0)
# labels = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
valid_labels = valid_labels / 255
pad0 = (CFG.tile_size - valid_labels.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_labels.shape[1] % CFG.tile_size)
valid_labels = np.pad(valid_labels, [(0, pad0), (0, pad1)], constant_values=0)

model.eval()  # Sets the model into evaluation mode
model.to(DEVICE)
mask_pred = np.zeros(valid_labels.shape, dtype=np.float32)
mask_count = np.zeros(valid_labels.shape, dtype=np.float32)
for batch_idx, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Inference Batches', leave=False):
    x = x.to(DEVICE)
    batch_size = y.size(0)

    with torch.no_grad():
        y_hat = model(x)
    # y_preds = torch.sigmoid(y_hat[2]).to('cpu').numpy()  # Might not need sigmoid with this model? todo
    y_preds = y_hat[2].to('cpu').numpy()  # Might not need sigmoid with this model? todo
    start_idx = batch_idx * CFG.valid_batch_size
    end_idx = start_idx + batch_size

    for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
        mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)  # dtype float32
        mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

save_predictions_image(mask_pred, file_name=f'{CFG.figures_dir}/{CFG.EXPERIMENT_NAME}_preds.png')  # plt.imshow(mask_pred); plt.savefig(f'{CFG.figures_dir}/{CFG.EXPERIMENT_NAME}_preds.png')
# endregion

# Garbage Collect before inference
del train_loader, train_dataset
gc.collect()
torch.cuda.empty_cache()

########################################################################
# #############             Final Inference              ###############
########################################################################
# region Final Inference
print("Do Inference")
print("Would do the model ensemble here in final submission!")

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
                    y_preds = torch.sigmoid(y_preds).to('cpu').numpy()  # Wasn't in the original code here.
                    # y_pred = TTA(images,model).cpu().numpy()  From TTA

                start_idx = step*CFG.valid_batch_size
                end_idx = start_idx + batch_size
                for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
                    # mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
                    mask_pred[y1:y2, x1:x2] += y_preds[i].reshape(mask_pred[y1:y2, x1:x2].shape)  # From TTA
                    mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

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
