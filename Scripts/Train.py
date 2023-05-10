from typing import List
import gc
import os
import sys
from torch import Tensor
import timm

sys.path.append(os.path.abspath("scripts"))
from utils import Timer

# CUDA_LAUNCH_BLOCKING = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import PIL.Image as Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import ToTensor
from sklearn.metrics import roc_auc_score
import wandb

sys.path.append(os.path.abspath("scripts"))

# Setup WandB
WANDB_API_KEY = 'local-a2cc501204f722abe273d32f382f7b7438873ad7'
wandb.login(host='http://192.168.0.225:8080', key=WANDB_API_KEY)

PREFIX = '../data/train/1/'
TEST_PREFIX = '../data/test/a/'
DATA_DIR = "../data"
# data_dir = 'data/train/1'
BUF = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 9  # Number of slices in the z direction
TRAINING_STEPS = 3000
LEARNING_RATE = 0.0003
BATCH_SIZE = 130
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_to_load = None  # '../model_checkpoints/efficientnet_b0-subvolume_30-zstart_27-zdim_10-round_8-step_4000-thr_0.6.pt'
rect = (200, 6700, 1400, 1000)
# Frag 1 {"x": 200, "y": 6700, "width": 1400, "height": 1000}
# Frag 2 {"x": 700, "y": 9450, "width": 3300, "height": 2300}
# Frag 3 {"x": 1500, "y": 2550, "width": 1100, "height": 1100}
rect2 = {"x": 2750, "y": 300, "width": 1500, "height": 1000}  # Todo rect per fragment

SEED = 1337
THRESHOLD = 0.4

# From 2.5d
train = True
model_name = "efficientnet_b0"
model_to_load = None  # "../model_checkpoints/efficientnet_b0-subvolume_30-zstart_27-zdim_10-round_8-step_4000-thr_0.6.pt"  # best_dice_epoch-efficientnet_b0-zdim_9-epoch_11-dice_0.4967948794364929-thr_0.6.pt"
fragment_id = "1"
EXPERIMENT_NAME = f"{model_name}-zdim_{Z_DIM}-epoch_{EPOCHS}-step_{TRAINING_STEPS}-fragment_{fragment_id}"

torch.cuda.empty_cache()


# region Classes
class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack: Tensor, label: Tensor, pixels: List, BUF, Z_DIM):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.BUF = BUF
        self.Z_DIM = Z_DIM

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[:,
                    y - self.BUF:y + self.BUF + 1,
                    x - self.BUF:x + self.BUF + 1] \
            .view(1, self.Z_DIM,
                  self.BUF * 2 + 1,
                  self.BUF * 2 + 1)  # .view(1,
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel


class TestSubvolumeDataset(data.Dataset):
    def __init__(self, image_stack: Tensor, pixels: List, BUF, Z_DIM):
        self.image_stack = image_stack
        self.pixels = pixels
        self.BUF = BUF
        self.Z_DIM = Z_DIM

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        # print(self.image_stack.size())  # [8, 2727, 6330]
        subvolume = self.image_stack[:, y - self.BUF:y + self.BUF + 1, x - self.BUF:x + self.BUF + 1] \
            .view(1, self.Z_DIM,
                  self.BUF * 2 + 1,
                  self.BUF * 2 + 1)  # .view(1,
        return subvolume


class InkClassifier(nn.Module):
    def __init__(self, model_name, z_dim, subvolume_size, pre_trained):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pre_trained, in_chans=z_dim, num_classes=0)
        self.backbone_dim = self.backbone(torch.rand(1, z_dim, 2 * subvolume_size, 2 * subvolume_size)).shape[-1]
        self.classifier = nn.Linear(in_features=self.backbone_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)  # Pretrained efficientnet_b0 model
        logits = self.classifier(x)  # Classifies as ink/no_ink
        out = self.sigmoid(logits)  # Sigmoid for some reason
        return out.flatten()  # Flatten that bitch?


class AverageCalc:
    # Calculates and stores the average and current value for the loss
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


def load_image(data_dir, filename: str = 'mask.png', viz: bool = False):
    assert os.path.exists(data_dir), f"data directory {data_dir} does not exist"
    filepath = os.path.join(data_dir, filename)
    assert os.path.exists(filepath), f"File path {filepath} does not exist"
    print(f'Show image: {filepath}')
    _image = Image.open(filepath)
    if viz:
        plt.title(filepath)
        plt.imshow(_image)
        plt.show()
    _pt = ToTensor()(_image)
    print(f"loaded image: {filepath} with shape {_pt.shape} and dtype: {_pt.dtype}")
    return _pt


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


if __name__ == "__main__":
    seed_all_the_things(SEED)
    print("Load an image")
    # mask_im = load_image(data_dir=PREFIX, viz=True)
    # Load the 3d x-ray scan, one slice at a time
    images = [np.array(Image.open(filename), dtype=np.float16) / 65535.0 for filename in
              tqdm(sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[Z_START:Z_START + Z_DIM])]
    image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0)  # .to(DEVICE)

    print("Generating pixel lists...")
    # Split our dataset into train and val.
    # The pixels inside the rect are the val set, and the pixels outside the rect are the train set.
    pixels_inside_rect = []
    pixels_outside_rect = []
    mask = np.array(Image.open(PREFIX + "mask.png").convert('1'))
    label = torch.from_numpy(np.array(Image.open(PREFIX + "inklabels.png"))).gt(0).float().to(DEVICE)
    for pixel in zip(*np.where(mask == 1)):  # Pixel should be a tuple? coordinates
        if pixel[1] < BUF or pixel[1] >= mask.shape[1] - BUF or pixel[0] < BUF or pixel[0] >= mask.shape[0] - BUF:
            continue  # Too close to the edge

        if rect[0] <= pixel[1] <= rect[0] + rect[2] and rect[1] <= pixel[0] <= rect[1] + rect[3]:
            pixels_inside_rect.append(pixel)
        else:
            pixels_outside_rect.append(pixel)

    print("Setup the data loaders ...")
    train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect, BUF, Z_DIM)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect, BUF, Z_DIM)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    wandb_config = {
        "model_name": "efficientnet_b0",
        "batch_size": BATCH_SIZE,
        "train_steps": TRAINING_STEPS,
        "learning_rate": LEARNING_RATE,
        "architecture": model_name,
        "epochs": EPOCHS,
    }

    print("Prepare the Model")
    model = nn.Sequential(
        nn.Conv3d(1, 16, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
        nn.Conv3d(16, 32, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
        nn.Conv3d(32, 64, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
        nn.Flatten(start_dim=1),
        nn.LazyLinear(128), nn.ReLU(),
        nn.Dropout(0.2),
        nn.LazyLinear(1), nn.Sigmoid()
    ).to(DEVICE)
    # model = InkClassifier(model_name, Z_DIM, BUF, pre_trained=True).to(DEVICE)
    # print(model)
    if model_to_load:
        model.load_state_dict(torch.load(model_to_load))  # loads model for inference?
    criterion = nn.BCEWithLogitsLoss()  # nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # optimizer = Adam(model.parameters()) From Other notebook
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=TRAINING_STEPS)

    print("Training loop ...")
    with Timer("Training took"):
        logger = wandb.init(project="Vesuvius", name=EXPERIMENT_NAME, config=wandb_config)
        print(f"Train the model for {EPOCHS} epochs")
        for epoch in range(EPOCHS):
            print(f"Training in {epoch} epoch")
            model.train()
            train_loss = 0  # Reset loss for this epoch.
            train_total = 0

            for batch_idx, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
                if batch_idx >= TRAINING_STEPS:
                    break
                # inklables = inklables.squeeze()
                optimizer.zero_grad()
                y_hat = model(subvolumes.to(DEVICE))

                loss = criterion(y_hat, inklabels.squeeze().to(DEVICE))
                train_loss += loss.item()
                train_total += y_hat.size(0)  # is this always = to batch size?

                loss.backward()
                optimizer.step()
                scheduler.step()

                # Report metrics every 25th batch/step
                if batch_idx % 50 == 0:  # todo only calculate the extra losses when its a modulo run. save compute
                    dice = dice_coef_torch(y_hat, inklabels)
                    train_auc = roc_auc_score(inklabels.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
                    logger.log({"train_loss":  loss.item(), "train_dice": dice, "train_auc": train_auc})  # 'train_accuracy': acc
            avg_loss = round((train_loss / train_total) * 100, 2)
            print(f"Average loss for this epoch was:{avg_loss}")

            print(f"Validation time for epoch {epoch}")
            model.eval()
            run_loss = AverageCalc()
            dice_loss = AverageCalc()
            for batch_idx, (x, y) in tqdm(enumerate(eval_loader), total=len(eval_loader), desc='Valid Batches', leave=False):
                x = x.to("cuda")
                y = y.to("cuda")
                y_hat = model(x)
                loss = criterion(y_hat, y)
                run_loss.update(loss.item(), x.shape[0])
                dice_loss.update(dice, x.shape[0])
                if batch_idx % 50 == 0:
                    logger.log({"val_loss": run_loss.avg, "val_dice": dice})

    # ##################### Final Testing Validation of Rect ######################
    print("Training over. show what's in the rect")
    torch.save(model.state_dict(), f"../model_checkpoints/{EXPERIMENT_NAME}_dict.pt")
    torch.save(model, f"../model_checkpoints/{EXPERIMENT_NAME}.pt")  # TODO which of these do i really need?
    output = torch.zeros_like(label).float()
    model.eval()
    with torch.no_grad():
        with Timer("Inference took"):
            for batch_idx, (subvolumes, _) in enumerate(tqdm(eval_loader)):
                for j, value in enumerate(model(subvolumes.to(DEVICE))):
                    output[pixels_inside_rect[batch_idx * BATCH_SIZE + j]] = value

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output.cpu(), cmap='gray')
    ax2.imshow(label.cpu(), cmap='gray')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output.gt(THRESHOLD).cpu(), cmap='gray')
    ax2.imshow(label.cpu(), cmap='gray')
    plt.savefig(f'../model_checkpoints/images/final_pred_Rect.png', transparent=False)
    plt.show()

    del train_dataset, label, pixels_outside_rect, train_loader
    del image_stack
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    print("Do test inference on fragment 'a'")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    test_mask = np.array(Image.open(TEST_PREFIX + "mask.png").convert('1'))
    test_images = [np.array(Image.open(filename), dtype=np.float16) / 65535.0 for filename in
                   tqdm(sorted(glob.glob(TEST_PREFIX + "surface_volume/*.tif"))[Z_START:Z_START + Z_DIM])]
    test_stack = torch.stack([torch.from_numpy(image) for image in test_images], dim=0)  # .to(DEVICE)

    print("Get inference pixels")
    with Timer("Gather test pixels"):
        test_pixels = []
        for pixel in zip(*np.where(test_mask == 1)):  # Pixel should be a tuple? coordinates
            if pixel[1] < BUF or pixel[1] >= test_mask.shape[1] - BUF or pixel[0] < BUF or pixel[0] >= test_mask.shape[
                0] - BUF:
                continue  # Too close to the edge
            else:
                test_pixels.append(pixel)
    test_dataset = TestSubvolumeDataset(test_stack, test_pixels, BUF, Z_DIM)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    output = torch.zeros_like(torch.from_numpy(test_mask)).float()

    with Timer("Test Inference took"):
        with torch.no_grad():
            for batch_idx, subvolumes in enumerate(tqdm(test_loader, total=len(test_loader), desc="Inference")):
                for j, value in enumerate(model(subvolumes.to(DEVICE))):
                    output[test_pixels[batch_idx * BATCH_SIZE + j]] = value

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output.cpu(), cmap='gray')
    ax2.imshow(output.gt(THRESHOLD).cpu(), cmap='gray')
    plt.savefig(f'../model_checkpoints/images/final_pred_whole_A.png', transparent=False)
    plt.show()
