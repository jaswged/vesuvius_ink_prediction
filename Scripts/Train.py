from typing import List
import gc
import os
import sys
from torch import Tensor

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

sys.path.append(os.path.abspath("scripts"))

PREFIX = '../data/train/1/'
BUF = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 8  # Number of slices in the z direction
TRAINING_STEPS = 3000
LEARNING_RATE = 0.03
BATCH_SIZE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack: Tensor, label: Tensor, pixels: List):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[:, y - BUF:y + BUF + 1, x - BUF:x + BUF + 1].view(1, Z_DIM,
                                                                                       BUF * 2 + 1,
                                                                                       BUF * 2 + 1)
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel


class TestSubvolumeDataset(data.Dataset):
    def __init__(self, image_stack: Tensor, pixels: List):
        self.image_stack = image_stack
        self.pixels = pixels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        # print(self.image_stack.size())  # [8, 2727, 6330]
        subvolume = self.image_stack[:, y - BUF:y + BUF + 1, x - BUF:x + BUF + 1].view(1, Z_DIM,
                                                                                       BUF * 2 + 1,
                                                                                       BUF * 2 + 1)
        return subvolume


model = nn.Sequential(
    nn.Conv3d(1, 16, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
    nn.Conv3d(16, 32, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
    nn.Conv3d(32, 64, 3, 1, 1), nn.Dropout3d(0.2), nn.MaxPool3d(2, 2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128), nn.ReLU(),
    nn.Dropout(0.2),
    nn.LazyLinear(1), nn.Sigmoid()
).to(DEVICE)


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


if __name__ == "__main__":
    data_dir = 'data/train/1'
    print("Load an image")
    # mask_im = load_image(data_dir=data_dir, viz=True)
    # mask_im = load_image(data_dir=PREFIX, viz=True)
    # Load the 3d x-ray scan, one slice at a time
    images = [np.array(Image.open(filename), dtype=np.float16) / 65535.0 for filename in
              tqdm(sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[Z_START:Z_START + Z_DIM])]
    image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0)  #.to(DEVICE)

    rect = (1100, 3500, 1200, 950)
    # fig, ax = plt.subplots()
    # ax.imshow(label.cpu())
    # patch = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor='r', facecolor='none')
    # ax.add_patch(patch)
    # plt.show()

    print("Generating pixel lists...")
    # Split our dataset into train and val. The pixels inside the rect are the
    # val set, and the pixels outside the rect are the train set.
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

    print("Training...")
    train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=TRAINING_STEPS)

    print("Model.train")
    model.train()

    print("do the for loop")
    with Timer("Training took"):
        for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
            if i >= TRAINING_STEPS:
                break
            optimizer.zero_grad()
            outputs = model(subvolumes.to(DEVICE))

            loss = criterion(outputs, inklabels.to(DEVICE))
            loss.backward()
            optimizer.step()
            scheduler.step()

    # print("Training over. show whats in the rect")
    # eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
    # eval_loader = data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # output = torch.zeros_like(label).float()
    model.eval()
    # with torch.no_grad():
    #     with Timer("Inference took"):
    #         for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
    #             for j, value in enumerate(model(subvolumes.to(DEVICE))):
    #                 output[pixels_inside_rect[i * BATCH_SIZE + j]] = value
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(output.cpu(), cmap='gray')
    # ax2.imshow(label.cpu(), cmap='gray')
    # plt.show()
    #
    THRESHOLD = 0.4
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(output.gt(THRESHOLD).cpu(), cmap='gray')
    # ax2.imshow(label.cpu(), cmap='gray')
    # plt.show()

    del train_dataset, label, pixels_outside_rect, train_loader
    del image_stack
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    print("Do test inference on fragment 'a'")
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    TEST_PREFIX = '../data/test/a/'
    test_mask = np.array(Image.open(TEST_PREFIX + "mask.png").convert('1'))
    test_images = [np.array(Image.open(filename), dtype=np.float16) / 65535.0 for filename in
              tqdm(sorted(glob.glob(TEST_PREFIX + "surface_volume/*.tif"))[Z_START:Z_START + Z_DIM])]
    test_stack = torch.stack([torch.from_numpy(image) for image in test_images], dim=0)  # .to(DEVICE)

    print("Get inference pixels")
    with Timer("Gather test pixels"):
        test_pixels = []
        for pixel in zip(*np.where(test_mask == 1)):  # Pixel should be a tuple? coordinates
            if pixel[1] < BUF or pixel[1] >= test_mask.shape[1] - BUF or pixel[0] < BUF or pixel[0] >= test_mask.shape[0] - BUF:
                continue  # Too close to the edge
            else:
                test_pixels.append(pixel)
    test_dataset = TestSubvolumeDataset(test_stack, test_pixels)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    output = torch.zeros_like(torch.from_numpy(test_mask)).float()

    with Timer("Test Inference took"):
        with torch.no_grad():
            for i, subvolumes in enumerate(tqdm(test_loader, total=len(test_loader), desc="Inference")):
                # for j, value in enumerate(model(subvolumes)):
                # sub_ten = subvolumes[0]
                for j, value in enumerate(model(subvolumes.to(DEVICE))):
                    output[test_pixels[i * BATCH_SIZE + j]] = value

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(output.cpu(), cmap='gray')
    ax2.imshow(output.gt(THRESHOLD).cpu(), cmap='gray')
    plt.show()
