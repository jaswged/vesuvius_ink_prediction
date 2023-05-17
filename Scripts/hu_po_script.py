import sys

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
from torchvision.transforms import ToTensor
import torch.utils.data as data
import logging

THRESHOLD = 0.4  # Since our output has to be binary, we have to choose a threshold, say 40% confidence.
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)


def rle(output):
    """
    Kaggle expects a "Runlength encoding" for the submission.
    :param output: torch.Tensor
    :return:
    """
    flat_img = np.where(output.flatten().cpu() > THRESHOLD, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))


class FragmentDataset(data.Dataset):
    def __init__(self):
        # Half-size of papyrus patches we'll use as model inputs
        buffer: int = 64
        # Number of slices in the z direction
        z_dim: int = 16
        # First slice in z direction
        z_start: int = 25


def load_image(data_dir, filename: str = 'mask.png', viz: bool = False):
    assert os.path.exists(data_dir), f"data directory {data_dir} does not exist"
    filepath = os.path.join(data_dir, filename)
    assert os.path.exists(filepath), f"File path {filepath} does not exist"
    log.info(f'Show image: {filepath}')
    _image = Image.open(filepath)
    if viz:
        plt.title(filepath)
        plt.imshow(_image)
        plt.show()
    _pt = ToTensor()(_image)
    log.info(f"loaded image: {filepath} with shape {_pt.shape} and dtype: {_pt.dtype}")
    return _pt


# Kaggle gpu : 'Tesla P100-PCIE-16GB'
if __name__ == "__main__":
    data_dir = 'data/train/1'
    print("Load an image")
    mask = load_image(data_dir=data_dir, viz=True)
