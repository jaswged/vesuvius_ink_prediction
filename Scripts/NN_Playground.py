import numpy as np
from Scripts.segmentation_model import ImageDataset
import os
import sys
import gc
from PIL import Image
import copy
Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import random
import torch
from torch import nn
import cv2
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import traceback
import albumentations as A
from albumentations.pytorch import ToTensorV2
from time import time
# import wandb
import math
from albumentations import ImageOnlyTransform
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
import segmentation_models_pytorch as smp


class CFG:
    seed = 1337
    comp_name = 'vesuvius'
    mode = "train"  # 'test'  # "train"

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'  # 'se_resnext50_32x4d'
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

    for frag_id in range(3, 5):
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


def get_transforms(data, cfg):
    return A.Compose(cfg.train_aug_list) if data == 'train' else A.Compose(cfg.valid_aug_list)


# ########################### Gather Data ###########################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
valid_xyxys = np.stack(valid_xyxys)

train_dataset = ImageDataset(train_images, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                          )
for images, labels in train_loader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    break

print(images[:, :3, :, :].shape)
print(images.shape)

gc.collect()
torch.cuda.empty_cache()


# ########################### Model definition ###########################
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:  # torch.Size([1, 196, 768])
            x = attn(x) + x
            x = ff(x) + x
        return x


class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, stride=2, padding=3, pool_kernel_size=3, pool_stride=2,
                 pool_padding=1):
        super(ConvEmbed, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            # Rearrange('b d h w -> b (h w) d')
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=pool_kernel_size, p2=pool_kernel_size)
        )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.conv_layers(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CompactTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1, dim=256, depth=12, heads=12, pool='cls',
                 in_channels=4, dim_head=64, dropout=0.1, emb_dropout=0.1, scale_dim=4, conv_embed=False):
        super().__init__()
        # dim was 768
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        if conv_embed:
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length()
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pool = nn.Linear(dim, num_classes)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        # self.sigmoid = nn.Sigmoid()  # Add sigmoid activation for binary segmentation instead of MLP

        # Additional convolution layer to convert the output shape to [1, 224, 224]
        self.output_conv = nn.Conv2d(dim, 1, kernel_size=1, bias=False)  # Set bias to False
        self.apply(self.init_weight)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # [1, 196, 768]
        b, n, _ = x.shape
        # 196 patches # 14 patches per side (224/16).
        # x += self.pos_embedding[:, :(n + 1)]  # [1, 196, 768]
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        g = self.pool(x)

        xl = torch.softmax(g, dim=1)
        x = torch.einsum('b n l, b n d -> b l d', xl, x)  # [1, 1, 768]
        # x = rearrange(x, 'b (h w) d -> b d h w', h=int(n ** 0.5))
        x = rearrange(x, 'b d (h w) -> b d h w', h=16)  # returns [1, 1, 16, 16]

        return self.output_conv(x)
        #
        # # xl = F.softmax(g, dim=1)
        # xl = self.sigmoid(g)  # Apply sigmoid activation instead of softmax
        # x = einsum('b n l, b n d -> b l d', xl, x)  # [1, 1, 768]
        #
        # # Reshape output to [1, 224, 224] using a 1x1 convolution
        # x = self.output_conv(x.unsqueeze(-2))
        #
        # # return self.mlp_head(x.squeeze(-2))
        # return x.squeeze(1)  # Output the segmentation mask instead of mlp_head?

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


# ######################### CS Unet #########################
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint


class Mlp(nn.Module):
    def __init__(self, dim, drop_path=0.2, layer_scale_init_value=0.7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv 7,3  5,2  3,1
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # B, H, W, C
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + self.drop_path(x)  # (N, H, W, C)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.conv_proj_q = self._build_projection(dim, kernel_size=3, stride=1, padding=1)
        self.conv_proj_k = self._build_projection(dim, kernel_size=3, stride=1, padding=1)
        self.conv_proj_v = self._build_projection(dim, kernel_size=3, stride=1, padding=1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def _build_projection(self, dim_in, kernel_size=3, stride=1, padding=1):
        proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim_in))
        return proj

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        Mh = int(N ** .5)
        x = x.view(B_, Mh, Mh, C).permute(0, 3, 1, 2)  # [batch_size*num_windows, Mh, Mw, total_embed_dim]
        # when we use conv the shape should be B, C, H, W. so use permute
        q = self.conv_proj_q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                            3)  # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        k = self.conv_proj_k(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.conv_proj_v(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @:multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]


        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B_, C, Mh, Mh)
        x = self.proj(x)
        x = x.reshape(B_, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class ConvSwinTransformerBlock(nn.Module):
    r""" Conv Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(dim=dim, drop_path=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-CMSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x.view(B, H, W, C)
        x = self.mlp(x)
        x = x.view(B, H * W, C)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.up = nn.Sequential(nn.ConvTranspose2d(dim, dim // dim_scale, kernel_size=2, stride=2), nn.GELU())
        self.norm = norm_layer(dim)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, x):
        """
        x: B, H*W, C → B, H*2*W*2, C/2
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.up(x)
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C // 2)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C → B, H*4*W*4, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic convolutional Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ConvSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"




class BasicLayer_up(nn.Module):
    """ A basic Convolutional Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            ConvSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, window_size=window_size,
                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1), nn.GELU(),
                                  nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=2, padding=1),
                                  nn.GELU(),
                                  Rearrange('b c h w -> b h w c'),
                                  norm_layer(embed_dim // 2),
                                  Rearrange('b h w c -> b c h w'),
                                  nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1), nn.GELU(),
                                  nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1), nn.GELU())
        if norm_layer is not None:
            self.norm = norm_layer(in_chans)
            self.norm2 = norm_layer(embed_dim)
        else:
            self.norm = None
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.drop(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm2(x)
        return x


class ConvSwinTransformerSys(nn.Module):
    """
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 3, 3, 3],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "ConvSwinTransformerSys expand initial----depths:{};depths_decoder:{};num_heads=:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, num_heads, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_cov = self.up = nn.Sequential(Rearrange('b (h w) c -> b c h w', h=patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)), w=patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                                 nn.Conv2d(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                           int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                           kernel_size=3, stride=1, padding=1), nn.GELU(),
                                                 nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                           int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                           kernel_size=3, stride=1, padding=1), nn.GELU(),
                                                 nn.Dropout(p=0.2),
                                                 Rearrange('b c h w -> b (h w) c', h=patches_resolution[0] // (
                                                         2 ** (self.num_layers - 1 - i_layer)),
                                                           w=patches_resolution[1] // (
                                                                   2 ** (self.num_layers - 1 - i_layer))))
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_cov)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x


class CS_Unet(nn.Module):
    def  __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(CS_Unet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        # self.config = config

        self.CS_Unet = ConvSwinTransformerSys(img_size=img_size,  #config.DATA.IMG_SIZE,
                                              patch_size=4,  #config.MODEL.SWIN.PATCH_SIZE,
                                              in_chans=3,  #config.MODEL.SWIN.IN_CHANS,
                                              num_classes=self.num_classes,
                                              embed_dim=96,  #config.MODEL.SWIN.EMBED_DIM,
                                              depths=[2, 2, 6, 2],  #config.MODEL.SWIN.DEPTHS,
                                              num_heads=[3, 6, 12, 24],  #config.MODEL.SWIN.NUM_HEADS,
                                              window_size=7,  #config.MODEL.SWIN.WINDOW_SIZE,
                                              mlp_ratio=4.,  #config.MODEL.SWIN.MLP_RATIO,
                                              qkv_bias=True,  #config.MODEL.SWIN.QKV_BIAS,
                                              qk_scale=None,  # config.MODEL.SWIN.QK_SCALE,
                                              drop_rate=0.0,  #config.MODEL.DROP_RATE,
                                              drop_path_rate=0.1,  #config.MODEL.DROP_PATH_RATE,
                                              ape=False,  # config.MODEL.SWIN.APE,
                                              patch_norm=True,  #config.MODEL.SWIN.PATCH_NORM,
                                              use_checkpoint=False)  #config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.CS_Unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                # print(k)
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.CS_Unet.load_state_dict(pretrained_dict,strict=False)
                print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

            model_dict = self.CS_Unet.state_dict()
            # print(self.swin_unet)
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.CS_Unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


# ########################### Call Model ###########################
# FCTSegmentationModel(in_channels=4, d_model=224, nhead=4, num_layers=6, dim_feedforward=1024, dropout=0.1).to(DEVICE)
# model = CompactTransformer().to(DEVICE)  https://github.com/kathyliu579/CS-Unet/blob/main/config.py
model = CS_Unet(img_size=224, num_classes=2, zero_head=False, vis=True)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters in CVT: %.3fM' % parameters)

output = model(images)
output.shape
