import numpy as np

from Scripts.FCT import FCT
from Scripts.FullyConvolutionalTransformer import FullyConvolutionalTransformer
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ########################### Setup Data ###########################
images = torch.randn(1, 6, 224, 224)
images.to(DEVICE)

# ########################### Model definition ###########################
# Full: 1.945,214
# FCT: 27,435,980
in_channels = 6  # Number of input channels

full_model = FullyConvolutionalTransformer().to(DEVICE)
num_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"Number of Full model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, full_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable FULL Parameters in Full: %.3fM' % parameters)

fct_model = FCT(224, in_channels, 1, 1e-4, 1e-6, 1e-6).to(DEVICE)
num_params = sum(p.numel() for p in fct_model.parameters() if p.requires_grad)
print(f"Number of FCT model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, fct_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable FCT Parameters in FCT: %.3fM' % parameters)

# ########################### Call Model ###########################
full_output = full_model(images)  # Ask chat gpt about this error.
# RuntimeError: Given normalized_shape=[1], expected input with shape [*, 1], but got input of size[1, 224, 224, 6]
print(full_output[0].shape)
print(full_output[1].shape)
print(full_output[2].shape)

fct_output = fct_model(images)
# RuntimeError: Given normalized_shape=[1, 224, 224], expected input with shape [*, 1, 224, 224], but got input of size[1, 6, 224, 224]
print("This model returns a tuple")
print(fct_output[0].shape)

# ########################### Convolution Attempt ###########################
in_channels = 6  # Number of input channels
out_channels = 1  # Number of output channels (should be equal to the number of input channels for depthwise convolution)
kernel_size = 3  # Size of the kernel/filter
padding = 1  # Padding size

# depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1, stride=1)
# 3,3,3 depthwise
# BatchNorm
# Relu
# 1 x 1 conv
# BatchNorm
# Relu
depthwise_conv = nn.Sequential(
    nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
    # nn.BatchNorm2d()
    nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1)
)

depthwise_conv = nn.Sequential(
    nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=True, padding=1),
    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
)
# depthwise_conv = nn.Sequential(
#     nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=0),
#     nn.LeakyReLU(),
#     nn.MaxPool3d((2, 2, 2))
# )

num_params = sum(p.numel() for p in depthwise_conv.parameters() if p.requires_grad)
print(f"\nNumber of model params is: {num_params:,}")

input_tensor = torch.randn(2, in_channels, 224, 224)
print("Input shape")
print(input_tensor.shape)
print("OUtput shape")
output_tensor = depthwise_conv(input_tensor)
print(output_tensor.shape)

print("\nModel attempt")

model = nn.Sequential(
    nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
    nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
)
depthwise_conv1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
depthwise_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

# From example
# nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
# Max-pooling (select the highest value), Min-pooling (select the lowest value), Average-pooling
# nn.MaxPool3d(kernel_size=2, stride=2),

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
output_tensor = model(input_tensor)
print(output_tensor.shape)
