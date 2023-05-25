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
print(f"Number of model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, full_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters in Full: %.3fM' % parameters)

fct_model = FCT(224, in_channels, 1e-4, 1e-6, 1e-6).to(DEVICE)
num_params = sum(p.numel() for p in fct_model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, fct_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters in FCT: %.3fM' % parameters)

# ########################### Call Model ###########################
# full_output = full_model(images)
# # RuntimeError: Given normalized_shape=[1], expected input with shape [*, 1], but got input of size[1, 224, 224, 6]
# print(full_output.shape)

# fct_output = fct_model(images)
# RuntimeError: Given normalized_shape=[1, 224, 224], expected input with shape [*, 1, 224, 224], but got input of size[1, 6, 224, 224]
# print(fct_output.shape)

# ########################### Convolution Attempt ###########################
in_channels = 6  # Number of input channels
out_channels = 1  # Number of output channels (should be equal to the number of input channels for depthwise convolution)
kernel_size = 3  # Size of the kernel/filter
padding = 1  # Padding size

depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1, stride=1)
# depthwise_conv = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1)

num_params = sum(p.numel() for p in depthwise_conv.parameters() if p.requires_grad)
print(f"\nNumber of model params is: {num_params:,}")

input_tensor = torch.randn(2, in_channels, 224, 224)
print(input_tensor.shape)

output_tensor = depthwise_conv(input_tensor)
print(output_tensor.shape)

# output_tensor2 = depthwise_conv2(output_tensor)
# print(output_tensor2.shape)

print("\nModel attempt")

model = nn.Sequential(
    nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
    nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
)
depthwise_conv1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
depthwise_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

# From example
# nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
# nn.MaxPool3d(kernel_size=2, stride=2),

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
output_tensor = model(input_tensor)
print(output_tensor.shape)
