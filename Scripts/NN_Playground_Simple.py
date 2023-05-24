import numpy as np

from Scripts.FCT import FCT
from Scripts.FullyConvolutionalTransformer import FullyConvolutionalTransformer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ########################### Setup Data ###########################
images = torch.randn(1, 6, 224, 224)
images.to(DEVICE)

# ########################### Model definition ###########################
# Full: 1.945,214
# FCT: 27,435,980

full_model = FullyConvolutionalTransformer().to(DEVICE)
num_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, full_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters in Full: %.3fM' % parameters)

fct_model = FCT(224, 1e-4, 1e-6, 1e-6).to(DEVICE)
num_params = sum(p.numel() for p in fct_model.parameters() if p.requires_grad)
print(f"Number of model params is: {num_params:,}")
parameters = filter(lambda p: p.requires_grad, fct_model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters in FCT: %.3fM' % parameters)

# ########################### Call Model ###########################
# output = full_model(images)
# RuntimeError: Given normalized_shape=[1], expected input with shape [*, 1], but got input of size[1, 224, 224, 6]

output = fct_model(images)
# RuntimeError: Given normalized_shape=[1, 224, 224], expected input with shape [*, 1, 224, 224], but got input of size[1, 6, 224, 224]
output.shape
