from torch import Tensor

# A list of 'mask' (B, 1, H, W) and 'norm' (B, 3, H, W) predictions
Predictions = dict[Tensor, Tensor]
ARKitSource = dict