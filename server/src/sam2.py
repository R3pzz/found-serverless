import os
import torch
import numpy as np

from PIL import Image

from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

CENTER_POINT_POS = np.array([[500, 610]])
CENTER_POINT_LABEL = np.array([1])

model = None
model_args = None

class SAM2Args:
  def __init__(self, **kwargs):
    self.root_p = kwargs['root_p']

    weights_file_name = kwargs['weights_file_name']
    self.weights_p = f'{self.root_p}/checkpoints/{weights_file_name}'

    cfg_file_name = kwargs['cfg_file_name']
    self.cfg_p = f'{self.root_p}/configs/{cfg_file_name}'

    self.device = kwargs['device']

def init_predictor(args: SAM2Args) -> None:
  global model
  model = SAM2ImagePredictor(build_sam2(args.cfg_p, args.weights_p))

  global model_args
  model_args = args

def process(source_images: list) -> torch.Tensor:
  masks = []

  with torch.inference_mode(), torch.autocast(model_args.device.type, dtype=torch.bfloat16):
    for image in source_images:
      model.set_image(image)
      pred_masks, scores, _ = model.predict(
        point_coords=CENTER_POINT_POS,
        point_labels=CENTER_POINT_LABEL,
        multimask_output=True,
      )

      sorted_ind = np.argsort(scores)[::-1]
      pred_masks = pred_masks[sorted_ind] # (3, H, W)
      best_mask = pred_masks[0] # (H, W)
      
      masks.append(best_mask.unsqueeze(-1).astype(np.uint8) * 255) # (H, W, 1)

  return torch.stack(masks) # (B, H, W, 1)