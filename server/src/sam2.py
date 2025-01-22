import os
import torch
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List

CKPT = "./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

CENTER_POINT_POS = np.array([[500, 610]])
CENTER_POINT_LABEL = np.array([1])

def _load_images(source_p: str) -> list:
  images = []

  for image in os.scandir(source_p):
    if image.name.endswith('.jpg'):
      images.append((image.name[:-4], np.array(Image.open(image).resize((960, 1280)).convert("RGB"))))
  
  return images

model = SAM2ImagePredictor(build_sam2(MODEL_CFG, CKPT))

def sam2_process(source_p: str, dest_p: str) -> None:
  images = _load_images(source_p)

  with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for image_name, image in images:
      model.set_image(image)
      masks, scores, _ = model.predict(
        point_coords=CENTER_POINT_POS,
        point_labels=CENTER_POINT_LABEL,
        multimask_output=True,
      )

      sorted_ind = np.argsort(scores)[::-1]
      masks = masks[sorted_ind]
      scores = scores[sorted_ind]

      if os.path.exists(f'dest_p/{image_name}.png'):
        os.remove(f'dest_p/{image_name}.png')

      Image.fromarray(
        masks[0].astype(np.uint8) * 255
      ).convert("RGB").save(
        f'dest_p/{image_name}.png'
      )