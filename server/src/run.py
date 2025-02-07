import os
import sys
import torch
import runpod
import numpy as np

from .arkit import load_arkit_from_dir
from .images import load_images_from_dir

from .detail.config import FOUND_IMAGE_SIZE

import found
import sam2
import snu

# architecture:
# 1. client uploads their files onto a cloud storage
# 2. client sends a request to the server with a link to the uploaded files
# 3. server generates a unique request id and a temporary folder
# 4. server processes the uploaded files and generates a result
# 5. server uploads the result onto a cloud storage
# 6. server sends the response containing a link to the uploaded results
# 7. temporary folders are deleted after processing

FOUND_P = '/app/FOUND'
SNU_P = '/app/surface_normal_uncertainty'
SAM2_P = '/app/sam2'
FILE_STORAGE_ROOT = '/app/temp'

def calc_size(kps: dict) -> float:
  big_toe = np.array(kps['big toe'])
  heel = np.array(kps['heel'])

  # As the foot points to +X, the foot size is determined by
  # subtracting the X component of the farthest and the closest points.
  foot_size = big_toe[0] - heel[0] 
  return foot_size

def pipeline(event) -> float:
  id = str(event['id'])
  
  # todo: implement file download logic
  images_p = os.path.join(FILE_STORAGE_ROOT, id)

  # load source images for this request as PIL.Image
  source_arkit = load_arkit_from_dir(images_p)
  source_images = load_images_from_dir(images_p)

  predictions = snu.process(source_images)
  predictions[:]['mask'] = sam2.process(source_images)

  mesh, kps = found.process(predictions, source_arkit)

  # do something with mesh(upload it somewhere, idk..) and calculate the foot size
  return calc_size(kps)
  
def runpod_handler(event):
  pass

if __name__ == '__main__':
  sys.path.append(f'/app')
  sys.path.append(f'/app/sam2')

  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    raise RuntimeError('cuda device not available')

  sam2_args = sam2.SAM2Args(
    root_p=SAM2_P,
    weights_file_name='sam2.1_hiera_large.pt',
    cfg_p='sam2.1/sam2.1_hiera_l.yaml',
    device=device
  )
  sam2.init_predictor(sam2_args)

  snu_args = snu.SNUArgs(
    root_p=SNU_P,
    weights_file_name='synfoot_10k_gn.pt',
    sampling_ratio=0.4,
    importance_ratio=0.7,
    device=device
  )
  snu.init_model(snu_args)

  found_args = found.FOUNDArgs(
    root_p=FOUND_P,
    find_dir='data/find_nfap',
    device=device,
    image_size=FOUND_IMAGE_SIZE
  )
  found.init_renderer(found_args)

  runpod.serverless.start({'handler': runpod_handler})