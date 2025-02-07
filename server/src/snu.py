import torch

from torchvision import transforms

from detail.types import Predictions

from surface_normal_uncertainty.models.NNET import NNET
from surface_normal_uncertainty.utils import utils

IMG_TRANSF = transforms.Compose([
  transforms.Resize(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
model_args = None

class SNUArgs:
  def __init__(self, **kwargs):
    self.root_p = kwargs['root_p']
    self.weights_file_name = kwargs['weights_file_name']

    # nnet parameters
    self.sampling_ratio = kwargs['sampling_ratio']
    self.importance_ratio = kwargs['importance_ratio']

    # deduced automatically given that the structure of the checkpoint name is <dataset>_<num_iters>_<arch>.pt
    weights_desc = self.weights_file_name.split('.')[0].split('_')
    if len(weights_desc) != 3:
      raise Exception('invalid checkpoint name format')
    
    self.architecture = weights_desc[2].toupper()
    if self.architecture not in ['BN', 'GN']:
      raise Exception('invalid architecture provided in the checkpoint name')
    
    self.device = kwargs['device']

def init_model(args: SNUArgs) -> None:
  global model_args
  model_args = model_args

  # load checkpoint
  weights_p = f'{args.root_p}/checkpoints/{args.weights_file_name}.pt'

  global model
  model = NNET(args).to(args.device)
  model = utils.load_checkpoint(weights_p, model)
  model.eval()

def process(source_images: list) -> Predictions:
  image_batch = torch.stack([IMG_TRANSF(img).to(model_args.device) for img in source_images]) # (B, 3, H, W)
  
  norm_out_list, _, _ = model(image_batch)
  norms = norm_out_list[-1].permute(0, 2, 3, 1) # (B, 3, H, W)

  return {'norm': norms, 'mask': []}