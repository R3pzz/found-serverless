import torch
from torchvision import transforms

from types import ARKitSource, Predictions

def _kappa_to_alpha(mask: torch.Tensor) -> torch.Tensor:
  alpha = ((2 * mask) / ((mask ** 2.0) + 1)) \
    + ((torch.exp(- mask * torch.pi) * torch.pi) / (1 + torch.exp(- mask * torch.pi)))
  return torch.rad2deg(alpha)

def _determine_batch_size(predictions: Predictions) -> int:
  return max(len(predictions), 8)

def make_batch(args, predictions: Predictions, source_arkit: ARKitSource) -> list[dict]:
  assert predictions['norm'].shape[0] == len(source_arkit['filename'])

  # image_size has a shape of (W, H)
  resize_fac = args.image_size[1] / predictions['norm'].shape[1]
  downscale = transforms.Resize(args.image_size)

  # We want to downscale the source images and predictions for FOUND due to speed restrictions.
  norm_rgb = downscale(predictions['norm']) # (N, H, W, 3)
  norm_xyz = norm_rgb * 2 - 1 # (N, H, W, 3)

  mask = downscale(predictions['mask']) # (N, H, W, 1)
  mask_alpha = _kappa_to_alpha(mask) # (N, H, W, 1)
  sil = (mask_alpha < 70).float() # (N, H, W, 1)

  result = {
      'filename': source_arkit['filename'],

      # Camera parameters
      'R': source_arkit['R'],
      'T': source_arkit['T'],
      'pp': map(lambda pp: pp * resize_fac, source_arkit['pp']),
      'f': map(lambda f: f * resize_fac, source_arkit['f']),

      # Image data and predictions
      'norm_xyz': norm_xyz.to(args.device),
      'norm_kappa': mask.to(args.device),
      'sil': sil.to(args.device),
  }

  return result