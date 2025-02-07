import os
import json
import torch
import numpy as np
from collections import namedtuple

from detail.make_batch import make_batch
from detail.types import ARKitSource, Predictions

from FOUND.FOUND.model import FIND
from FOUND.FOUND.utils import Renderer
from FOUND.FOUND.utils.forward import batch_to_device, calc_losses, LOSS_KEYS

class FOUNDArgs:
  def __init__(self, **kwargs):
    self.root_p = kwargs['root_p']

    find_dir = kwargs['find_dir']
    self.find_weights_p = f'{self.root_p}/{find_dir}'

    self.device = kwargs['device']

    self.image_size = kwargs['image_size']
    self.disable_keypoints = True
    self.mesh_scale = 0.85

    # Loss weights(default settings)
    loss_defaults = dict(
      sil=1., norm=0.1, smooth=2e3,
		  kp=0.5, kp_l1=0.5, kp_l2=0.5, kp_nll=0.5,
		  edge=1., norm_nll=0.1, norm_al=0.1
    )
    
    for k, v in loss_weights:
      setattr(self, f'weight_{k}', v)

Stage = namedtuple('Stage', 'name num_epochs lr params losses')
STAGES = [
#	Stage('Registration', 50, .001, ['reg'], ['kp_nll']), // not implemented yet
	Stage('Deform verts', 250, .001, ['deform', 'reg'], ['sil', 'norm_nll']),
]

renderer: Renderer = None
model_args: FOUNDArgs = None
loss_weights: dict[float] = None

def _get_kps(kps: torch.Tensor, labels: list[str]) -> dict:
  obj = {}
  for kp, label in zip(kps[0], labels):
    kp = kp.cpu().detach().numpy().tolist()
    obj[label] = kp

def init_renderer(args: FOUNDArgs):
  # Instantiate the renderer
  global renderer
  renderer = Renderer(
      device=args.device,
      image_size=args.image_size,
      max_faces_per_bin=100000,
      cam_params=[]
  )

  # Parse the loss weights from the config file
  global loss_weights
  loss_weights = {k: getattr(args, f'weight_{k}') for k in LOSS_KEYS}

  global model_args
  model_args = args

def process(predictions: Predictions, source_arkit: ARKitSource):
  model = FIND(model_args.find_weights_p, kp_labels=None, opt_posevec=True)
  model.to(model_args.device)

  batch = make_batch(model_args, predictions, source_arkit)
  batch = batch_to_device(batch, model_args.device)

  for stage in STAGES:
    optimiser = torch.optim.Adam(model.get_params(stage.params), lr=stage.lr)

    render_normals = 'norm_nll' in stage.losses or 'norm_al' in stage.losses
    render_sil = 'sil' in stage.losses or render_normals

    for _ in range(stage.num_epochs):
      optimiser.zero_grad()

      # Generate new mesh
      new_mesh = model().scale_verts_(model_args.mesh_scale)

      # Render the normal map and extract the silhouette
      res = renderer(
        new_mesh,
        batch['R'], batch['T'], batch['pp'], batch['f'],
        normals_fmt='ours',
        render_rgb=False,
        render_normals=render_normals,
        render_sil=render_sil,
        keypoints=model.kps_from_mesh(new_mesh),
        mask_out_faces=model.get_mask_out_faces()
      )
      res['new_mesh'] = new_mesh

      # Calculate the total loss
      loss_dict = calc_losses(res, batch, stage.losses, {'img_size': model_args.image_size}, disable_keypoints=model_args.disable_keypoints)
      loss = sum(loss_dict[k] * loss_weights[k] for k in stage.losses)

      loss.backward()
      optimiser.step()

  mesh = model().scale_verts_(model_args.mesh_scale)
  kps = _get_kps(model.kps_from_mesh(mesh), model.kp_labels)
  
  return mesh, kps