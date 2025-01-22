import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

SHOULD_FLIP_Z = True

IMAGE_SIZE = np.array([1440.0, 1920.0])
TARGET_IMAGE_SIZE = np.array([960.0, 1280.0])
DOWNSCALE_FACTOR = TARGET_IMAGE_SIZE / IMAGE_SIZE

# arkit to pytorch3d coordinate system transformation matrix
TRANSF_A2P = np.diag([-1., 1., -1.])

# helper constant rotation matrices
ROTATE_90_X = Rotation.from_euler('x', 90, degrees=True).as_matrix()
ROTATE_NEG90_Z = Rotation.from_euler('z', -90, degrees=True).as_matrix()
ROTATE_180_Z = Rotation.from_euler('z', 180, degrees=True).as_matrix()

def _load_arkit_data_from_file(path: str) -> object:
  arkit_data = {}

  for image in os.scandir(path):
    if image.name.endswith('.json'):
      with open(image.path, 'r') as f:
        json_data = json.load(f)

    image_name = image.name[:-5] + '.jpg'
    arkit_data[image_name] = json_data

  return arkit_data

def _serialize_arkit_data(data: object) -> object:
  json_obj = {
    'images': []
  }

  for pth, camera in data.items():
    json_obj['images'].append({
      'pth': pth,
      'R': camera['R'].tolist(), # world-to-camera rotation transform matrix
      'T': camera['T'].tolist(), # world-to-camera translation vector
      'pp': camera['pp'].tolist(), # downscaled principal point coordinates
      'f': camera['f'], # downscaled focal length
    })

def process_arkit_data(path: str) -> object:
  data = _load_arkit_data_from_file(path)

  for _, camera in data.items():
    # downscale pp and f
    camera['pp'] = np.array([camera['oy'], camera['ox']]) * DOWNSCALE_FACTOR
    camera['f'] = camera['fx'] * DOWNSCALE_FACTOR[0]

    # world center coordinate
    camera['C'] = np.array([
        camera['x'],
        camera['y'],
        camera['z']
    ])

    # move the cameras
    camera['C'] = ROTATE_90_X @ camera['C']
    if SHOULD_FLIP_Z:
      camera['C'] = ROTATE_180_Z @ camera['C']

    # calculate R
    camera['R'] = TRANSF_A2P @ Rotation.from_euler(
      'xyz',
      [
        camera['angleX'],
        camera['angleY'],
        camera['angleZ']
      ]
    ).as_matrix().T

    # flip the orientation from landscape to portrait
    camera['R'] = (camera['R'].T @ ROTATE_NEG90_Z).T

    # rotate the cameras
    camera['R'] = (ROTATE_90_X @ camera['R'].T).T
    if SHOULD_FLIP_Z:
      camera['R'] = (ROTATE_180_Z @ camera['R'].T).T

    # calculate T
    camera['T'] = (-camera['R'] @ camera['C'][:, None])[..., 0]

  return _serialize_arkit_data(data)