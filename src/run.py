import os
import sys
import runpod

from .arkit import process_arkit_data

# architecture:
# 1. client uploads their files onto a cloud storage
# 2. client sends a request to the server with a link to the uploaded files
# 3. server generates a unique request id and a temporary folder
# 4. server processes the uploaded files and generates a result
# 5. server uploads the result onto a cloud storage
# 6. server sends the response containing a link to the uploaded results
# 7. temporary folders are deleted after processing

DRIVE_ROOT = '/app/drive/MyDrive/Sizer'

FOUND_P = '/app/FOUND'
SNU_P = '/app/surface_normal_uncertainty'

IMAGES_P = f'{DRIVE_ROOT}/resources/'
RESULTS_P = f'{DRIVE_ROOT}/results/'

SNU = 'surface_normal_uncertainty'

def pipeline(event):
  pass

def runpod_handler(event):
  pass

if __name__ == '__main__':
  sys.path.append(f'/app')
  sys.path.append(f'/app/sam2')

  runpod.serverless.start({'handler': runpod_handler})