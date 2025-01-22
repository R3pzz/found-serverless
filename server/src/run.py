import os
import sys
import runpod

from .arkit import process_arkit_data
from .found import found_process
from .sam2 import sam2_process

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
FILE_STORAGE_ROOT = SNU_P + '/examples'

def pipeline(event):
  id = str(event['id'])
  
  # todo: implement file download logic
  
  images_p = os.path.join(FILE_STORAGE_ROOT, id)
  norm_unc_p = os.path.join(FILE_STORAGE_ROOT, id, 'results', 'norm_unc')

  process_arkit_data(images_p)
  sam2_process(images_p, norm_unc_p)
  found_process(FOUND_P, id)

def runpod_handler(event):
  pass

if __name__ == '__main__':
  sys.path.append(f'/app')
  sys.path.append(f'/app/sam2')

  runpod.serverless.start({'handler': runpod_handler})

# todo: all intermediate results and source images are stored in /app/surface_normal_uncertainty/examples/*id*
#       this is bad because SNU iterates over all of the folders and generates the results for them but and we
#       may have more than 1 temporary folder during a single run.
#       to mitigate this issue we should import SNU as a python library and feed the images directly rather
#       calling it via os.system. this logic should be implemented in the SNU itself and I will do it ASAP.
#       same applies for FOUND