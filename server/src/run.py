import os
import sys
import torch
import runpod
import numpy as np
import logging
from datetime import datetime

from data import init_cloud, download_from_cloud
from detail.config import FOUND_IMAGE_SIZE

import found, snu, sam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# architecture:
# 1. client generates a unique task id
# 2. client uploads source files into a supabase bucket and puts them into an associated task id folder
# 3. client sends a request to the server with the associated task id
# 4. server processes the uploaded files and generates a result
# 5. server uploads the result into the supabase result bucket and puts it into an associated task id folder
# 6. server sends the response with the confirmation and the foot size
# 7. client downloads the results from the supabase result bucket
# 8. client performs the clean-up

# Current problems:
# 1. Task tracking is not implemented. We should create a database using SupaBase to
#    track the tasks which we're in progress of doing, which we completed, which we
#    dropped, etc..
# 2. Task ID is generated on the client, meaning that we can run into an ID collision.

FOUND_P = '/app/FOUND'
SNU_P = '/app/surface_normal_uncertainty'
SAM2_P = '/app/sam2'
FILE_STORAGE_ROOT = '/app/temp'

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_API_KEY = os.environ.get('SUPABASE_API_KEY', '')

def calc_size(kps: dict) -> float:
    logger.debug("Calculating foot size from keypoints")
    big_toe = np.array(kps['big toe'])
    heel = np.array(kps['heel'])

    # As the foot points to +X, the foot size is determined by
    # subtracting the X component of the farthest and the closest points.
    foot_size = big_toe[0] - heel[0]
    logger.debug(f"Calculated foot size: {foot_size}")
    return foot_size

def pipeline(id: str) -> float:
    logger.info(f"Starting pipeline for task ID: {id}")
    
    # Download files from the cloud storage
    logger.debug("Downloading files from cloud storage")
    source_images, source_arkit = download_from_cloud(id)
    logger.debug(f"Downloaded {len(source_images)} images and ARKit data")

    logger.debug("Processing with SNU model")
    predictions = snu.process(source_images)
    logger.debug("Processing with SAM model")
    predictions[:]['mask'] = sam.process(source_images)

    logger.debug("Processing with FOUND model")
    mesh, kps = found.process(predictions, source_arkit)
    logger.debug("FOUND model processing completed")

    # do something with mesh(upload it somewhere, idk..) and calculate the foot size
    return calc_size(kps)

def runpod_handler(event):
    logger.info("Received new RunPod event")
    id: str = event['input']['id']
    
    logger.debug(f'Processing task {id}')
    try:
        foot_size = pipeline(id)
        logger.debug(f"Successfully completed task {id} with foot size: {foot_size}")
        return {'status': 'completed', 'id': id, 'foot_size': foot_size}
    except Exception as e:
        logger.error(f"Error processing task {id}: {str(e)}", exc_info=True)
        return {'status': 'error', 'id': id, 'error': str(e)}

if __name__ == '__main__':
    logger.info("Initializing serverless environment")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.error("CUDA device not available")
        raise RuntimeError('cuda device not available')

    logger.info("Initializing cloud storage connection")
    init_cloud(SUPABASE_URL, SUPABASE_API_KEY)

    logger.info("Initializing SAM2 model")
    sam2_args = sam.SAM2Args(
        root_p=SAM2_P,
        weights_file_name='sam2.1_hiera_large.pt',
        cfg_p='sam2.1/sam2.1_hiera_l.yaml',
        device=device
    )
    sam.init_predictor(sam2_args)
    logger.info("SAM2 model initialized successfully")

    logger.info("Initializing SNU model")
    snu_args = snu.SNUArgs(
        root_p=SNU_P,
        weights_file_name='synfoot_10k_gn.pt',
        sampling_ratio=0.4,
        importance_ratio=0.7,
        device=device
    )
    snu.init_model(snu_args)
    logger.info("SNU model initialized successfully")

    logger.info("Initializing FOUND model")
    found_args = found.FOUNDArgs(
        root_p=FOUND_P,
        find_dir='data/find_nfap',
        device=device,
        image_size=FOUND_IMAGE_SIZE
    )
    found.init_renderer(found_args)
    logger.info("FOUND model initialized successfully")

    logger.info("Starting RunPod serverless handler")
    runpod.serverless.start({'handler': runpod_handler})