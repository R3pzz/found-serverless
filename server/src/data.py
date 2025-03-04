import os
import json
import logging
from io import BytesIO
from PIL import Image
from supabase import Client, create_client

from detail.config import IMAGE_SIZE
from detail.types import ARKitSource
from detail.process_arkit import process_arkit

logger = logging.getLogger(__name__)

SUPABASE_SOURCE_BUCKET_ID = os.environ.get('SUPABASE_SOURCE_BUCKET_ID', '') # 'found-serverless-source'

supabase: Client = None

# A utility function to replace an extension of a file path
def _replace_extension(filename: str, new_extension: str) -> str:
  return filename.rsplit('.', 1)[0] + new_extension

def _get_filename(path: str, replace_ext: str = None) -> str:
  filename = os.path.basename(path)

  if replace_ext is not None:
    filename = _replace_extension(filename, replace_ext)

  return filename

# A custom data loader function used to load ARKit files directly from the cloud storage
def _custom_arkit_loader(**kwargs) -> dict:
  content: list[tuple[str, str]] = kwargs['content']
  
  json_objects = {}
  for f, json_str in content:
    json_data = json.loads(json_str)

    img_name = _get_filename(f, '.jpg')
    json_objects[img_name] = json_data

  return json_objects

def init_cloud(url: str, key: str) -> None:
    logger.info("Initializing Supabase client")
    global supabase
    try:
        supabase = create_client(url, key)
        logger.info("Successfully initialized Supabase client")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        raise

def download_from_cloud(id: str) -> tuple[list, ARKitSource]:
    logger.info(f"Starting download from cloud for task ID: {id}")
    task_folder_path = f'tasks/{id}/'
    
    # Get all files in the bucket
    logger.debug(f"Listing files in bucket {SUPABASE_SOURCE_BUCKET_ID} at path: {task_folder_path}")
    bucket = supabase.storage.from_(SUPABASE_SOURCE_BUCKET_ID)
    try:
        files = bucket.list(task_folder_path)
        logger.debug(f"Found {len(files)} files in bucket")
    except Exception as e:
        logger.error(f"Failed to list files in bucket: {str(e)}")
        raise

    # Filter out image file names
    image_names = [
        f['name']
        for f in files
        if f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    logger.debug(f"Found {len(image_names)} image files")

    # Load all images
    images = []
    for f in image_names:
        logger.debug(f"Downloading and processing image: {f}")
        try:
            image_data = bucket.download(f)
            image = Image.open(BytesIO(image_data)).resize(IMAGE_SIZE)
            images.append(image)
            logger.debug(f"Successfully processed image: {f}")
        except Exception as e:
            logger.error(f"Failed to process image {f}: {str(e)}")
            raise
    
    # Get associated ARKit file names
    arkit_names = list(map(lambda f: _replace_extension(f, '.json'), image_names))
    logger.debug(f"Looking for {len(arkit_names)} ARKit files")

    # Load all ARKit files
    arkit_data_strings = []
    for f in arkit_names:
        logger.debug(f"Downloading ARKit file: {f}")
        try:
            arkit_json_content = bucket.download(f).decode('utf-8')
            arkit_data_strings.append((f, arkit_json_content))
            logger.debug(f"Successfully downloaded ARKit file: {f}")
        except Exception as e:
            logger.error(f"Failed to download ARKit file {f}: {str(e)}")
            raise

    logger.debug("Processing ARKit data")
    try:
        arkit_data = process_arkit('', _custom_arkit_loader, content=arkit_data_strings)
        logger.info(f"Successfully completed cloud download for task {id}")
        return images, arkit_data
    except Exception as e:
        logger.error(f"Failed to process ARKit data: {str(e)}")
        raise