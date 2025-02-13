import os
import json

from io import BytesIO
from PIL import Image
from supabase import Client, create_client

from .detail.config import IMAGE_SIZE
from .detail.types import ARKitSource
from .detail.process_arkit import process_arkit

SUPABASE_SOURCE_BUCKET_ID = 'found-serverless-source'

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
  global supabase
  supabase = create_client(url, key)

def download_from_cloud(id: str) -> tuple[list, ARKitSource]:
  task_folder_path = f'tasks/{id}/'
  
  # Get all files in the bucket
  bucket = supabase.storage.from_(SUPABASE_SOURCE_BUCKET_ID)
  files = bucket.list(task_folder_path)

  # Filter out image file names
  image_names = [
    f['name']
    for f in files
    if f['name'].lower().endswith(('.png', '.jpg', '.jpeg'))
  ]

  # Load all images
  images = []
  for f in image_names:
    image_data = bucket.download(f)
    image = Image.open(BytesIO(image_data)).resize(IMAGE_SIZE)

    images.append(image)
  
  # Get associated ARKit file names
  arkit_names = list(map(lambda f: _replace_extension(f, '.json'), image_names))

  # Load all ARKit files
  arkit_data_strings = []
  for f in arkit_names:
    arkit_json_content = bucket.download(f).decode('utf-8')
    arkit_data_strings.append((f, arkit_json_content))

  arkit_data = process_arkit('', _custom_arkit_loader, content=arkit_data_strings)

  return images, arkit_data