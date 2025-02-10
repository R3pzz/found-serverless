import os
import uuid
import runpod

from supabase import Client, create_client

SUPABASE_URL = ''
SUPABASE_API_KEY = ''
SUPABASE_SOURCE_BUCKET_ID = 'found-serverless-source'

RUNPOD_API_KEY = ''
RUNPOD_ENDPOINT_ID = ''

IMAGE_FOLDER = ''

supabase: Client = None
serverless: runpod.Endpoint = None

def _get_extension(filename: str) -> str:
  return os.path.splitext(filename)[1]

def init_cloud(url: str, key: str) -> None:
  global supabase
  supabase = create_client(url, key)

def init_runpod(key: str, endpoint: str) -> None:
  runpod.api_key = key

  global serverless
  serverless = runpod.Endpoint(endpoint)

def create_unique_task_id() -> str:
  return str(uuid.uuid4())

def upload_data(images_p: str, id: str) -> None:
  # Get the SupaBase bucket
  bucket = supabase.storage.from_(SUPABASE_SOURCE_BUCKET_ID)
  
  task_folder_path = f'tasks/{id}/'

  # List source file names
  source_file_names = [
    filename.rsplit('.', 1)[0]
    for filename
    in os.listdir(images_p)
    if filename.endswith(('.json', '.png', '.jpg'))
  ]

  # Create image and arkit file names
  image_file_names = list(map(lambda x: f'{x}.jpg', source_file_names))
  arkit_file_names = list(map(lambda x: f'{x}.json', source_file_names))

  # Upload the files
  all_file_names = image_file_names + arkit_file_names
  for fname in all_file_names:
    fpath = os.path.join(images_p, fname)

    # Do something with the file...
    with open(fpath, 'rb') as f:
      bucket.upload(task_folder_path + fname, f)
      print(f'Uploaded {fname}')

def cleanup_data(id: str) -> None:
  # Get the SupaBase bucket
  bucket = supabase.storage.from_(SUPABASE_SOURCE_BUCKET_ID)

  task_folder_path = f'tasks/{id}/'
  bucket.remove(task_folder_path)

def run_endpoint(id: str) -> dict:
  try:
    run_request = serverless.run_sync(
      {
        'input': {
          'id': id,
        }
      },
      timeout=300,
    )

    print(run_request)
  except TimeoutError:
    print("Job timed out.")

if __name__ == "__main__":
  init_cloud(SUPABASE_URL, SUPABASE_API_KEY)
  init_runpod(RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID)

  id = create_unique_task_id()
  upload_data(IMAGE_FOLDER, id)
  run_endpoint(id)
  cleanup_data(id)