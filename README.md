# Dockerfile

Use with `docker build . -t found-serverless`
Run docker in repository folder.
Build and publish

# Setup Supabase
1. Create project
2. Put config to
  SUPABASE_URL = ''
  SUPABASE_API_KEY = ''
3. Bucket
  - Setup bucket via dashhboard
  - Put bucketId to SUPABASE_SOURCE_BUCKET_ID = 'found-serverless-source'
  - Put bucketId to run.py

# Runpod
1. https://www.runpod.io/console/serverless
  - [+] New Endpoint
    - https://www.runpod.io/console/serverless/new-endpoint
  - Docker Image
    - https://www.runpod.io/console/serverless/new-endpoint/custom
  - Container Image
    - Docker publish -
2. Get endpoint key and put to client/src/run.py
  - RUNPOD_API_KEY = ''
  - RUNPOD_ENDPOINT_ID = ''

# Launch
1. pip install runpod supabase
2. Run.py - put Image folder to folder with data
  - IMAGE_FOLDER = ''

3. Data files
  -
