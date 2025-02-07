import os

from PIL import Image

def load_images_from_dir(dir: str, default_size: tuple[int, int] = (1440, 1920)) -> list:
  files = os.listdir(dir)
  result = []

  for f in files:
    if f.endswith(['.png', '.jpg', '.jpeg']):
      img_p = os.path.join(dir, f)
      image = Image.open(img_p).resize(default_size)

      result.append(image)
  
  return result