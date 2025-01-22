import os
import json
import numpy as np

def _calculate_sizes(found_p: str, task_id: str) -> dict:
  with open(f'{found_p}/exp/{task_id}/fit_00_kps.json', 'r') as f:
    keypoints = json.load(f)

  big_toe = np.array(keypoints['big toe'])
  heel = np.array(keypoints['heel'])
  lower_heel = np.array(keypoints['lower heel'])

  heel_diff = np.linalg.norm(big_toe[0:2] - heel[0:2])
  lower_heel_diff = np.linalg.norm(big_toe[0:2] - lower_heel[0:2])
  
  return {'heel': heel_diff, 'lower_heel': lower_heel_diff}

def found_process(found_p: str, task_id: str) -> dict:
  os.system(f'python3 {found_p}/fit.py --cfg example-cfg.yaml --device cuda')

  return _calculate_sizes(found_p, task_id)