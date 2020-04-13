import argparse
import importlib
import numpy as np
import os
import ray
from ray import tune
import subprocess

import paths

flags = None

# Options for each dataset variation: (Texture, Color, Viewpoint, Lighting)
d_params = [
  [['material.use_texture = False'],
   ['material.use_texture = True', ]],
  [['material.use_color_ref = True', 'material.color_intensity = 0.4'],
   ['material.use_color_ref = False']],
  [['object.random_viewpoint = 0'],
   ['object.random_viewpoint = 0.5']],
  [['lighting.r = 0'],
   ['lighting.r = 2']],
]


def is_complete(config):
  """Check that all images in batch have been rendered.

  This is called to ensure that no samples are missed due to any odd errors that
  may pop up. Also guarantees that completed batches are not re-rendered if the
  distributed session needs to be restarted.
  """
  d = config['dataset_type']
  tmp_dir = f'{paths.DATA_DIR}/{flags.dataset_name}_{d}/img'
  offset = config['batch.idx_offset']
  n = config['batch.num_samples']
  file_suffix = 'exr' if flags.is_multi else 'png'

  try:
    tmp_files = os.listdir(tmp_dir)
  except:
    return False

  for i in range(offset, offset+n):
    if f'{i:05d}.{file_suffix}' not in tmp_files:
      return False

  return True


def run_batch_render(config):
  """Call blender and generate a batch of samples."""
  d = config['dataset_type']
  cmd = ['blender', '--background', '--python', 'render.py']
  config_file = 'render_multi' if flags.is_multi else 'render_single'
  cmd += ['--', '-r', '-d', flags.dataset_name + '_' + d, '-g', config_file, '-p']
  for k in config:
    if k != 'dataset_type':
      cmd += [f'{k} = {config[k]}']

  for p_idx, p_val in enumerate(d.split('-')):
    cmd += d_params[p_idx][int(p_val)]

  while not is_complete(config):
    with open(os.devnull, 'w') as f:
      subprocess.call(cmd, cwd=paths.PROJECT_DIR, stderr=f)

  tune.track.log(done=1)


def main():
  global flags

  # Command line args
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset_name', type=str, default='tmp')
  parser.add_argument('-q', '--quiet', action='store_true')
  parser.add_argument('-c', '--continue_exp', action='store_true')
  parser.add_argument('-l', '--local_mode', action='store_true')
  parser.add_argument('-a', '--redis_address', type=str, default=None)
  parser.add_argument('-n', '--num_samples', type=int, default=4800)
  parser.add_argument('-b', '--batch_size', type=int, default=40)
  parser.add_argument('--is_multi', action='store_true')
  parser.add_argument('--num_classes', type=int, default=10)
  flags = parser.parse_args()

  if not flags.is_multi:
    """
    A bit of a side-effect of how this code is put together requires that this
    holds in order to guarantee an equal number of samples per class are rendered
    and that the proper train/val/test split of models is guaranteed. (only
    applies to single-object setting)
    """
    num_batch, num_ref = flags.num_samples // flags.batch_size, flags.num_classes * 12
    assert num_batch % num_ref == 0, (
      'Number of batches should be multiple of (number of buckets (12) * number of classes): '
      f'(num batches: {num_batch}, num_buckets * num_classes: {num_ref})'
    )

  dataset_choices = [
    '0-0-0-0', '1-0-0-0', '0-1-0-0', '0-0-1-0',
    '0-0-0-1', '1-1-0-0', '0-1-1-0', '0-1-0-1',
    '0-1-1-1', '1-1-1-0', '1-0-1-1', '1-1-1-1',
  ]

  # Index offset reference for each batch job
  offsets = [i * flags.batch_size for i in range(flags.num_samples // flags.batch_size)]

  # Set up ray config
  ray_config = {
    'dataset_type': tune.grid_search(dataset_choices),
    'object.total_num_samples': flags.num_samples,
    'batch.num_samples': flags.batch_size,
    'batch.idx_offset': tune.grid_search(offsets),
  }

  if not flags.is_multi:
    class_fn = lambda spec: [(spec.config['batch.idx_offset'] // flags.batch_size) % flags.num_classes]
    ray_config['sample.class_idxs'] = tune.sample_from(class_fn)

  exp_args = {
    'verbose': True,
    'resume': False,
    'resources_per_trial': {'cpu': 2},
    'config': ray_config
  }

  if flags.quiet: exp_args['verbose'] = False
  if flags.continue_exp: exp_args['resume'] = True

  # Start Ray
  ray.init(local_mode=flags.local_mode,
           redis_address=flags.redis_address)

  trials = tune.run(run_batch_render, **exp_args)


if __name__ == '__main__':
  main()
