import json
import numpy as np
import os
import sys

import paths

# ==============================================================================
# DTD and ShapeNet helper functions
# ==============================================================================

# DTD reference
dtd_img_dir = f'{paths.DTD_DIR}/images/'
with open(f'{paths.DTD_DIR}/labels/labels_joint_anno.txt', 'r') as f:
  dtd_files = [line.split(' ')[0] for line in f]


def random_texture_paths(n):
  tex_paths = np.random.choice(dtd_files, n, replace=False)
  return [dtd_img_dir + t for t in tex_paths]


# ShapeNet reference
data_dir = paths.SHAPENET_DIR
with open(f'{data_dir}/taxonomy.json', 'r') as f:
  taxonomy = json.load(f)

# 10 categories that we'll be using
cats = [
  'airplane,aeroplane,plane',
  'bench',
  'cabinet',
  'car,auto,automobile,machine,motorcar',
  'chair',
  'lamp',
  'sofa,couch,lounge',
  'table',
  'vessel,watercraft',
  'motorcycle,bike',
]

cat_dirs = []
for c in cats:
  for t in taxonomy:
    if t['name'] == c:
      cat_dirs += [t['synsetId']]

n_models = [len(os.listdir(f'{data_dir}/{c}')) for c in cat_dirs]


def get_model_path(class_idx, model_idx):
  tmp_class = cat_dirs[class_idx]
  ex_dirs = os.listdir(f'{data_dir}/{tmp_class}')
  tmp_ex = ex_dirs[model_idx]
  model_path = f'{data_dir}/{tmp_class}/{tmp_ex}/models/model_normalized.obj'
  return model_path


# ==============================================================================
# Misc helper functions
# ==============================================================================


class Suppress():
  def __enter__(self, logfile=os.devnull):
    open(logfile, 'w').close()
    self.old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

  def __exit__(self, type, value, traceback):
    os.close(1)
    os.dup(self.old)
    os.close(self.old)


def suppress_output():
  f = open(os.devnull,'w')
  sys.stdout = f
  sys.stderr = f


def cleanup_unused(d):
  """Blender garbage collection."""
  for d_ in d:
    if d_.users == 0:
      d.remove(d_)


def get_x_y(n, d=1.75, spacing=1.25):
  """Hacky heuristic to distribute multiple objects."""
  curr_pts = [np.array([0,0], np.float32)]
  for i in range(n-1):
    done = False
    attempts = 0
    while not done:
      done = True
      new_ang = np.random.rand() * 2 * np.pi
      new_dist = d + np.random.randn() * d / 10
      new_pt = curr_pts[-1] + np.array([np.cos(new_ang) * new_dist,
                                        np.sin(new_ang) * new_dist])
      for p in curr_pts:
        d = np.linalg.norm(new_pt - p)
        if d < spacing:
          done = False

      attempts += 1
      if attempts > 10:
        done = True

      if done:
        curr_pts += [new_pt]

  curr_pts = np.array(curr_pts)

  # Center at 0
  curr_pts -= curr_pts.mean(0)
  # Shuffle
  curr_pts = curr_pts[np.random.permutation(np.arange(len(curr_pts)))]

  return curr_pts


def get_random_obj_locs(n):
  locations = get_x_y(n) / 5
  locations = np.array([[l[0], l[1], np.random.randn()*.05]
                        for l in locations])
  locations += [.05, 0, 0.05]

  return locations


def get_obj_metadata(obj, class_idx, light_pos):
  eul = obj.rotation_euler
  quat = eul.to_quaternion()
  rot_mat = eul.to_matrix()
  bbox = np.array(obj.bound_box)
  bbox = np.stack([bbox.min(0), bbox.max(0)])

  return np.array((class_idx, rot_mat, quat, eul, bbox,
                   obj.scale, obj.location, light_pos),
                  dtype=[('class', int),
                         ('rot_mat', float, (3, 3)),
                         ('quaternion', float, 4),
                         ('euler', float, 3),
                         ('bbox', float, (2, 3)),
                         ('scale', float, 3),
                         ('location', float, 3),
                         ('light', float, 3)])


def rgb2hsv(rgb):
  max_idx = rgb.argmax()
  vals = np.roll(rgb, -max_idx)

  h = 0
  v = vals[0]
  s = 1 - vals.min() / v if v > 0 else 0

  if s and v:
    dv = (vals[2] - vals[1]) / (6 * s * v)
    h = (max_idx / 3 - dv) % 1

  return np.array([h, s, v])


def hsv2rgb(hsv):
  h, s, v = hsv
  vals = np.ones(3) * v
  vals[1:] *= (1 - s)

  if h > (5/6): h -= 1
  diffs = h - np.arange(3) / 3
  max_idx = np.abs(diffs).argmin()

  dv = diffs[max_idx] * 6 * s * v
  vals[1] += max(0, dv)
  vals[2] += max(0, -dv)

  return np.roll(vals, max_idx)


def to_srgb(v):
  thr = 0.0031308
  v_ = v.copy()
  v_[v <= thr] = v[v <= thr] * 12.92
  v_[v > thr] = 1.055 * (v[v > thr]**.41667) - 0.055
  return v_.clip(0, 1)


def read_channel(exr_data, c, dtype=np.float32):
  b = exr_data.header()['dataWindow']
  shape = [b.max.y - b.min.y + 1, b.max.x - b.min.x + 1]
  data = exr_data.channel(c)
  data = np.frombuffer(data, dtype=dtype).reshape(shape)
  return data


def prepare_rgb(exr_data):
  rgb = [read_channel(exr_data, f'View Layer.Combined.{c}')
         for c in ['R', 'G', 'B']]
  rgb = to_srgb(np.stack(rgb, 2))
  return (rgb * 255).astype(np.uint8)


def prepare_target(exr_data):
  ch = [read_channel(exr_data, f'View Layer.{c}')
        for c in ['IndexOB.X', 'IndexMA.X', 'Depth.Z']]
  ch[1] = (ch[1] + 1) % 100         # Rearrange semantic label (bg: 99 -> 0)
  ch[2] = ch[2].clip(0, 2.55) * 100 # Convert depth
  return np.stack(ch, 2).astype(np.uint8)


def img_read_and_resize(png_data, res=None):
  """Read raw PNG bytes, reshape to target resolution if needed."""

  import imageio
  import io
  from PIL import Image

  img = imageio.imread(io.BytesIO(png_data))
  if res is not None and res != img.shape[1]: # (assumes square images)
    img = Image.fromarray(img[:,:,:3]).resize([res,res])

  return img


def get_random_mapping(n_samples, n_buckets=12, split_idx=10):
  """Mix up samples across dataset.

  The dataset is rendered such that each "bucket" contains its own batch of
  models. This guarantees that the shape models presented at validation/test
  time have never been seen during training. This function mixes up samples
  such that training/validation/test splits of shapes are preserved.

  Args:
    n_samples: Number of rendered samples across entire dataset
    n_buckets: Number of buckets consisting of a unique set of shape models
    split_idx: Indicates the bucket where validation + testing samples start

  Returns:
    idx_ref: Mapping of samples to shuffled version of dataset.
  """

  idx_ref = np.arange(n_samples)
  per_bucket = n_samples // n_buckets
  n_training = split_idx * per_bucket
  idx_ref[:n_training] = np.random.permutation(idx_ref[:n_training])

  # Shuffle remaining bins (maintaining separationg between validation/testing)
  for i in range(split_idx, n_buckets):
    i0, i1 = i * per_bucket, (i+1) * per_bucket
    idx_ref[i0:i1] = np.random.permutation(idx_ref[i0:i1])

  return idx_ref


def calculate_mean_std(data_dir, res, is_lab):
  """Report dataset mean and standard deviation per image channel."""
  ds_suffix = '_lab' if is_lab else ''
  data_path = f'{data_dir}/data_{res}{ds_suffix}.h5'

  with h5py.File(data_path, 'r') as f:
    d = f['data'][:2000].float()
    d = d.transpose(1,0,2,3).reshape(3,-1)

  return d.mean(1), d.std(1)
