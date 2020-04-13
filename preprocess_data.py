import argparse
import imageio
import io
import numpy as np
import os
from PIL import Image
from skimage import color
import torch
from tqdm import tqdm
import h5py
import OpenEXR

from syn import paths, util


def merge_to_h5(data_dir, idx_ref, bucket_idx, n_buckets=12, is_exr=False):
  """Aggregate individual image files into single database.

  Rather than have hundreds of thousands of tiny files lying around, this
  puts all the raw png data into an HDF5 dataset along with any metadata
  associated with the sample.

  Args:
    data_dir: Directory in which the images were rendered
    idx_ref: Reference to reorganize the samples
    bucket_idx: Only process a subset of data as decided by the bucket index
    n_buckets: Number of files in which to split data
    is_exr: Whether or not to process EXR/PNG image files
  """
  dt = h5py.special_dtype(vlen=np.dtype('uint8'))
  idxs = np.split(idx_ref, n_buckets)[bucket_idx]
  metadata = []

  h5py_path = f'{data_dir}/imgs_{bucket_idx:02d}.h5'
  print(f'Building: {h5py_path}')
  with h5py.File(h5py_path, 'w') as f:
    ds = f.create_dataset('png_images', (len(idxs), ), dtype=dt)
    if is_exr:
      ds_out = f.create_dataset('png_targets', (len(idxs), ), dtype=dt)

    for i, idx in tqdm(enumerate(idxs)):
      sample_path = f'{data_dir}/img/{idx:05d}'
      metadata += [np.load(sample_path + '.npy')]

      if is_exr:
        # EXR data
        exr_data = OpenEXR.InputFile(sample_path + '.exr')
        rgb_img = util.prepare_rgb(exr_data)
        target_img = util.prepare_target(exr_data)

        # Write each image as PNG
        with io.BytesIO() as b:
          imageio.imwrite(b, rgb_img, format='png')
          b.seek(0)
          ds[i] = np.frombuffer(b.read(), dtype='uint8')

        with io.BytesIO() as b:
          imageio.imwrite(b, target_img, format='png')
          b.seek(0)
          ds_out[i] = np.frombuffer(b.read(), dtype='uint8')

      else:
        # PNG image
        with open(sample_path + '.png', 'rb') as f_img:
          ds[i] = np.frombuffer(f_img.read(), dtype='uint8')

    # Store all metadata
    metadata = np.stack(metadata).squeeze()
    for k in metadata.dtype.names:
      f[k] = metadata[k]

    if not is_exr:
      f['targets'] = f['class']


def preprocess_data(data_dir, flags, bucket_idx, n_buckets=12):
  """Do basic preprocessing of data.

  Assumes images have already been put into hdf5 datasets. Operations include
  image resizing, normalization, color conversion which usually add extra
  overhead during training.

  HDF5 doesn't support multiple writers, so can't parallelize this. Could
  instead save individual files, but that introduces extra step of merging files.
  """
  to_save = {}
  h5py_path = f'{data_dir}/imgs_{bucket_idx:02d}.h5'
  r = flags.res
  ds_suffix = ''
  normalize_vals = np.array([[98.12, 89.92, 84.36],
                             [43.95, 42.66, 44.05]])

  if flags.to_lab:
    # Changes for LAB images
    ds_suffix += '_lab'
    normalize_vals = np.array([[38.44,  3.03,  4.77],
                               [17.42,  9.41, 11.37]])

  print(f'Loading from: {h5py_path}')
  with h5py.File(h5py_path, 'r') as f:
    # Keep all metadata
    for k in f.keys():
      if not 'png' in k:
        to_save[k] = f[k][:]

    # Process images
    imgs = f['png_images']
    n_images = len(imgs)
    data = np.zeros((n_images, r, r, 3), np.float32)

    is_dense = 'png_targets' in f
    if is_dense:
      target_imgs = f['png_targets']
      targets = np.zeros((n_images, r, r, 3), np.uint8)

    for i in tqdm(range(n_images)):
      # Decode png bytes
      tmp_im = util.img_read_and_resize(imgs[i], r)
      if flags.to_lab:
        tmp_im = color.rgb2lab(tmp_im)

      data[i] = tmp_im
      if is_dense:
        targets[i] = util.img_read_and_resize(target_imgs[i], r)

    # Normalize images, convert to FP16
    data = (data - normalize_vals[0]) / normalize_vals[1]
    data = data.transpose(0,3,1,2)
    data = data.astype(np.float16)
    to_save['data'] = data

    if is_dense:
      targets = targets.transpose(0,3,1,2)
      to_save['targets'] = targets

  # Save to disk
  out_path = f'{data_dir}/data_{r}{ds_suffix}.h5'
  per_bucket = to_save['data'].shape[0]
  total_n_samples = n_buckets * per_bucket
  offset = bucket_idx * per_bucket

  if not os.path.exists(out_path):
    # Initializing dataset file
    print('Creating output file:', out_path)
    with h5py.File(out_path, 'w') as f:
      for k,v in to_save.items():
        tmp_shape = (total_n_samples, *v.shape[1:])
        print(k, tmp_shape)
        f.create_dataset(k, tmp_shape, dtype=v.dtype)

  print(f'Saving processed data to disk... ({out_path})')
  print(f'Index offset: {offset}')
  with h5py.File(out_path, 'a') as f:
    for k,v in to_save.items():
      print(k, v.shape)
      f[k][offset:offset+per_bucket] = v


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str, default='default')
  parser.add_argument('-b', '--num_buckets', type=int, default=12)
  parser.add_argument('-f', '--to_hdf5', action='store_true')
  parser.add_argument('-p', '--preprocess', action='store_true')
  parser.add_argument('-i', '--bucket_idxs', type=int, default=[], nargs='+')
  parser.add_argument('-r', '--res', type=int, default=128)
  parser.add_argument('-l', '--to_lab', action='store_true')
  parser.add_argument('-c', '--calc_mean', action='store_true')
  parser.add_argument('-e', '--is_exr', action='store_true')
  flags = parser.parse_args()

  data_dir = f'{paths.DATA_DIR}/{flags.dataset}'
  n_buckets = flags.num_buckets

  if flags.bucket_idxs == []:
    # If no bucket is selected, do them all
    flags.bucket_idxs = [i for i in range(n_buckets)]

  if flags.to_hdf5:
    idx_path = f'{data_dir}/idx_ref.pt'

    # Initialize index reference
    if not os.path.exists(idx_path):
      # Count total number of images
      n_samples = len(os.listdir(f'{data_dir}/img')) // 2
      print(f'{n_samples} samples total')

      # Prepare shuffled order of dataset
      idx_ref = util.get_random_mapping(n_samples, n_buckets)

      print(f'Saving index reference: {idx_path}')
      torch.save(idx_ref, idx_path)

    else:
      idx_ref = torch.load(idx_path)
      print(f'Loaded index reference: {idx_path}')

    # Create HDF5 files
    for i in flags.bucket_idxs:
      merge_to_h5(data_dir, idx_ref, i, n_buckets, flags.is_exr)

  if flags.preprocess:
    for i in flags.bucket_idxs:
      preprocess_data(data_dir, flags, i, n_buckets)

  if flags.calc_mean:
    mean, std = util.calculate_mean_std(data_dir, flags.res, flags.to_lab)
    print('Mean:', mean, 'Std:', std)


if __name__ == '__main__':
  main()
