# Synthetic dataset rendering


![](img/example_syn_imgs.png)

Framework for producing the synthetic datasets used in:

> **How Useful is Self-Supervised Pretraining for Visual Tasks?** <br/>
>   [Alejandro Newell](https://www.alejandronewell.com/) and [Jia Deng](https://www.cs.princeton.edu/~jiadeng/). CVPR, 2020.
[arXiv:2003.14323](https://arxiv.org/abs/2003.14323)

Experiment code can be found [here](https://www.github.com/princeton-vl/selfstudy).

This is a general purpose synthetic setting supporting single-object or multi-object images providing annotations for object classification, object pose estimation, segmentation, and depth estimation.

## Setup

Download and set up Blender 2.80 (this code has not been tested on more recent Blender versions).

Blender uses its own Python, to which we need to add an extra package. In the Blender installation, find the python directory and run:

```
cd path/to/blender/2.80/python/bin
./python3.7m -m ensure pip
./pip3 install gin_config
```

For distributed rendering and additional dataset prep, use your own Python installation (not the Blender version). Everything was tested with Python 3.7 and the following extra packages:

```
sudo apt install libopenexr-dev
pip install ray ray[tune] h5py openexr scikit-image
```

### External data

Download [ShapeNetCore.v2](https://www.shapenet.org/) and [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz).

By default, it is assumed external datasets will be placed in `syn_benchmark/datasets` (e.g. `syn_benchmark/datasets/ShapeNetCore.v2`). If this is not the case, change any paths as necessary in `paths.py`.

## Dataset Generation

Try a test run with:

`blender --background --python render.py -- -d test_dataset`

The argument `-d, --dataset_name` specifies the output directory which will be placed in the directory defined by `pahs.DATA_DIR`. Dataset settings can be modified either by selecting a gin config file (`-g`) or by modifying parameters (`-p`), for example:

```
blender --background --python render.py -- -g render_multi
blender --background --python render.py -- -p "material.use_texture = False" "object.random_viewpoint = 0"
blender --background --python render.py -- -g render_multi -p "batch.num_samples = 100"
```

Manual arguments passed in through `-p` will override those in the provided gin file. Please check out `config/render_single.gin` to see what options can be modified.

### Distributed rendering

To scale up dataset creation, rendering is split into smaller jobs that can be sent out to individual workers for parallelization on a single machine or on a cluster. The library [Ray](https://github.com/ray-project/ray) is used to manage workers automatically. This allows large-scale distributed, parallel processes which are easy to restart in case anything crashes.

Calling `python distributed_render.py` will by default produce small versions of the 12 single-object datasets used in the paper. Arguments are available to control the overall dataset size and to interface with Ray. The script can be modified as needed to produce individual datasets or to modify dataset properties (e.g. texture, lighting, etc).

To produce multi-object images with depth and segmentation ground truth, add the argument `--is_multi`.

![](img/example_multiobj_img.png)

### Further processing

After running the rendering script, you will be left with a large number of individual files containing rendered images and metadata pertaining to class labels and other scene information. Before running the main [experiment code](https://www.github.com/princeton-vl/selfstudy) it is important that this data is preprocessed.

There are two key steps:
- consolidation of raw data to HDF5 datasets: `python preprocess_data.py -d test_dataset -f`
- image resizing and preprocessing: `python preprocess_data.py -d test_dataset -p`

If working with EXR images produced for segmentation/depth data make sure to add the argument `-e`.

`-f, --to_hdf5`: The first step will move all image files and metadata into HDF5 dataset files.

An important step that occurs here is conversion of EXR data to PNG data. The EXR output from Blender contains both the rendered image and corresponding depth, instance segmentation, and semantic segmentation data. After running this script, the rendered image is stored as one PNG and the depth and segmentation channels are concatenated into another PNG image.

After this step, I recommend removing the original small files if disk space is a concern, all raw data is fully preserved in the `img_XX.h5` files. Note, the data is stored as an encoded PNG, if you want to read the image into Python you can do the following:

```
f = h5py.File('path/to/your/dataset/imgs_00.h5', 'r')
img_idx = 0
png_data = f['png_images'][img_idx]

img = imageio.imread(io.BytesIO(png_data))
# or alternatively
img = util.img_read_and_resize(png_data)
```

`-p, --preprocess`: Once the raw data has been moved into HDF5 files, it can be quickly processed for use in experiments. This preprocessing simply takes care of steps that would otherwise be performed over and over again during training such as image resizing and normalization. One of the more expensive steps that is taken care of here is conversion to LAB color space.

This preprocessing step prepares a single HDF5 file which ready to be used with the experiment code. Unlike the files created in the previous step, this data has been processed and some information may be lost from the original images especially if they have been resized to a lower resolution.
