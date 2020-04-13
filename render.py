import argparse
import bpy
import gin
import numpy as np
import os
import sys
import time

import paths, util

C = bpy.context
D = bpy.data

MAX_OBJS = 20

color_ref = None
obj_ref = [None for _ in range(MAX_OBJS)]
obj_classes = None
cached_obj_scale = [None for _ in range(MAX_OBJS)]


@gin.configurable('rendering')
def config_rendering(resolution=128, renderer='cycles', render_samples=32,
                     use_gpu=False, render_exr=False):
  """Adjust rendering settings.

  Args:
    resolution: Integer for image resolution (always square)
    renderer: Either 'cycles' or 'eevee'
    render_samples: Integer that determines sample quality, rendering time
    use_gpu: Whether to use the GPU for rendering (never tested)
    render_exr: Set true to output segmentation and depth ground truth
  """
  if renderer == 'eevee':
    C.scene.eevee.taa_render_samples = render_samples
  elif renderer == 'cycles':
    C.scene.render.engine = 'CYCLES'
    C.scene.cycles.device = 'GPU' if use_gpu else 'CPU'
    C.scene.cycles.samples = render_samples
    C.window.view_layer.cycles.use_denoising = True

  C.scene.render.resolution_x = resolution
  C.scene.render.resolution_y = resolution

  if render_exr:
    C.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    C.scene.render.image_settings.color_mode = 'RGBA'
    C.scene.render.image_settings.color_depth = '32'
    C.window.view_layer.use_pass_object_index = True
    C.window.view_layer.use_pass_material_index = True
  else:
    C.scene.render.image_settings.color_mode = 'RGB'


@gin.configurable('scene')
def init_scene():
  """Clear and reset scene."""
  camera_pos = (1, 0, .4)
  camera_angle = (np.pi * .36, 0, np.pi / 2)
  light_pos = (0, -.75, 1.35)

  # Delete everything
  bpy.ops.object.select_all(action='SELECT')
  bpy.ops.object.delete(confirm=False)

  # Initialize empty materials
  while len(D.materials) < (MAX_OBJS + 1):
    bpy.ops.material.new()

  # Camera
  bpy.ops.object.camera_add(location=camera_pos,
                            rotation=camera_angle)
  C.scene.camera = D.objects[0]

  # Lights
  bpy.ops.object.light_add(type='POINT', location=light_pos)
  fg_light = C.selected_objects[0]
  fg_light.data.energy = 100
  fg_light.data.shadow_soft_size = 1

  bpy.ops.object.light_add(type='SUN', location=[-2*v for v in camera_pos])
  bg_light = C.selected_objects[0]
  bg_light.data.energy = 1

  # Background plane
  bpy.ops.mesh.primitive_plane_add(size=20,
                                   location=[-3*v for v in camera_pos],
                                   rotation=camera_angle)
  bg_plane = C.selected_objects[0]
  bg_plane.pass_index = 0
  bg_plane.active_material = D.materials[0]


def get_tex_node(material_idx):
  """Returns Image Texture node, creates one if it doesn't exist."""
  nt = D.materials[material_idx].node_tree

  # Check whether the Image Texture node has been added
  img_node = nt.nodes.get('Image Texture')
  mix_node = nt.nodes.get('Mix Shader')
  aux_node = nt.nodes.get('Principled BSDF.001')

  if img_node is None:
    # Create new node for linking image
    nt.nodes.new('ShaderNodeTexImage')
    img_node = nt.nodes.get('Image Texture')

    # Link to main node
    main_node = nt.nodes.get('Principled BSDF')
    nt.links.new(img_node.outputs.get('Color'),
                 main_node.inputs.get('Base Color'))

    # Set up nodes for mixing in color
    nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.nodes.new('ShaderNodeMixShader')

    aux_node = nt.nodes.get('Principled BSDF.001')
    mix_node = nt.nodes.get('Mix Shader')
    out_node = nt.nodes.get('Material Output')

    # Link everything appropriately
    nt.links.new(main_node.outputs.get('BSDF'),
                 mix_node.inputs[1])
    nt.links.new(aux_node.outputs.get('BSDF'),
                 mix_node.inputs[2])
    nt.links.new(mix_node.outputs.get('Shader'),
                 out_node.inputs.get('Surface'))

    # For now set fraction to 0 to default to texture
    mix_node.inputs[0].default_value = 0

  return img_node, mix_node, aux_node


def get_emission_node(material_idx):
  """Returns Emission node, creates one if it doesn't exist."""
  nt = D.materials[material_idx].node_tree

  # Check whether the Emission node has been added
  img_node = nt.nodes.get('Emission')
  if img_node is None:
    # Create new node for linking image
    nt.nodes.new('ShaderNodeEmission')
    img_node = nt.nodes.get('Emission')

    # Link to main node
    main_node = nt.nodes.get('Material Output')
    nt.links.new(img_node.outputs.get('Emission'),
                 main_node.inputs.get('Surface'))

  return img_node


def initialize_color_ref():
  global color_ref
  out_dir = gin.query_parameter('sample.out_dir')

  # Set up color reference
  color_ref_path = f'{paths.DATA_DIR}/{out_dir}/color_reference.npy'
  while color_ref is None:
    try:
      color_ref = np.load(color_ref_path)
    except:
      if not os.path.exists(color_ref_path):
        try: os.makedirs(f'{paths.DATA_DIR}/{out_dir}')
        except FileExistsError: pass
        np.save(color_ref_path, np.random.rand(10,10,3))


@gin.configurable('material')
def setup_materials(class_idxs, use_texture=True, use_emission=False,
                    use_color_ref=True, color_variation=.2,
                    color_intensity=.4, num_modes=1):
  """Set up materials.

  Args:
    class_idxs: Reference for number of objects and their class
    use_texture: Whether or not to apply a texture image
    use_emission: For a completely flat, shadowless image
    use_color_ref: Decides whether to class-condition color choice
    color_variation: Spread from hue value provided in color_ref
    color_intensity: Trade-off between color and texture image
    num_modes: Support for multiple hues associated with each class
  """
  if isinstance(class_idxs, int):
    class_idxs = [class_idxs]

  n_txts = len(class_idxs) + 1
  fg_colors = []

  for class_idx in class_idxs:
    if use_color_ref:
      c = color_ref[class_idx, np.random.randint(num_modes)]
      fg_color = np.array([c[0], c[1]*.8 + .2, c[2]*.4 + .6])
      rnd_offset = np.random.randn(3) * color_variation
      rnd_offset[0] *= .3
      fg_color += rnd_offset
      fg_color[0] = fg_color[0] % 1
      fg_color = np.array(list(util.hsv2rgb(fg_color.clip(0,1))) + [1])
    else:
      fg_color = np.random.rand(4)
      fg_color[3] = 1

    fg_color[1:3] = fg_color[0]
    fg_colors += [fg_color]

  if use_texture:
    # Choose random textures
    tex_img_paths = np.random.choice(util.dtd_files, n_txts, replace=False)
    tex_img_paths = [util.dtd_img_dir + t for t in tex_img_paths]

    for i in range(n_txts):
      # Load texture images
      D.images.load(tex_img_paths[i])
      img_choice = tex_img_paths[i].split('/')[-1]

      # Update material node
      tex_node, mix_node, aux_node = get_tex_node(i)
      tex_node.image = D.images.get(img_choice)

      if i > 0 and use_color_ref:
        # Only update foreground objects
        aux_node.inputs[0].default_value = fg_colors[i-1]
        mix_node.inputs[0].default_value = color_intensity

  else:
    for i in range(n_txts):
      # Choose colors
      if i > 0:
        new_color = fg_colors[i-1]
      else:
        new_color = np.random.rand(4)
        new_color[1:3] = new_color[0]
        new_color[3] = 1

      if use_emission:
        node = get_emission_node(i)
      else:
        nt = D.materials[i].node_tree
        node = nt.nodes.get('Principled BSDF')

      node.inputs[0].default_value = new_color

  for i in range(n_txts):
    if i == 0:
      D.materials[i].pass_index = 99
    else:
      D.materials[i].pass_index = class_idxs[i-1]


@gin.configurable('lighting')
def setup_lighting(r=0):
  """Position light (possibly randomly)."""
  random_light_ang = -.5 + np.random.randn() * r
  for i in range(len(D.objects)):
    if 'Point' in D.objects[i].name:
      light_pos = (0, 1.5 * np.sin(random_light_ang), 1.5 * np.cos(random_light_ang))
      D.objects[i].location = light_pos

  return light_pos


@gin.configurable('object', blacklist=['obj_idx', 'class_idx', 'sample_idx', 'reset_obj'])
def setup_object(obj_idx, class_idx, sample_idx,
                 reset_obj=True, total_num_samples=60000, n_buckets=12,
                 random_scaling=0, random_viewpoint=0, random_warp=.1):
  """Load object model, update its material, and place appropriately."""
  global cached_obj_scale, obj_ref

  if reset_obj:
    # Distribute models across sample indices (for clean train/valid splits)
    per_bucket = util.n_models[class_idx] // n_buckets
    bucket_choice = sample_idx // (total_num_samples // n_buckets)
    min_idx = bucket_choice * per_bucket

    # Load object
    obj_path = None
    while obj_path is None:
      try:
        model_idx = np.random.randint(per_bucket) + min_idx
        obj_path = util.get_model_path(class_idx, model_idx)
        bpy.ops.import_scene.obj(filepath=obj_path)
      except:
        obj_path = None

    obj = C.selected_objects[0]
    obj.pass_index = obj_idx + 1
    C.view_layer.objects.active = obj
    obj_ref[obj_idx] = obj

    # Remap UV coordinates
    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.cube_project()
    bpy.ops.object.editmode_toggle()

    # Assign material
    for i in range(len(C.object.material_slots)):
        C.object.active_material_index = i
        C.object.active_material = D.materials[obj_idx + 1]

    cached_obj_scale[obj_idx] = None

  C.object.rotation_euler = (0, 0, 0)
  C.object.location = -np.array(C.object.bound_box).mean(0)
  bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

  if cached_obj_scale[obj_idx] is None:
    C.object.scale *= (.7 / max(C.object.dimensions))
    cached_obj_scale[obj_idx] = [C.object.scale[i] for i in range(3)]
  else:
    C.object.scale = cached_obj_scale[obj_idx]

  C.object.rotation_euler = (np.pi / 2, 0, -np.pi / 6)

  # Random scaling
  scale_factor = np.clip(np.random.randn() * random_scaling + .8, 0.1, 1.5)
  C.object.scale *= scale_factor

  # Slight warping of individual object scales
  warp_factor = np.random.randn(3)
  warp_factor -= warp_factor.mean() # Keep total scale roughly consistent
  warp_factor = np.clip(warp_factor * random_warp + 1, .5, 1.5)
  for i in range(3):
    C.object.scale[i] *= warp_factor[i]

  # Random viewpoint shift
  viewpoint_shift = np.random.randn(3) * random_viewpoint
  C.object.rotation_euler += viewpoint_shift


@gin.configurable('sample')
def render_sample(idx, num_objs=1, class_idxs=10, out_dir='tmp', do_reset=True):
  """Render a dataset sample."""
  global obj_ref, obj_classes

  if do_reset:
    init_scene()
    obj_classes = np.random.choice(class_idxs, size=num_objs)

  setup_materials(obj_classes)
  light_pos = setup_lighting()

  obj_data = []
  obj_locations = util.get_random_obj_locs(num_objs)

  # Initialize and arrange objects
  for o_idx in range(num_objs):
    setup_object(o_idx, obj_classes[o_idx], idx, reset_obj=do_reset)
    if num_objs > 1:
      # Move objects if there is more than one in scene
      obj_ref[o_idx].location = obj_locations[o_idx]

    obj_data += [util.get_obj_metadata(obj_ref[o_idx],
                                       obj_classes[o_idx],
                                       light_pos)]

  # Render
  try:
    use_exr = gin.query_parameter('rendering.render_exr')
  except ValueError:
    use_exr = False

  out_path = f'{paths.DATA_DIR}/{out_dir}/img/{idx:05d}'
  C.scene.render.filepath = out_path + ('.exr' if use_exr else '.png')
  bpy.ops.render.render(write_still=True)

  # Save sample metadata
  np.save(out_path + '.npy', np.stack(obj_data, 0))

  # Cleanup
  for d in [D.materials, D.textures, D.images, D.meshes]:
    util.cleanup_unused(d)


@gin.configurable('batch')
def render_batch(num_samples, idx_offset=0, cache_obj_frames=10):
  """Render a set of samples.

  Args:
    num_samples: Number of images to render
    idx_offset: Starting index of samples
    cache_obj_frames: Number of consecutive images to reuse object models
  """
  with util.Suppress():
    config_rendering()

    for idx in range(idx_offset, idx_offset + num_samples):
      render_sample(idx, do_reset=(idx == idx_offset) or
                                  (idx % cache_obj_frames == 0))


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset_name', type=str, default='tmp')
  parser.add_argument('-r', '--is_ray_process', action='store_true')
  parser.add_argument('-g', '--gin_config', nargs='+', default=['render_single'],
    help='Set of config files for gin (separated by spaces) '
         'e.g. --gin_config f1 f2 (exclude .gin from path)')
  parser.add_argument('-p', '--gin_param', nargs='+', default=[],
    help='Parameter settings that override config defaults '
         'e.g. --gin_param \'module_1.a = 2\' \'module_2.b = 3\'')

  argv = sys.argv[sys.argv.index("--") + 1:]
  flags = parser.parse_args(argv)

  if flags.is_ray_process:
    print("Starting ray process:")
    print(', '.join(flags.gin_param))
    sys.stdout.flush()
    util.suppress_output()

  # Parse config file
  gin_files = [f'{paths.CONFIG_DIR}/{g}.gin' for g in flags.gin_config]
  gin.parse_config_files_and_bindings(gin_files, flags.gin_param)
  with gin.unlock_config():
    gin.bind_parameter('sample.out_dir', flags.dataset_name)

  # Start rendering
  initialize_color_ref()
  render_batch()


if __name__ == '__main__':
  main()
