"""
source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
"""

import numpy as np
import scipy.misc as misc
import tensorflow as tf
import matplotlib.pyplot as plt


DEFAULT_IMAGE_SIZE = 224

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
K5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def get_base_image(height=DEFAULT_IMAGE_SIZE, width=DEFAULT_IMAGE_SIZE, means=None):
  """
  Get base image for filter visualization. Gray image with gaussian noise.
  :param height:        Height of the base image.
  :param width:         Width of the base image.
  :param means:         Means to subtract from the image.
  :return:              Base image as a numpy Tensor.
  """

  background_color = np.float32([200.0, 200.0, 200.0])
  base_image = np.random.normal(background_color, 8, (height, width, 3))

  if means is not None:
    base_image -= means

  return base_image

def create_input_placeholder(means=None):
  """
  Creates input placeholder.
  :param means:         Means to subtract from the image.
  :return:              Input placeholder (one with reshaped for batches).
  """
  input_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="input")
  input_t = tf.expand_dims(input_pl, axis=0)

  if means is not None:
    input_t = input_t - means

  return input_pl, input_t

def calc_grad_tiled(image, t_grad, session, image_pl, tile_size=512, is_training_pl=None):
  """
  Compute the value of tensor t_grad over the image in a tiled way.
  Random shifts are applied to the image to blur tile boundaries over
  multiple iterations.
  :param image:           Image as base.
  :param t_grad:          Gradient to compute.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param tile_size:       Size of tile.
  :param is_training_pl:  Is training placeholder used to determine if the network si being trained.
  :return:                Computed gradient.

  """

  sz = tile_size
  h, w = image.shape[:2]
  sx, sy = np.random.randint(sz, size=2)
  image_shift = np.roll(np.roll(image, sx, 1), sy, 0)
  grad = np.zeros_like(image)

  for y in range(0, max(h - sz // 2, sz), sz):
    for x in range(0, max(w - sz // 2, sz), sz):
      sub = image_shift[y:y + sz, x:x + sz]

      feed_dict = {
        image_pl: sub
      }


      if is_training_pl is not None:
        feed_dict[is_training_pl] = False
      g = session.run(t_grad, feed_dict=feed_dict)
      grad[y:y + sz, x:x + sz] = g

  return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(objective, image_pl, session, resize_op, resize_image_pl, resize_shape_pl, iter_n=10, step=1.0,
                      octave_n=3, octave_scale=1.4, means=None, is_training_pl=None, base_image=None,
                      width=DEFAULT_IMAGE_SIZE, height=DEFAULT_IMAGE_SIZE):
  """
  Render multiscale optimization for selected objective.
  :param objective:       Objective to be optimized.
  :param image_pl:        Input placeholder tensor.
  :param session:         Tensorflow session.
  :param resize_op:       Resize operation.
  :param resize_image_pl: Resize operation input tensor.
  :param resize_shape_pl: Resize operation shape input tensor.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :param octave_n:        Number of octaves.
  :param octave_scale:    Scale for octave.
  :param means:           Means to be substracted from base image.
  :param is_training_pl:  Is training placeholder.
  :param base_image:      Base image to use (If none random image will be used).
  :param width:           Width of generated image.
  :param height:          Height of generated image.

  """

  # compute a scalar value to optimize and derive its gradient
  score = tf.reduce_mean(objective)
  gradient = tf.gradients(score, image_pl)[0]

  if base_image is None:
    image = get_base_image(width, height, means=means)
  else:
    image = base_image

  for octave in range(octave_n):

    if octave > 0:

      hw = np.int32(np.float32(image.shape[:2]) * octave_scale)

      feed_dict = {
        resize_image_pl: image,
        resize_shape_pl: hw
      }

      if is_training_pl is not None:
        feed_dict[is_training_pl] = False

      image = session.run(resize_op, feed_dict=feed_dict)

    for _ in range(iter_n):

      g = calc_grad_tiled(image, gradient, session, image_pl, is_training_pl=is_training_pl)
      # normalizing the gradient, so the same step size should work
      g /= g.std() + 1e-8
      image += g * step

  return image


def lap_split(img):
    """ Split the image into lo and hi frequency components. """
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, K5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, K5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    """ Build Laplacian pyramid with n splits. """
    levels = []
    for _ in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    """ Merge Laplacian pyramid. """
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, K5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    """ Normalize image by making its standard deviation = 1.0. """
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    """ Perform the Laplacian pyramid normalization. """
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def setup_resize():
  """
  Setup resize operation.
  :return:      Resize operation, Resize operation input tensor, Resize shape input tensor.

  """
  resize_image_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="resize_image_pl")
  resize_shape_pl = tf.placeholder(tf.int32, shape=(2,), name="resize_shape_pl")
  resize_op = tf.image.resize_bilinear(tf.expand_dims(resize_image_pl, 0), resize_shape_pl)[0, ...]

  return resize_op, resize_image_pl, resize_shape_pl

def setup_lapnorm(scale_n=4):
  """
  Setup laplace normalization.
  :param scale_n:       Number of scales.
  :return:              Laplace output, Laplace input tensor.

  """

  lapnorm_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="lapnorm_pl")
  lapnorm = lap_normalize(lapnorm_pl, scale_n=scale_n)

  return lapnorm, lapnorm_pl

def render_lapnorm(objective, session, image_pl, lap_norm, lap_norm_pl, resize_op, resize_image_pl,
                   resize_shape_pl, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, means=None,
                   is_training_pl=None, base_image=None, width=DEFAULT_IMAGE_SIZE, height=DEFAULT_IMAGE_SIZE):
  """
  Render laplace normalization for selected objective.
  :param objective:       Objective to be optimized.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param lap_norm:        Laplace normalization output.
  :param lap_norm_pl:     Laplace normalization input tensor.
  :param resize_op:       Resize operation.
  :param resize_image_pl: Resize operation input tensor.
  :param resize_shape_pl: Resize operation shape input tensor.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :param octave_n:        Number of octaves.
  :param octave_scale:    Scale for octave.
  :param means:           Means to be substracted from base image.
  :param is_training_pl:  Is training placeholder.
  :param base_image:      Base image to use (If none random image will be used).
  :param width:           Width of generated image.
  :param height:          Height of generated image.
  :return:                Generated image.

  """

  score = tf.reduce_mean(objective)
  gradient = tf.gradients(score, image_pl)[0]

  if base_image is None:
    image = get_base_image(width, height, means=means)
  else:
    image = base_image

  for octave in range(octave_n):
    if octave > 0:
      hw = np.int32(np.float32(image.shape[:2]) * octave_scale)

      feed_dict = {
        resize_image_pl: image,
        resize_shape_pl: hw
      }

      if is_training_pl is not None:
        feed_dict[is_training_pl] = False

      image = session.run(resize_op, feed_dict=feed_dict)

    for _ in range(iter_n):
      g = calc_grad_tiled(image, gradient, session, image_pl, is_training_pl=is_training_pl)
      feed_dict = {
        lap_norm_pl: g
      }

      if is_training_pl is not None:
        feed_dict[is_training_pl] = False

      g = session.run(lap_norm, feed_dict=feed_dict)
      image += g * step

  return image

def resize(image, size, resize_image_pl, resize_shape_pl, resize_op, session):
  """
  Performs resizing of image.
  :param image:           Image to resize.
  :param size:            Size of resized image.
  :param resize_image_pl: Resize operation input tensor.
  :param resize_shape_pl: Resize operation shape input tensor.
  :param resize_op:       Resize operation.
  :param session:         Tensorflow session.
  :return:                Resized image.

  """
  feed_dict = {
    resize_image_pl: image,
    resize_shape_pl: size,
  }

  return session.run(resize_op, feed_dict=feed_dict)

def render_deepdream(objective, session, image_pl, img0, resize_op, resize_image_pl, resize_shape_pl,
                    iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
  """
  Renders deep dream image.
  :param objective:       Objective to be optimized.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param img0:            Base image to use.
  :param resize_op:       Resize operation.
  :param resize_image_pl: Resize operation input tensor.
  :param resize_shape_pl: Resize operation shape input tensor.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :return:                Generated image.

  """

  t_score = tf.reduce_mean(objective)
  t_grad = tf.gradients(t_score, image_pl)[0]

  # split the image into a number of octaves
  img = img0
  octaves = []
  for _ in range(octave_n-1):
      hw = img.shape[:2]
      lo = resize(img, np.int32(np.float32(hw)/octave_scale), resize_image_pl, resize_shape_pl, resize_op, session)
      hi = img-resize(lo, hw, resize_image_pl, resize_shape_pl, resize_op, session)
      img = lo
      octaves.append(hi)

  
  # generate details octave by octave
  for octave in range(octave_n):
      if octave>0:
          hi = octaves[-octave]
          img = resize(img, hi.shape[:2], resize_image_pl, resize_shape_pl, resize_op, session)+hi
      for _ in range(iter_n):
          g = calc_grad_tiled(img, t_grad, session, image_pl)
          img += g*(step / (np.abs(g).mean()+1e-7))
      
  return img

def render_image_lapnorm(objective, session, image_pl,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, means=None,
                   is_training_pl=None, base_image=None):
  """
  Render laplace normalization for selected objective.
  :param objective:       Objective to be optimized.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :param octave_n:        Number of octaves.
  :param octave_scale:    Scale for octave.
  :param means:           Means to be substracted from base image.
  :param is_training_pl:  Is training placeholder.
  :param base_image:      Base image to use (If none random image will be used).
  :return:                Generated image.

  """

  lapnorm, lapnorm_pl = setup_lapnorm()
  resize_op, resize_image_pl, resize_shape_pl = setup_resize()

  return render_lapnorm(objective, session, image_pl, lapnorm, lapnorm_pl, resize_op, resize_image_pl, 
                        resize_shape_pl, iter_n, step, octave_n, octave_scale, means, is_training_pl, base_image)


def render_image_multiscale(objective, session, image_pl, iter_n=10, step=1.0,
                      octave_n=3, octave_scale=1.4, means=None, is_training_pl=None, base_image=None):
  """
  Render multiscale optimization for selected objective.
  :param objective:       Objective to be optimized.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :param octave_n:        Number of octaves.
  :param octave_scale:    Scale for octave.
  :param means:           Means to be substracted from base image.
  :param is_training_pl:  Is training placeholder.
  :param base_image:      Base image to use (If none random image will be used).
  :return:                Generated image.

  """

  resize_op, resize_image_pl, resize_shape_pl = setup_resize()

  return render_multiscale(objective, image_pl, session, resize_op, resize_image_pl, 
                           resize_shape_pl, iter_n, step, octave_n, octave_scale, means, is_training_pl, base_image)

def render_image_deepdream(objective, session, image_pl, base_image,
                    iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
  """
  Render deep dream image for selected objective.
  :param objective:       Objective to be optimized.
  :param session:         Tensorflow session.
  :param image_pl:        Input placeholder tensor.
  :param base_image:      Base image to use.
  :param iter_n:          Number of iterations.
  :param step:            Size of step.
  :param octave_n:        Number of octaves.
  :param octave_scale:    Scale for octave.
  :return:                Generated image.

  """

  resize_op, resize_image_pl, resize_shape_pl = setup_resize()
  
  return render_deepdream(objective, session, image_pl, base_image, resize_op, resize_image_pl, resize_shape_pl, iter_n, step, octave_n, octave_scale)

def setup(means=None):
  """
  Creates input placeholder.
  :param means:           Means to be substracted from base image.
  :return:                Input placeholder.

  """

  return create_input_placeholder(means)