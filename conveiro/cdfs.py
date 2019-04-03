import os, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import conveiro.utils as model_utils

MEANS_RGB = [0.48, 0.46, 0.41]
COLOR_CORRELATION_SVD_SQRT_RGB = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype("float32")
CROP_NONE = 1
CROP_CENTER = 2
CROP_RANDOM = 3

def get_coeffs_zeros(size):
  """
  Get Fourier coefficients (that will be used to parameterize the image) initialized with zeros.
  :param size:      Size of the image.
  :return:          Three sets of Fourier coefficients for B, G and R channels.
  """

  return tf.Variable(np.zeros((size, size // 2 + 1, 3, 2), dtype=np.float32), name="coeffs", dtype=tf.float32)

def get_coeffs_random_noise(size, std=0.01):
  """
  Get Fourier coefficients (that will be used to parameterize the image) initialized with random gaussian noise.
  :param size:      Size of the image.
  :param std:       Sample uniformly from -N to N.
  :return:          Three sets of Fourier coefficients for B, G and R channels.
  """
  return tf.Variable(std * np.random.randn(size, size // 2 + 1, 3, 2),
                     name="coeffs", dtype=tf.float32)

def get_image_random_noise(size, std=1):
  """
  Get an image variable filled with random noise.
  :param size:        Size of the image.
  :param std:    Variance of the noise.
  :return:            Image variable.
  """

  return tf.Variable(np.random.normal(0, std, size=size), name="image", dtype=tf.float32)

def coeffs_to_spectrum(coeffs):
  """
  Convert real-values Fourier coefficients into a complex spectrum.
  :param coeffs:    Fourier coefficients.
  :return:          Fourier spectrum.
  """

  return tf.cast(tf.complex(coeffs[..., 0], coeffs[..., 1]), tf.complex64)

def get_spectrum_scale(size):
  """
  Construct a frequency filter that scales spectrum values by their frequencies.
  :param size:      Size of the image.
  :return:          Frequency filter.
  """

  fy = np.fft.fftfreq(size)[:, None]

  if size % 2 == 1:
    fx = np.fft.fftfreq(size)[:size // 2 + 2]
  else:
    fx = np.fft.fftfreq(size)[:size // 2 + 1]

  frequencies = np.sqrt(fx * fx + fy * fy)

  spectrum_scale = 1.0 / np.maximum(frequencies, 1.0 / max(size, size))
  spectrum_scale *= np.sqrt(size * size)

  return spectrum_scale

def scale_frequency_spectrum(spectrum, spectrum_scale):
  """
  Applies spectrum scale to generated spectrum.
  :param spectrum:        Three Fourier spectra of R, G and B channels.
  :param spectrum_scale:  Frequency scale for Fourier spectrum.
  :return:                Scaled spectrum.
  """
  return tf.stack([spectrum[..., i] * spectrum_scale for i in range(spectrum.shape[-1].value)], axis=-1)

def spectrum_to_image(spectrum, size):
  """
  Convert Fourier spectrum to an RGB image.
  :param spectrum:        Three Fourier spectra of R, G and B channels.
  :param size:            Size of the image.
  :return:                RGB image.
  """

  return tf.stack([tf.spectral.irfft2d(spectrum[..., i], fft_length=(size, size)) for i in range(3)], axis=-1)

def decorrelate_colors(image, color_correlation_svd_sqrt):
  """
  Decorrelate colors of image.
  :param image:                       Input image.
  :param color_correlation_svd_sqrt:  Color correlation matrix.
  :return:                            Decorrelated image.
  """

  max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

  image_flat = tf.reshape(image, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  image_flat = tf.matmul(image_flat, color_correlation_normalized.T)
  image = tf.reshape(image_flat, tf.shape(image))

  return image


def crop_center(image, height, width):
  """
  Crop the center of an image.
  :param image:       An image.
  :param height:      Height of the image.
  :param width:       Width of the image.
  :return:            Cropped image.
  """

  image =  tf.image.crop_to_bounding_box(
    image, (tf.shape(image)[0] - height) // 2, (tf.shape(image)[1] - width) // 2, height, width)

  image.set_shape((height, width, 3))

  return image


def data_augmentation(image, padding=16, scaling_factors=(1, 0.975, 1.025, 0.95, 1.05),
                      angles=(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5), jitter_1=16, jitter_2=8,
                      crop_type=CROP_CENTER, crop_size=None):
  """
  Perform data augmentation on an image.
  :param image:               An image.
  :param padding:             How many pixel to pad on each side, None for no padding.
  :param scaling_factors:     Scales to randomly choose from, None for no scale augmentation.
  :param angles:              Angles to randomly choose from, None for no rotations.
  :param jitter_1:            Jitter before scaling and rotating up to N pixels.
  :param jitter_2:            Jitter after scaling and rotating up to N pixels.
  :param crop_type:         Crop the padded image randomly.
  :param crop_size:           How big image to crop. Default is the original image size.
  :return:                    Augmented image.
  """

  image_rank = model_utils.rank(image)

  if image_rank not in [3, 4]:
    raise ValueError("Unsuported Tensor rank ({:d})".format(model_utils.rank(image)))

  # remember original height and width
  if crop_size is not None:
    height = crop_size
    width = crop_size
  else:
    height = image.shape[-3].value
    width = image.shape[-2].value

  # convert scaling factors and angles into Tensors
  scaling_factors = tf.convert_to_tensor(scaling_factors, dtype=tf.float32)
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)

  # apply padding
  if padding is not None:
    if image_rank == 3:
      paddings = [[16, 16], [16, 16], [0, 0]]
    elif image_rank == 4:
      paddings = [[0, 0], [16, 16], [16, 16], [0, 0]]
    else:
      raise ValueError("Unsuported Tensor rank ({:d})".format(model_utils.rank(image)))

    image = tf.pad(image, paddings)

  # first jitter
  if jitter_1 is not None:
    random_jitter_1 = tf.random_uniform([2], minval=-jitter_1, maxval=jitter_1 + 1, dtype=tf.int32)
    image = model_utils.roll_2d(image, random_jitter_1[0], random_jitter_1[1])

  # random scaling
  if scaling_factors is not None:
    random_scale_idx = tf.random_uniform([], minval=0, maxval=scaling_factors.shape[0].value, dtype=tf.int32)
    random_scale_height = tf.cast(image.shape[-3].value * scaling_factors[random_scale_idx], tf.int32)
    random_scale_width = tf.cast(image.shape[-2].value * scaling_factors[random_scale_idx], tf.int32)

    if image_rank == 3:
      image = tf.image.resize_bilinear(tf.expand_dims(image, 0), (random_scale_height, random_scale_width))[0]
    else:
      image = tf.image.resize_bilinear(image, (random_scale_height, random_scale_width))

  else:
    # no scaling performed
    random_scale_height = image.shape[-3].value
    random_scale_width = image.shape[-2].value

  # random rotation
  if angles is not None:
    random_angle_idx = tf.random_uniform([], minval=0, maxval=angles.shape[0].value, dtype=tf.int32)
    image = tf.contrib.image.rotate(image, angles[random_angle_idx] * math.pi / 180, interpolation="BILINEAR")

  # second jitter
  if jitter_2 is not None:
    random_jitter_2 = tf.random_uniform([2], minval=-jitter_2, maxval=jitter_2 + 1, dtype=tf.int32)
    image = model_utils.roll_2d(image, random_jitter_2[0], random_jitter_2[1])

  # crop out an image of the same size as the original image
  if crop_type == CROP_RANDOM:
    return tf.random_crop(image, (height, width, 3))
  elif crop_type == CROP_CENTER:
    return tf.image.crop_to_bounding_box(
      image, (random_scale_height - height) // 2, (random_scale_width - width) // 2, height, width)
  elif crop_type == CROP_NONE:
    return image
  else:
    raise ValueError("Invalid crop type.")

def get_objective(target_layer, filter_idx, middle=False):
  """
  Get single value to optimize.
  :param target_layer:      Layer to optimize.
  :param filter_idx:        Index of a filter (or logit) to optimize.
  :param middle:            Optimize the middle neuron.
  :return:                  Single value to optimize - either a single neuron or a mean activation of a channel.
  """

  if middle:
    if model_utils.rank(target_layer) != 4:
      raise ValueError("Middle neuron can be optimized only for convolutional layers with output of rank 4 "
                       "(batch size, height, width, channels).")

    y = target_layer.shape[0].value // 2
    x = target_layer.shape[1].value // 2

    return target_layer[0, x, y, filter_idx]
  else:
    return tf.reduce_mean(target_layer[..., filter_idx])

def setup_optimizer(target, coeffs, learning_rate, use_adam=False, namespace="opt"):
  """
  Setup Gradient Ascent optimizer.
  :param target:            Value to increase.
  :param coeffs:            Coefficients to optimize.
  :param learning_rate:     Learning rate.
  :param use_adam:          Use an adaptive optimizer.
  :param namespace:         Namespace of the optimizer.
  :return:                  Optimization step operation.
  """

  with tf.variable_scope(namespace):

    if use_adam:
      opt = tf.train.AdamOptimizer(learning_rate)
    else:
      opt = tf.train.GradientDescentOptimizer(learning_rate)

    grads = opt.compute_gradients(-target, var_list=[coeffs])
    return opt.apply_gradients(grads)

def reset_optimizer(session, coeffs, opt_vars):
  """
  Reset the optimizer and the image parameters.
  :param session:           A Tensorflow session.
  :param coeffs:            Fourier coefficients.
  :param opt_vars:          Optimizer variables (e.g. momenta for Adam optimizer)
  :return:                  None.
  """

  session.run(tf.variables_initializer(opt_vars))
  session.run(tf.variables_initializer([coeffs]))

def optimize(opt_step, session, filter_pl, filter_idx, num_steps=512, is_training_pl=None):
  """
  Perform Gradient Ascent on an image.
  :param opt_step:          Gradient Ascent step operation.
  :param session:           Tensorflow session.
  :param filter_pl:         Filter index placeholder.
  :param filter_idx:        Filter index.
  :param num_steps:         Number of optimization steps.
  :param is_training_pl:    Placeholder for is_training variable.
  :return:                  None.
  """

  feed_dict = {
    filter_pl: filter_idx
  }

  if is_training_pl is not None:
    feed_dict[is_training_pl] = False

  for _ in range(num_steps):
      session.run(opt_step, feed_dict=feed_dict)


def render_image(sess, decorrelated_image, coeffs_t, objective, learning_rate, num_steps = 512, is_training_pl=None):
  opt_step = setup_optimizer(objective, coeffs_t, learning_rate)
  opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "opt")

  reset_optimizer(sess, coeffs_t, opt_vars)

  feed_dict = {}

  if is_training_pl is not None:
    feed_dict[is_training_pl] = False

  for _ in range(num_steps):
    sess.run(opt_step, feed_dict=feed_dict)

  return sess.run(decorrelated_image)


def setup(size):
    spectrum_scale = get_spectrum_scale(size)

    coeffs_t = get_coeffs_random_noise(size)
    spectrum_t = coeffs_to_spectrum(coeffs_t)

    scaled_spectrum_t = scale_frequency_spectrum(spectrum_t, spectrum_scale)

    image_t = spectrum_to_image(scaled_spectrum_t, size)

    decorrelated_image_t = decorrelate_colors(image_t, COLOR_CORRELATION_SVD_SQRT_RGB)

    decorrelated_image_t = tf.nn.sigmoid(decorrelated_image_t)

    input_t = (data_augmentation(decorrelated_image_t - MEANS_RGB))

    channels = tf.unstack(input_t, axis=-1)
    input_t = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    input_t = tf.expand_dims(input_t, axis=0)

    return input_t, decorrelated_image_t, coeffs_t
