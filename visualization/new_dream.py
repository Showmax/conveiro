import cv2, os, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import visualization.utils as model_utils

means_rgb = [0.48, 0.46, 0.41]
color_correlation_svd_sqrt_rgb = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype("float32")

def get_coeffs_zeros(size):
  """
  Get Fourier coefficients (that will be used to parameterize the image) initialized with zeros.
  :param size:      Size of the image.
  :return:          Three sets of Fourier coefficients for B, G and R channels.
  """

  return tf.Variable(np.zeros((size, size // 2 + 1, 3, 2), dtype=np.float32), name="coeffs", dtype=tf.float32)

def get_coeffs_random_noise(size, std=0.01):
  """
  Get Fourier coefficients (that will be used to parameterize the image) initialized with random uniform noise.
  :param size:      Size of the image.
  :param interval:  Sample uniformly from -N to N.
  :return:          Three sets of Fourier coefficients for B, G and R channels.
  """
  return tf.Variable(std * np.random.randn(size, size // 2 + 1, 3, 2),
                     name="coeffs", dtype=tf.float32)

def get_coeffs_random_noise_3d(size, std=0.01, stack_size=32):
  """
  Get Fourier coefficients (that will be used to parameterize the image) initialized with random uniform noise.
  :param size:      Size of the image.
  :param interval:  Sample uniformly from -N to N.
  :return:          Three sets of Fourier coefficients for B, G and R channels.
  """
  return tf.Variable(std * np.random.randn(stack_size, size, size // 2 + 1, 3, 2),
                     name="coeffs", dtype=tf.float32)

def get_image_random_noise(size, variance=1):
  """
  Get an image variable filled with random noise.
  :param size:        Size of the image.
  :param variance:    Variance of the noise.
  :return:            Image variable.
  """

  return tf.Variable(np.random.normal(0, variance, size=size), name="image", dtype=tf.float32)

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

  return tf.stack([spectrum[..., i] * spectrum_scale for i in range(spectrum.shape[-1].value)], axis=-1)

def spectrum_to_image(spectrum, size):
  """
  Convert Fourier spectrum to an RGB image.
  :param spectrum:        Three Fourier spectra of R, G and B channels.
  :return:                RGB image.
  """

  return tf.stack([tf.spectral.irfft2d(spectrum[..., i], fft_length=(size, size)) for i in range(3)], axis=-1)

def decorrelate_colors(image, color_correlation_svd_sqrt):

  max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

  image_flat = tf.reshape(image, [-1, 3])
  color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
  image_flat = tf.matmul(image_flat, color_correlation_normalized.T)
  image = tf.reshape(image_flat, tf.shape(image))

  return image

def decompose_cov_matrix(cov_matrix, size=None):
  """
  Perform Cholesky decomposition on a covariance matrix.
  :param cov_matrix:      Covariance matrix.
  :param size:            Crop covariance matrix to this size (optional).
  :return:                L matrix.
  """

  # maybe crop covariance matrix
  if size is not None:
    if cov_matrix.shape[0] < size or cov_matrix.shape[1] < size:
      raise ValueError("Covariance matrix is smaller than {}x{}.".format(size, size))
    elif cov_matrix.shape[0] > size or cov_matrix.shape[2] > size:
      diff_x = cov_matrix.shape[0] - size
      diff_y = cov_matrix.shape[1] - size

      offset_left = math.ceil(diff_x / 2)
      offset_right = math.floor(diff_x / 2)
      offset_top = math.ceil(diff_y / 2)
      offset_bottom = math.floor(diff_y / 2)

      cov_matrix = cov_matrix[offset_left:-offset_right, offset_top:-offset_bottom]

  return np.linalg.cholesky(cov_matrix).astype(np.float32)

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

CROP_NONE = 1
CROP_CENTER = 2
CROP_RANDOM = 3

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
  :param random_crop:         Crop the padded image randomly.
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

def process_image(image, scale=0.4, bgr=False):
  """
  Normalize image and maybe swap channels.
  :param image:     An image.
  :param scale:     Scaling parameter.
  :param bgr:       Swap red and blue channels.
  :return:          Normalized image
  """

  image = image.copy()

  # BGR to RGB
  if bgr:
    tmp = image[..., 0].copy()
    image[..., 0] = image[..., 2]
    image[..., 2] = tmp

  # normalize
  image = (image - image.mean()) / max(image.std(), 1e-4) * scale + 0.5
  image = np.clip(image, 0, 1)

  return image

def show_image(image, scale=0.4, bgr=False, axis=False):
  """
  Preprocess and show an image.
  :param image:       An image.
  :param scale:       Scaling parameter.
  :param bgr:         Swap red and blue channels.
  :return:            None.
  """

  image = process_image(image, scale=scale, bgr=bgr)

  if not axis:
    plt.axis('off')

  plt.imshow(image)

  plt.show()

def save_image(image, path, scale=0.1, bgr=False, normalize=False, enumerate_image=False):
  """
  Preprocess and save an image.
  :param image:               An image.
  :param path:                Save path.
  :param scale:               Scaling parameter.
  :param bgr:                 Swap red and blue channels.
  :param enumerate_image:     Path is a folder and an image should be assigned a number.
  :return:                    None.
  """

  if enumerate_image:
    if not os.path.isdir(path):
      os.makedirs(path)

    i = 0
    while True:
      image_path = os.path.join(path, "image{}.jpg".format(i))

      if not os.path.isfile(image_path):
        break
      else:
        i += 1
  else:
    image_path = path

  if normalize:
    image = process_image(image, scale=scale, bgr=not bgr)
  else:
    if not bgr:
      c = image[..., 0].copy()
      image[..., 0] = image[..., 2]
      image[..., 2] = c

  cv2.imwrite(image_path, (image * 255).astype(np.uint8))

def save_images(images, path, bgr=False):

  if not os.path.isdir(path):
    os.makedirs(path)

  if not bgr:
    c = images[..., 0].copy()
    images[..., 0] = images[..., 2]
    images[..., 2] = c

  for i in range(images.shape[0]):
    cv2.imwrite(os.path.join(path, "{:d}.jpg".format(i)), (images[i] * 255).astype(np.uint8))

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

    decorrelated_image_t = decorrelate_colors(image_t, color_correlation_svd_sqrt_rgb)

    decorrelated_image_t = tf.nn.sigmoid(decorrelated_image_t)

    input_t = (data_augmentation(decorrelated_image_t - means_rgb))

    channels = tf.unstack(input_t, axis=-1)
    input_t = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    input_t = tf.expand_dims(input_t, axis=0)

    return input_t, decorrelated_image_t, coeffs_t
