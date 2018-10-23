import tensorflow as tf
import numpy as np

def rank(tensor):
  """
  Gets rank of tensor.
  :param tensor:      Tensor.
  :return:
  """

  return len(tensor.get_shape())


def roll_2d(tensor, pixels_x, pixels_y):
  """
  Roll an image. Analogous to Numpy roll.
  :param tensor:      2D input Tensor, an image.
  :param pixels_x:    Number of pixels to roll by in the first dimension.
  :param pixels_y:    Number of pixels to roll by in the second dimension.
  :return:
  """

  original_shape = tensor.shape

  x_len = tf.shape(tensor)[0]
  y_len = tf.shape(tensor)[1]

  tensor = tf.concat([tensor[:, y_len - pixels_y:], tensor[:, :y_len - pixels_y]], axis=1)
  tensor = tf.concat([tensor[x_len - pixels_x:], tensor[:x_len - pixels_x]], axis=0)

  tensor.set_shape(original_shape)

  return tensor

def process_image(image, scale=0.1, bgr=False):
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
