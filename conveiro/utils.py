import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CONTRAST = 0.3

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

def bgr_to_rgb(image):
  """Swap blue and red channel in an image."""
  image = image.copy()
  tmp = image[..., 0].copy()
  image[..., 0] = image[..., 2]
  image[..., 2] = tmp
  return image

def normalize_image(image, contrast=DEFAULT_CONTRAST, bw=False):
  """Normalize the image using mean and std dev.

  :param image:     An image.
  :param contrast:  Multiplication factor for the distance from medium gray (0.5, 0.5, 0.5).
  :param bw:        If True, convert the image to grayscale
  :return:          Normalized image (as numbers 0 to 1)

  Note: The larger the `contrast`, the more visible features in the image and the 
  larger areas of the image will be clipped to 0 or 1.
  """
  image = (image - image.mean()) / max(image.std(), 1e-4) * contrast + 0.5
  if bw:
    image[:] = image.mean(axis=2)[..., np.newaxis]
  image = np.clip(image, 0, 1)
  return image

def save_image(image, path):
  """Save image.

  :param image: Image to be saved (as n x m x 3 numpy array) in range 0.0 to 1.0
  :param path: Where to store the image.

  Note: Requires PIL (or pillow).
  """
  from PIL import Image
  im = Image.fromarray((image * (255)).astype(np.uint8), mode="RGB")
  im.save(path)

def show_image(image, axis=None):
  """ Show an image.

  :param image:       An image.
  :param axis:        Matplotlib axis where to
  :return:            None.
  """
  if not axis:
    plt.axis('off')
  plt.imshow(image)
  plt.show()

def create_graph(model_constructor):
  """Create a graphviz graph of a network.
  
  :param model_constructor: Constructor that takes input placeholder.
  :return: The dot graph
  """
  from graphviz import Digraph

  input_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="input")
  input_t = tf.expand_dims(input_pl, axis=0)

  _ = model_constructor(input_t)
  graph = tf.get_default_graph()

  # Inspired by https://blog.jakuba.net/2017/05/30/Visualizing-TensorFlow-Graphs-in-Jupyter-Notebooks/
  dot = Digraph()
  for n in graph.as_graph_def().node:
      dot.node(n.name, label=n.name)
      for i in n.input:
          # Edges are determined by the names of the nodes
          dot.edge(i, n.name)

  return dot
