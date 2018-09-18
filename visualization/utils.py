import os, random
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def count_trainable_params():
  total_parameters = 0
  for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
      variable_parametes *= dim.value
    total_parameters += variable_parametes
  return total_parameters

def log_params(path, params):
  with open(path, "w") as file:
    for key in params.keys():
      file.write("{}: {}\n".format(key, params[key]))

def top_k_error(predictions, labels, k):
  batch_size = tf.shape(predictions)[0]
  in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=k))
  num_correct = tf.reduce_sum(in_top1)

  return (batch_size - num_correct) / batch_size

def print_tensors_in_checkpoint(path, print_values=False, print_means=False):
  """
  Print all tensors in a checkpoint.
  :param print_values:    Print Tensor values in addition to their names.
  :return:                None.
  """

  reader = pywrap_tensorflow.NewCheckpointReader(path)
  var_to_shape_map = reader.get_variable_to_shape_map()

  for key in sorted(var_to_shape_map):
    print("tensor_name: ", key)
    if print_values:
      print(reader.get_tensor(key))

    if print_means:
      print(np.mean(reader.get_tensor(key)))

def get_tensors_in_checkpoint(path):

  reader = pywrap_tensorflow.NewCheckpointReader(path)
  var_to_shape_map = reader.get_variable_to_shape_map()

  tensors = {}

  for key in sorted(var_to_shape_map):

    values = reader.get_tensor(key)
    tensors[key] = values

  return tensors

def print_collection(collection):

  vars = tf.get_collection(collection)

  for var in vars:
    print(var.name)

def rank(tensor):
  return len(tensor.get_shape())

def create_new_run_dirs(summary_dir, save_dir, run_dir_template="run{:d}", log_file_name="log.txt"):
  """
  Create current run summary and save directories.
  :param summary_dir:         Root of run summaries.
  :param save_dir:            Root of run saves.
  :param run_dir_template:    Template for the run directory (uses the new Python formatting).
  :param log_file_name:       Name of a hyperparameter log file.
  :return:                    Tuple: current run summary directory, save directory and log path.
  """

  i = 1
  while True:
    run_summary_dir = os.path.join(summary_dir, run_dir_template.format(i))
    run_save_dir = os.path.join(save_dir, run_dir_template.format(i))

    if not os.path.exists(run_summary_dir) and not os.path.exists(run_save_dir):
      os.makedirs(run_summary_dir)
      os.makedirs(run_save_dir)

      log_file = os.path.join(run_summary_dir, log_file_name)
      break

    i += 1

  return run_summary_dir, run_save_dir, log_file

def set_random_seed():
  tf.set_random_seed(config.RANDOM_SEED)
  np.random.seed(config.RANDOM_SEED)
  random.seed(config.RANDOM_SEED)

def restore_pretrained(sess, path, collection):
  """
  Restore ResNet from pretrained weights.
  :return:    None.
  """

  # gather all convolutional params
  params_dict = create_params_dict(collection)

  # load the params from the checkpoint
  saver = tf.train.Saver(params_dict)
  saver.restore(sess, path)

def create_params_dict(collection, trim_namespace=True, trim_namespace_num_levels=1, ignore=None, prefix=None):
  """
  Create a dictionary of parameters to load or save.
  :param collection:      Collection containing the parameters.
  :param trim_namespace:  Delete the global namespace of each variable (e.g. one/two/three -> two/three).
  :param ignore:          List of parameters to ignore.
  :param prefix:          Prefix for variable names in checkpoint.
  :return:                Dictionary of parameters.
  """

  params = tf.get_collection(collection)
  params_dict = {}

  for param in params:

    if ignore is not None:
      if param in ignore:
        continue

    if trim_namespace:
      # original format: resnet/name1/name2/../nameN:0
      # target format: name1/name2/.../nameN
      key = "/".join(param.name.split(":")[0].split("/")[trim_namespace_num_levels:])
    else:
      key = param.name

    if prefix is not None:
      key = "{}/{}".format(prefix, key)

    params_dict[key] = param

  return params_dict

def save_pretrained(sess, path, ignore, collection):
  """
  Saves modified pretrained weights.
  :param sess:      Current session.
  :param path:      Save path.
  :param ignore:    Parameters to ignore.
  :return:          None.
  """

  # gather all convolutional params
  params_dict = create_params_dict(collection, ignore=ignore)

  # load the params from the checkpoint
  saver = tf.train.Saver(params_dict)
  saver.save(sess, path)

def argmax_2d(tensor):

  assert rank(tensor) == 4

  flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))
  argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

  argmax_x = argmax // tf.shape(tensor)[2]
  argmax_y = argmax % tf.shape(tensor)[2]

  return tf.stack((argmax_x, argmax_y), axis=1)

def fill_filters_with_zeros(tensor, filter_idx):

  zeros = tf.zeros_like(tensor)

  zeros_1 = zeros[..., :filter_idx]
  zeros_2 = zeros[..., filter_idx+1:]

  zeros = tf.concat([zeros_1, tensor[..., filter_idx:filter_idx+1], zeros_2], axis=-1)

  return zeros

def find_convolution_operations(conv_layer_name="Conv2D"):
  """
  Find all convolutional operations.
  :param conv_layer_name:     Name of the convolutional operation.
  :return:                    List of convolutional operations.
  """

  ops = tf.get_default_graph().get_operations()
  conv_ops = []

  for op in ops:
    op_id = op.name
    op_name = op_id.split("/")[-1]

    if op_name == conv_layer_name:
      conv_ops.append(op)

  return conv_ops

def find_convolution_kernels(conv_layer_name="Conv2D", conv_weights_name="weights"):
  """
  Find all weights of convolutional layers.
  :param conv_layer_name:       Name of convolutional operations.
  :param conv_weights_name:     Name of convolutional weights.
  :return:                      List of convolutional kernels.
  """

  conv_ops = find_convolution_operations(conv_layer_name=conv_layer_name)
  conv_op_names = [op.name for op in conv_ops]

  weight_tensors = []

  for op_name in conv_op_names:

    op_prefix = "/".join(op_name.split("/")[:-1])
    weights_name = "{}/{}:0".format(op_prefix, conv_weights_name)

    weights = tf.get_default_graph().get_tensor_by_name(weights_name)
    weight_tensors.append(weights)

  return weight_tensors

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

def add_to_dict_or_create_dict(key_1, key_2, value, dictionary):
  """
  Add a value to a dictionary under a key in a parent dictionary or create a new dictionary under the key.
  :param key_1:         Key for the child dictionary.
  :param key_2:         Key for the value.
  :param value:         The value.
  :param dictionary:    Parent dictionary.
  :return:              None.
  """

  if key_1 not in dictionary:
    dictionary[key_1] = {
      key_2: value
    }
  else:
    dictionary[key_1][key_2] = value

def find_variable(name, variables):
  """
  Finds a variable by name.
  :param name:          Name of the variable, :0 postfix is optional.
  :param variables:     List of Tensorflow variables.
  :return:              The first found variables or None if no variable was found.
  """

  results = []

  for variable in variables:
    if variable.name == name or variable.name == "{:s}:0".format(name):
      results.append(variable)

  if len(results) == 0:
    return None
  else:
    return results[0]

def reduce_to_last_dimension_np(tensor):
  """
  Return Tensor with rank 1.
  :param tensor:    A tensor.
  :return:          A tensor with rank 1.
  """

  shape_len = len(tensor.shape)

  for i in range(shape_len - 1):

    if tensor.shape[0] > 1:
      raise ValueError("Reducing Tensor dimension with shape > 1 is forbidden (strict).")

    tensor = tensor[0]

  return tensor

def get_bn_moments_and_means(allowed_types=("VariableV2", "Mean"),
                             mean_op_name="moments/mean", var_op_name="moments/variance",
                             moving_mean_tensor_name="moving_mean", moving_var_tensor_name="moving_variance"):
  """
  Get batch normalization moments and moving averages.
  :param allowed_types:                       Allowed Tensorflow operation types, None for all.
  :param mean_op_name:                        Name of the operation that computes mean.
  :param var_op_name:                         Name of the operation that computes variance.
  :param moving_mean_tensor_name:             Name of the Tensor that stores moving mean.
  :param moving_var_tensor_name:              Name of the Tensor that stores moving variance.
  :param results_mean_op_name:                Key for the found mean operations.
  :param results_var_op_name:                 Key for the found variance operations.
  :param results_moving_mean_tensor_name:     Key for the found moving mean Tensors.
  :param results_moving_var_tensor_name:      Key for the found moving variance Tensors.
  :return:                                    Dictionaries with found Tensors and variables.
  """

  ops = tf.get_default_graph().get_operations()

  result_tensors = {}
  result_variables = {}

  for op in ops:

    if allowed_types is not None and op.type not in allowed_types:
      continue

    op_name = op.name

    if len(op_name.split("/")) >= 1:
      last_term = op_name.split("/")[-1]
      last_term_complement = "/".join(op_name.split("/")[:-1])
    else:
      last_term = None
      last_term_complement = None

    if len(op_name.split("/")) >= 2:
      last_two_terms = "/".join(op_name.split("/")[-2:])
      last_two_terms_complement = "/".join(op_name.split("/")[:-2])
    else:
      last_two_terms = None
      last_two_terms_complement = None

    if last_two_terms == mean_op_name:
      add_to_dict_or_create_dict(last_two_terms_complement, "mean", op.outputs[0], result_tensors)
    elif last_two_terms == var_op_name:
      add_to_dict_or_create_dict(last_two_terms_complement, "variance", op.outputs[0], result_tensors)
    elif last_term == moving_mean_tensor_name:
      add_to_dict_or_create_dict(last_term_complement, "mean",
                                 find_variable(op.name, tf.global_variables()), result_variables)
    elif last_term == moving_var_tensor_name:
      add_to_dict_or_create_dict(last_term_complement, "variance",
                                 find_variable(op.name, tf.global_variables()), result_variables)

  return result_tensors, result_variables

def create_running_means_for_bn_moments(bn_ops):
  """
  Create running means for each batch normalization operations.
  :param bn_ops:      Dictionary of batch normalization moments generated by get_bn_moments_and_means.
  :return:            Dictionary of running means.
  """

  running_means = {}

  for key_1, value_1 in bn_ops.items():
    running_means[key_1] = {}

    for key_2, value_2 in value_1.items():
      running_means[key_1][key_2] = data_utils.RunningMean()

  return running_means

def update_running_means_for_bn_moments(running_means_dict, values_dict, batches):
  """
  Update running means for batch normalization operations.
  :param running_means_dict:    Dictionary of running means.
  :param values_dict:           Dictionary of values to update.
  :param batches:               Values dict contains multiple batches.
  :return:                      None.
  """

  for key_1, value_1 in running_means_dict.items():
    for key_2, value_2 in value_1.items():
      if batches:
        for batch_idx in range(values_dict[key_1][key_2].shape[0]):
          running_means_dict[key_1][key_2].add(values_dict[key_1][key_2][batch_idx])
      else:
        running_means_dict[key_1][key_2].add(values_dict[key_1][key_2])

def update_running_mean_tensors_for_bn_ops(running_mean_tensors, running_mean_values, session):
  """
  Update running mean tensors for batch normalization operations.
  :param running_mean_tensors:        Running mean Tensors (Tensorflow).
  :param running_mean_values:         Running mean values: RunningMean objects containing Numpy arrays.
  :param session:                     Tensorflow session.
  :return:                            None.
  """

  for key_1, value_1 in running_mean_tensors.items():
    for key_2, value_2 in value_1.items():

      value = reduce_to_last_dimension_np(running_mean_values[key_1][key_2].get())

      session.run(running_mean_tensors[key_1][key_2].assign(value))

def find_operation(op_name):
  """
  Find operation by name.
  :param op_name:     Operation name.
  :return:            Found operation or None.
  """

  ops = tf.get_default_graph().get_operations()

  for op in ops:
    full_name = op.name
    last_part = full_name.split("/")[-1]

    if last_part == op_name:
      return op

  return None

def find_models(models_path):
  """
  Find all model files (search for files with .index suffix).
  :param models_path:     Path to a directory to search in.
  :return:                Sorted list of model paths (without the suffix).
  """

  models = os.listdir(models_path)
  models = [model.split(".")[0] for model in models if len(model.split(".")) == 2 and model.split(".")[1] == "index"]
  models = [os.path.join(models_path, model) for model in models]
  models = sorted(models)

  return models