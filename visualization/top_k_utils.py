import numpy as np
import tensorflow as tf

def get_objectives_dict(objective_names, body_name):

  objectives = {}

  for name in objective_names:

    objectives[name] = tf.get_default_graph().get_tensor_by_name("{:s}/{:s}:0".format(body_name, name))

  return objectives

def get_receptive_fields_dict(objective_names, receptive_fields):

  d = {}

  for name in objective_names:

    name = "resnet/" + name.replace(":0", "")

    d[name] = receptive_fields[name]

  return d

def get_reduces_dict(objectives_dict, filters, max_reduce=False):

  reduces = {}

  for name, objective in objectives_dict.items():

    reduces[name] = {}

    for filter_idx in filters:

      if filter_idx >= objective.shape[-1].value:
        break

      if max_reduce:
        reduce = tf.reduce_max(objective[:, :, :, filter_idx])
      else:
        reduce = tf.reduce_mean(objective[:, :, :, filter_idx])

      reduces[name][filter_idx] = reduce

  return reduces

def get_start_max_reduces_dict(reduces_dict, top_k):

  max_reduces = {}

  for name, objective in reduces_dict.items():

    max_reduces[name] = {}

    for filter_idx, value in objective.items():

      max_reduces[name][filter_idx] = [-10e+9] * top_k

  return max_reduces

def find_and_replace_min(value, values_list):
  """
  Find and replace the minimum value in a list.
  :param value:           Value to compare.
  :param values_list:     List of values.
  :return:                Index where the new value was inserted or None if all values are bigger than the new value.
  """

  if value > np.min(values_list):
    index = np.argmin(values_list)
    values_list[index] = value
    return index
  else:
    return None

def get_filters_to_save(new_reduces, max_reduces):

  filters_to_save = {}
  image_indexes = {}

  for name, objective in new_reduces.items():

    for filter_idx, value in objective.items():

      idx = find_and_replace_min(value, max_reduces[name][filter_idx])

      if idx is not None:

        if not name in filters_to_save:
          filters_to_save[name] = []
          image_indexes[name] = []

        filters_to_save[name].append(filter_idx)
        image_indexes[name].append(idx)

  return filters_to_save, image_indexes