# Conveiro

Conveiro is an open source library for feature visualization in deep convolutional networks. It implements multiple techniques for visualization, such as laplace, multiscale,  deep dream and CDFS.

All of these methods are based on:
* https://distill.pub/2017/feature-visualization/ 
* https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

## Requirements

* Python 3.4 and above
* Tensorflow (CPU or GPU variant)
* Numpy
* Matplotlib
* OpenCV-Python

## Installation

```
pip install conveiro
```

Development version

```
pip install -e .    # from cloned repository
```

## Examples

For examples how to use this library please take a look at jupyter notebooks in `docs/` folder:

* https://github.com/Showmax/conveiro/tree/master/docs/deep_dream.ipynb
* https://github.com/Showmax/conveiro/tree/master/docs/cdfs.ipynb

Simplest example:

```python
import tensorflow as tf
import tensornets as nets
from conveiro import cdfs

input_t, decorrelated_image_t, coeffs_t = cdfs.setup(224)

model = nets.Inception1(input_t)
graph = tf.get_default_graph()

with tf.Session() as sess:
    sess.run(model.pretrained())

    objective = graph.get_tensor_by_name("inception1/block3b/concat:0")
    image = cdfs.render_image(sess, decorrelated_image_t, coeffs_t, objective[..., 55], 0.01)
    cdfs.show_image(cdfs.process_image(image))
```

![CDFS output](docs/example.png)

**Note** The API is preliminary and may change in future versions.
