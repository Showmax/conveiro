#!/usr/bin/env python3
"""The CLI tool of conveiro library.

This gets called as `conveiro` after installing using `pip`.
"""
import itertools
import os
import re

import click


@click.group()
def run_app():
    pass


DEFAULT_SIZE = 224


@run_app.command()
@click.option("-r", "--renderer", default="deep-dream", type=click.Choice(["deep-dream", "cdfs"]))
@click.option("-n", "--network", help="Architecture of the neural network", type=str, required=True)
@click.option("-l", "--layers", help="Tensors (layers) to render (as comma-separated regexes to match).", type=str, required=True)
@click.option("-s", "--slices", help="Slices to render (comma-separated numbers, ':' for all, omit for whole tensor).",
              type=str, required=False)
@click.option("-c", "--contrast", help="Contrast for the resulting image", type=float, default=0.3)
@click.option("-R", "--resolution", help="Number of pixels along one dimension (applicable only without input image).",
              type=int, default=DEFAULT_SIZE)
@click.option("-i", "--input-images", help="If present, source image(s) for hallucination.")
@click.option("-o", "--output-dir", help="Directory to write the image to (otherwise just show in a new window).")
@click.option("-v", "--verbose", is_flag=True, help="Produce verbose output.")
@click.option("-N", "--num-steps", type=int, help="Number of steps (128 for CDFS, 10 for deep dream.")
@click.option("-A", "--deep-dream-algorithm", type=click.Choice(["deep-dream", "multi-scale", "laplace"]), default="deep-dream")
@click.option("-L", "--cdfs-learning-rate", type=float, default=0.01, help="Learning rate for CDFS algorithm (default=0.01).")
@click.option("-g", "--grayscale", is_flag=True, help="Produce grayscale image.")
def render(renderer, layers, network, input_images, output_dir, contrast, slices, verbose, resolution, **kwargs):
    """Hallucinate an image for a layer / neuron.
    
    Examples:

    \b
      # Hallucinate on mountains in different concat layers
      conveiro render -n Inception1 -l "inception1/block4c/concat" -i docs/mountain.jpeg -o mountains/

    \b
      # Relatively large images for all filters in one convolutional layer
      conveiro render -n Inception1 -l "inception1/block4e/5x5/1/conv/Conv2D" -s : -o conv2d/ -R 512

    \b
      # Get a few "Rorschach" images from CDFS
      conveiro render -r cdfs -n Inception1 -l "inception1/block../concat" -N 10 -o cfs-concats/

    \b
      # Create an artificial grayscale cheetah-like image using Inception1
      conveiro render -n Inception1 -l inception1/logits/MatMul -s 293 -R 256 -o cheetah/ -N 100 -g
    """
    if verbose:
        print("Loading tensorflow...")
    import tensorflow as tf
    if verbose:
        print("Loading tensornets...")
    import tensornets as nets
    from conveiro import utils

    # Get network
    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available. Run `conveiro networks` to display valid options.")
        exit(-1)

    # Set up algorithm
    if verbose:
        print("Loading {0}...".format(renderer))
    if renderer == "deep-dream":
        renderer = DeepDreamRenderer(size=resolution,
                                     algorithm=kwargs.get("deep_dream_algorithm"),
                                     num_steps=kwargs.get("num_steps", None))
    elif renderer == "cdfs":
        renderer = CDFSRenderer(size=resolution,
                                learning_rate=kwargs.get("cdfs_learning_rate"),
                                num_steps=kwargs.get("num_steps", None))

    # Get model and objective
    if verbose:
        print(f"Creating model {network}...")
    model = constructor(renderer.input_t)
    graph = tf.get_default_graph()
    session = tf.Session()
    session.run(model.pretrained())

    all_tensors = available_tensors(graph)
    op_patterns = layers.split(",")
    operations = (tensor for tensor in all_tensors if any(re.fullmatch(pattern + ":0", tensor[0]) for pattern in op_patterns))

    if not input_images:
        input_images = [None]
    else:
        input_images = input_images.split(",")

    for tensor_name, _, op_shape in operations:
        tensor = graph.get_tensor_by_name(tensor_name)

        # Find slices (due to potentially different tensor sizes, this must be in inner loop)
        if slices is None:
            objectives = (
                (None, tensor),
            )
        elif slices == ":":
            objectives = (
                (i, tensor[..., i]) for i in range(op_shape[-1])
            )
        else:
            indices = [int(i) for i in slices.split(",")]
            objectives = (
                (i, tensor[..., i]) for i in indices
            )

        for index, objective in objectives:
            for input_image in input_images:
                if verbose:
                    print(f"Rendering {tensor_name[:-2]}[{index if index is not None else ':'}] from {input_image if input_image else 'random noise'}...")

                raw_image = renderer.render(objective, session, image=input_image)
                output_image = utils.normalize_image(raw_image, contrast=contrast, bw=kwargs.get("grayscale", False))

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    basename = os.path.splitext(os.path.basename(input_image))[0] if input_image else "random"
                    output_filename = (
                        basename + "-" +
                        tensor_name[:-2].replace("/", "__") +
                        (f"-{index}" if index is not None else "") +
                        ".jpg"
                    )
                    output_path = os.path.join(output_dir, output_filename)
                    utils.save_image(output_image, output_path)
                else:
                    utils.show_image(output_image)


@run_app.command()
def networks():
    """List available network architectures (from tensornets).
    
    Note that not all architectures and not all tensors
    can be visualized.
    """
    print("Available network architectures:")
    for candidate in available_nets():
        print(" ", candidate)


@run_app.command()
@click.argument("network")
@click.option("-t", "--type", help="Operation type to filter")
@click.option("-n", "--name", help="Regular expression for operation name to search for")
def layers(network, type, name):
    """List available layers (operations) in a network.
    
    Examples:

    \b
      conveiro layers Inception1
      conveiro layers Inception1 -t Conv2D -n block5b

    Note that not all architectures and not all layers
    can be visualized.
    """
    import tensorflow as tf
    import tensornets as nets

    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available.")
        exit(-1)

    input_pl = tf.placeholder(tf.float32, shape=(None, None, 3), name="input")
    input_t = tf.expand_dims(input_pl, axis=0)

    _ = constructor(input_t)
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    if name:
        ops = [op for op in ops if re.search(name, op.name)]
    if type:
        ops = [op for op in ops if op.type == type]
    for m in ops:
        for tensor in m.outputs:
            print(tensor.name, m.type, tensor.shape.as_list())


@run_app.command()
@click.argument("network")
@click.option("-o", "--output-path", help="Path to write the graph to (in dot format).")
def graph(network, output_path):
    """Create a graph of the network architecture."""
    import tensornets as nets
    from conveiro import utils

    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available.")
        exit(-1)

    dot = utils.create_graph(constructor)

    if not output_path:
        dot.view()
    else:
        dot.save(output_path)


def available_nets():
    """All network architectures available in tensornets.

    Based on the assumption that each network name starts
    with a capital letter."""
    import tensornets as nets
    candidates = dir(nets)
    return [candidate for candidate in candidates if candidate[0].isupper()]


def available_tensors(graph):
    """All tensors in the model's graph."""
    tensors = []
    ops = graph.get_operations()
    for m in ops:
        for tensor in m.outputs:
            tensors.append((tensor.name, m.type, tensor.shape.as_list()))
    return tensors



class CDFSRenderer:
    """Class representation of the CDFS rendering algorithm with settings."""

    DEFAULT_NUMBER_OF_STEPS = 128

    def __init__(self, size, *, learning_rate, num_steps):
        from conveiro import cdfs
        self.input_t, self._decorrelated_image_t, self._coeffs_t = cdfs.setup(size)
        self.learning_rate = learning_rate
        self.num_steps = num_steps or self.DEFAULT_NUMBER_OF_STEPS

    def render(self, objective, session, image=None):
        if image:
            raise ValueError("Image input for CDFS not implemented.")
        from conveiro import cdfs
        return cdfs.render_image(
            sess=session,
            decorrelated_image=self._decorrelated_image_t,
            coeffs_t=self._coeffs_t,
            objective=objective,
            learning_rate=self.learning_rate,
            num_steps=self.num_steps
        )


class DeepDreamRenderer:
    """Class representation of the deep dream rendering algorithm with settings."""

    DEFAULT_NUMBER_OF_STEPS = 10

    def __init__(self, size, algorithm="deep-dream", num_steps=None):
        from conveiro import deep_dream
        self.size = size or DEFAULT_SIZE
        self.algorithm = algorithm
        self.num_steps = num_steps or self.DEFAULT_NUMBER_OF_STEPS
        self._input_pl, self.input_t = deep_dream.setup()

    def render(self, objective, session, image=None):
        from conveiro import deep_dream
        if image:
            import matplotlib.pyplot as plt
            base_image = plt.imread(image)
        else:
            base_image = deep_dream.get_base_image(self.size, self.size)

        functions = {
            "laplace": deep_dream.render_image_lapnorm,
            "multiscale": deep_dream.render_image_multiscale,
            "deep-dream": deep_dream.render_image_deepdream,
        }
        return functions[self.algorithm](
            objective=objective,
            session=session,
            image_pl=self._input_pl,
            base_image=base_image,
            iter_n=self.num_steps
        )


if __name__ == "__main__":
    run_app()
