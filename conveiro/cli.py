import re

import click


@click.group()
def run_app():
    pass


def available_nets():
    """All network architectures available in tensornets.

    Based on the assumption that each network name starts
    with a capital letter."""
    import tensornets as nets
    candidates = dir(nets)
    return [candidate for candidate in candidates if candidate[0].isupper()]


@run_app.command()
@click.option("-a", "--algorithm", default="deep-dream", type=click.Choice(["deep-dream", "cdfs"]))
@click.option("-n", "--network", help="Architecture of the neural network")
@click.option("-t", "--tensor", help="Tensor to display.")
@click.option("-s", "--slice", help="Use only one slice of the tensor.", type=int, required=False)
@click.option("-c", "--contrast", help="Contrast for the resulting image", type=float, default=0.3)
@click.option("-r", "--resolution", help="Number of pixels along one dimension (applicable only without input image).", type=int, default=224)
@click.option("-i", "--input-image", help="If present, source image for hallucination.")
@click.option("-o", "--output-image", help="Path to write the image to (otherwise just show in a new window).")
@click.option("-v", "--verbose", is_flag=True, help="Produce verbose output.")
@click.option("--cdfs-steps", type=int, default=128, help="Number of steps for CDFS algorithm (default=128).")
@click.option("--learning-rate", type=float, default=0.01, help="Learning rate for CDFS algorithm (default=0.01).")
def render(algorithm, tensor, network, input_image, output_image, contrast, slice, verbose, resolution, **kwargs):
    """Hallucinate an image for a layer / neuron.
    
    Examples:

    \b
      conveiro render -n Inception1 -t "inception1/block4c/concat:0" -i docs/mountain.jpeg -o mountain-out.jpg
      conveiro render -a cdfs -n Inception1 -t "inception1/block3b/concat:0"
    """
    if verbose:
        print("Loading tensorflow...")
    import tensorflow as tf
    if verbose:
        print("Loading tensornets...")
    import tensornets as nets
    import matplotlib.pyplot as plt
    from conveiro import utils

    # Get network
    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available. Run `conveiro networks` to display valid options.")
        exit(-1)

    # Set up algorithm
    if verbose:
            print("Loading {0}...".format(algorithm))
    if algorithm == "deep-dream":
        from conveiro import deep_dream
        input_pl, input_t = deep_dream.setup()
    elif algorithm == "cdfs":
        from conveiro import cdfs
        input_t, decorrelated_image_t, coeffs_t = cdfs.setup(resolution)

    # Get model and objective
    if verbose:
        print(f"Creating model {network}...")
    model = constructor(input_t)
    graph = tf.get_default_graph()
    session = tf.Session()
    session.run(model.pretrained())
    objective = graph.get_tensor_by_name(tensor)
    if slice is not None:
        objective = objective[..., slice]
        
    # Render the image
    if algorithm == "deep-dream":
        if input_image:
            base_image = plt.imread(input_image)
        else:
            base_image = deep_dream.get_base_image(resolution, resolution)
        image = deep_dream.render_image_deepdream(objective, session, input_pl, base_image=base_image)
        result = utils.normalize_image(image, contrast=contrast)
            
    elif algorithm == "cdfs":
        if input_image:
            print("Input for CDFS not implemented.")
            exit(-1)
        else:
            image = cdfs.render_image(session, decorrelated_image_t, coeffs_t,
                                      objective=objective,
                                      learning_rate=kwargs["learning_rate"],
                                      num_steps=kwargs["cdfs_steps"])
            result = utils.normalize_image(image, contrast=contrast)
   
    # Output (show or save)    
    if output_image:
        utils.save_image(result, output_image)
    else:
        utils.show_image(result)


@run_app.command()
def nets():
    """List available network architectures (from tensornets).
    
    Note that not all architectures and not all tensors
    can be visualized.
    """
    import tensornets as nets
    print("Available network architectures:")
    for candidate in available_nets():
        print(" ", candidate)


@run_app.command()
@click.argument("network")
@click.option("-t", "--type", help="Operation type to filter")
@click.option("-n", "--name", help="Regular expression for operation name to search for")
def tensors(network, type, name):
    """List available tensors in a network.
    
    Examples:

    \b
      conveiro tensors Inception1
      conveiro tensors Inception1 -t Conv2D -n block5b

    Note that not all architectures and not all tensors
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

    model = constructor(input_t)
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
    import tensorflow as tf
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
