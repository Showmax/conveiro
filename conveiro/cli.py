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
@click.option("-i", "--input-image", help="If present, source image for hallucination.")
@click.option("-o", "--output-image", help="Path to write the image to (otherwise just show in a new window).")
@click.option("--cdfs-steps", type=int, default=128, help="Number of steps for CDFS algorithm (default=128).")
@click.option("--learning-rate", type=float, default=0.01, help="Learning rate for CDFS algorithm (default=0.01).")
def render(algorithm, tensor, network, input_image, output_image, **kwargs):
    """Hallucinate an image for a layer / neuron.
    
    Examples:

    \b
      conveiro render -n Inception1 -t "inception1/block4c/concat:0" -i docs/mountain.jpeg -o mountain-out.jpg
      conveiro render -a cdfs -n Inception1 -t "inception1/block3b/concat:0"
    """
    print("Loading tensorflow...")
    import tensorflow as tf
    print("Loading tensornets...")
    import tensornets as nets
    import matplotlib.pyplot as plt
    from conveiro import utils

    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available.")
        exit(-1)

    if algorithm == "deep-dream":
        print("Loading deep dream...")
        from conveiro import deep_dream
        input_pl, input_t = deep_dream.setup()

        print(f"Creating model {network}...")
        model = constructor(input_t)
        graph = tf.get_default_graph()
        session = tf.Session()
        session.run(model.pretrained())

        objective = graph.get_tensor_by_name(tensor)
        
        if input_image:
            base_image = plt.imread(input_image)
            # TODO: Add laplace
            image = deep_dream.render_image_deepdream(objective, session, input_pl, base_image)
            result = utils.process_image(image)
        else:
            result = deep_dream.render_image_multiscale(objective[..., 10], session, input_pl) / 255     

        if output_image:
            deep_dream.save_image(result, output_image)
        else:
            deep_dream.show_image(result)

    elif algorithm == "cdfs":
        # TODO: Unify with the deep dream branch
        from conveiro import cdfs
        input_t, decorrelated_image_t, coeffs_t = cdfs.setup(224)

        print(f"Creating model {network}...")
        model = constructor(input_t)
        graph = tf.get_default_graph()
        session = tf.Session()
        session.run(model.pretrained())

        objective = graph.get_tensor_by_name(tensor)

        if input_image:
            print("Input for CDFS not implemented yet.")
            exit(-1)
        else:
            # TODO: Generalize
            image = cdfs.render_image(session, decorrelated_image_t, coeffs_t, objective[..., 55],
                                      learning_rate=kwargs["learning_rate"],
                                      num_steps=kwargs["cdfs_steps"])
            result = utils.process_image(image)
        
        if output_image:
            print("Output for CDFS not implemented yet.")
            exit(-1)
        else:
            cdfs.show_image(result)


@run_app.command()
def nets():
    """List available network architectures."""
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
    """
    import tensorflow as tf
    import tensornets as nets
    from conveiro import deep_dream

    if network in available_nets():
        constructor = getattr(nets, network)
    else:
        print(f"Network {network} not available.")
        exit(-1)

    input_pl, input_t = deep_dream.setup()
    model = constructor(input_t)
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    if name:
        ops = [op for op in ops if re.search(name, op.name)]
    if type:
        ops = [op for op in ops if op.type == type]
    for m in ops:
        try:
            shape = m.get_attr("shape")
            print(m.name, m.type, shape.size)
        except:
            print(m.name, m.type)


