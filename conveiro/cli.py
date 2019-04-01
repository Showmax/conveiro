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
def render(algorithm, tensor, network, input_image, output_image):
    """Hallucinate an image for a layer / neuron.
    
    Example:
        conveiro render -n Inception1 -t "inception1/block4c/concat:0"
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


@run_app.command()
def nets():
    """List available network architectures."""
    import tensornets as nets
    print("Available network architectures:")
    for candidate in available_nets():
        print(" ", candidate)

