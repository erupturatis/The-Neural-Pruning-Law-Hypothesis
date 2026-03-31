from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, BATCH_NORM_2D_LAYER


def get_vgg19_variable_cifar_attributes(alpha: float, num_classes: int = 10):
    if alpha < 1e-2:
        raise ValueError("Alpha is too small for this network")

    # Scaling function for widths
    def scaled(w):
        return max(1, round(w * alpha))

    # Original VGG widths
    w1, w2, w3, w4, w5 = 64, 128, 256, 512, 512

    # Registered layers
    def _conv(name, ic, oc, k, s, p):
        return {"name": name, "type": CONV2D_LAYER,
                "in_channels": ic, "out_channels": oc,
                "kernel_size": k, "stride": s, "padding": p, "bias_enabled": False}

    def _fc(name, inf, outf):
        return {"name": name, "type": FULLY_CONNECTED_LAYER,
                "in_features": inf, "out_features": outf, "bias_enabled": True}

    def _bn(name, features):
        return {"name": name, "type": BATCH_NORM_2D_LAYER, "num_features": features}

    registered = []
    unregistered = []

    # Block 1
    registered.append(_conv("conv1_1", 3, scaled(w1), 3, 1, 1))
    unregistered.append(_bn("bn1_1", scaled(w1)))
    registered.append(_conv("conv1_2", scaled(w1), scaled(w1), 3, 1, 1))
    unregistered.append(_bn("bn1_2", scaled(w1)))

    # Block 2
    registered.append(_conv("conv2_1", scaled(w1), scaled(w2), 3, 1, 1))
    unregistered.append(_bn("bn2_1", scaled(w2)))
    registered.append(_conv("conv2_2", scaled(w2), scaled(w2), 3, 1, 1))
    unregistered.append(_bn("bn2_2", scaled(w2)))

    # Block 3
    registered.append(_conv("conv3_1", scaled(w2), scaled(w3), 3, 1, 1))
    unregistered.append(_bn("bn3_1", scaled(w3)))
    registered.append(_conv("conv3_2", scaled(w3), scaled(w3), 3, 1, 1))
    unregistered.append(_bn("bn3_2", scaled(w3)))
    registered.append(_conv("conv3_3", scaled(w3), scaled(w3), 3, 1, 1))
    unregistered.append(_bn("bn3_3", scaled(w3)))
    registered.append(_conv("conv3_4", scaled(w3), scaled(w3), 3, 1, 1))
    unregistered.append(_bn("bn3_4", scaled(w3)))

    # Block 4
    registered.append(_conv("conv4_1", scaled(w3), scaled(w4), 3, 1, 1))
    unregistered.append(_bn("bn4_1", scaled(w4)))
    registered.append(_conv("conv4_2", scaled(w4), scaled(w4), 3, 1, 1))
    unregistered.append(_bn("bn4_2", scaled(w4)))
    registered.append(_conv("conv4_3", scaled(w4), scaled(w4), 3, 1, 1))
    unregistered.append(_bn("bn4_3", scaled(w4)))
    registered.append(_conv("conv4_4", scaled(w4), scaled(w4), 3, 1, 1))
    unregistered.append(_bn("bn4_4", scaled(w4)))

    # Block 5
    registered.append(_conv("conv5_1", scaled(w4), scaled(w5), 3, 1, 1))
    unregistered.append(_bn("bn5_1", scaled(w5)))
    registered.append(_conv("conv5_2", scaled(w5), scaled(w5), 3, 1, 1))
    unregistered.append(_bn("bn5_2", scaled(w5)))
    registered.append(_conv("conv5_3", scaled(w5), scaled(w5), 3, 1, 1))
    unregistered.append(_bn("bn5_3", scaled(w5)))
    registered.append(_conv("conv5_4", scaled(w5), scaled(w5), 3, 1, 1))
    unregistered.append(_bn("bn5_4", scaled(w5)))

    # Classifier
    # After AvgPool2d(2) on 2x2 output of block 5, we have 1x1xscaled(w5)
    registered.append(_fc("fc1", scaled(w5), num_classes))

    return registered, unregistered
