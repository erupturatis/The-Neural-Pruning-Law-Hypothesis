from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER, BATCH_NORM_2D_LAYER


def get_resnet50_variable_cifar10_attributes(alpha: float, num_classes: int = 10):
    if alpha < 1e-2:
        raise ValueError("Alpha is too small for this network")

    # Bottleneck neck widths, scaled by alpha
    p1 = max(4, round(64 * alpha))
    p2 = max(4, round(128 * alpha))
    p3 = max(4, round(256 * alpha))
    p4 = max(4, round(512 * alpha))

    # Layer output widths (bottleneck expansion x4)
    stem = p1       # initial conv output = same as layer1 neck
    o1   = p1 * 4   # layer1 output
    o2   = p2 * 4   # layer2 output
    o3   = p3 * 4   # layer3 output
    o4   = p4 * 4   # layer4 output

    def _conv(name, ic, oc, k, s, p, prunable=True):
        return {"name": name, "type": CONV2D_LAYER,
                "in_channels": ic, "out_channels": oc,
                "kernel_size": k, "stride": s, "padding": p, "bias_enabled": False,
                "prunable": prunable}

    def _fc(name, inf, outf):
        return {"name": name, "type": FULLY_CONNECTED_LAYER,
                "in_features": inf, "out_features": outf, "bias_enabled": True}

    def _bn(name, features):
        return {"name": name, "type": BATCH_NORM_2D_LAYER, "num_features": features}

    registered = []
    unregistered = []

    # ── Initial stem (CIFAR-10: 3×3, stride=1, no maxpool) ───────────────
    registered.append(_conv("conv1", 3, stem, 3, 1, 1))
    unregistered.append(_bn("bn1", stem))

    # ── Layer 1 (3 blocks, no spatial downsampling) ───────────────────────
    # Block 1 — needs downsample (stem → o1)
    registered += [
        _conv("layer1_block1_conv1", stem, p1, 1, 1, 0),
        _conv("layer1_block1_conv2", p1,   p1, 3, 1, 1),
        _conv("layer1_block1_conv3", p1,   o1, 1, 1, 0),
        _conv("layer1_block1_downsample", stem, o1, 1, 1, 0, prunable=False),
    ]
    unregistered += [
        _bn("layer1_block1_bn1", p1),
        _bn("layer1_block1_bn2", p1),
        _bn("layer1_block1_bn3", o1),
        _bn("layer1_block1_downsample_bn", o1),
    ]
    # Blocks 2–3 — no downsample
    for i in range(2, 4):
        registered += [
            _conv(f"layer1_block{i}_conv1", o1, p1, 1, 1, 0),
            _conv(f"layer1_block{i}_conv2", p1, p1, 3, 1, 1),
            _conv(f"layer1_block{i}_conv3", p1, o1, 1, 1, 0),
        ]
        unregistered += [
            _bn(f"layer1_block{i}_bn1", p1),
            _bn(f"layer1_block{i}_bn2", p1),
            _bn(f"layer1_block{i}_bn3", o1),
        ]

    # ── Layer 2 (4 blocks, stride=2 at block 1) ──────────────────────────
    registered += [
        _conv("layer2_block1_conv1", o1, p2, 1, 1, 0),
        _conv("layer2_block1_conv2", p2, p2, 3, 2, 1),   # stride=2
        _conv("layer2_block1_conv3", p2, o2, 1, 1, 0),
        _conv("layer2_block1_downsample", o1, o2, 1, 2, 0, prunable=False),  # stride=2
    ]
    unregistered += [
        _bn("layer2_block1_bn1", p2),
        _bn("layer2_block1_bn2", p2),
        _bn("layer2_block1_bn3", o2),
        _bn("layer2_block1_downsample_bn", o2),
    ]
    for i in range(2, 5):
        registered += [
            _conv(f"layer2_block{i}_conv1", o2, p2, 1, 1, 0),
            _conv(f"layer2_block{i}_conv2", p2, p2, 3, 1, 1),
            _conv(f"layer2_block{i}_conv3", p2, o2, 1, 1, 0),
        ]
        unregistered += [
            _bn(f"layer2_block{i}_bn1", p2),
            _bn(f"layer2_block{i}_bn2", p2),
            _bn(f"layer2_block{i}_bn3", o2),
        ]

    # ── Layer 3 (6 blocks, stride=2 at block 1) ──────────────────────────
    registered += [
        _conv("layer3_block1_conv1", o2, p3, 1, 1, 0),
        _conv("layer3_block1_conv2", p3, p3, 3, 2, 1),   # stride=2
        _conv("layer3_block1_conv3", p3, o3, 1, 1, 0),
        _conv("layer3_block1_downsample", o2, o3, 1, 2, 0, prunable=False),  # stride=2
    ]
    unregistered += [
        _bn("layer3_block1_bn1", p3),
        _bn("layer3_block1_bn2", p3),
        _bn("layer3_block1_bn3", o3),
        _bn("layer3_block1_downsample_bn", o3),
    ]
    for i in range(2, 7):
        registered += [
            _conv(f"layer3_block{i}_conv1", o3, p3, 1, 1, 0),
            _conv(f"layer3_block{i}_conv2", p3, p3, 3, 1, 1),
            _conv(f"layer3_block{i}_conv3", p3, o3, 1, 1, 0),
        ]
        unregistered += [
            _bn(f"layer3_block{i}_bn1", p3),
            _bn(f"layer3_block{i}_bn2", p3),
            _bn(f"layer3_block{i}_bn3", o3),
        ]

    # ── Layer 4 (3 blocks, stride=2 at block 1) ──────────────────────────
    registered += [
        _conv("layer4_block1_conv1", o3, p4, 1, 1, 0),
        _conv("layer4_block1_conv2", p4, p4, 3, 2, 1),   # stride=2
        _conv("layer4_block1_conv3", p4, o4, 1, 1, 0),
        _conv("layer4_block1_downsample", o3, o4, 1, 2, 0, prunable=False),  # stride=2
    ]
    unregistered += [
        _bn("layer4_block1_bn1", p4),
        _bn("layer4_block1_bn2", p4),
        _bn("layer4_block1_bn3", o4),
        _bn("layer4_block1_downsample_bn", o4),
    ]
    for i in range(2, 4):
        registered += [
            _conv(f"layer4_block{i}_conv1", o4, p4, 1, 1, 0),
            _conv(f"layer4_block{i}_conv2", p4, p4, 3, 1, 1),
            _conv(f"layer4_block{i}_conv3", p4, o4, 1, 1, 0),
        ]
        unregistered += [
            _bn(f"layer4_block{i}_bn1", p4),
            _bn(f"layer4_block{i}_bn2", p4),
            _bn(f"layer4_block{i}_bn3", o4),
        ]

    # ── Classifier ────────────────────────────────────────────────────────
    registered.append(_fc("fc", o4, num_classes))

    return registered, unregistered
