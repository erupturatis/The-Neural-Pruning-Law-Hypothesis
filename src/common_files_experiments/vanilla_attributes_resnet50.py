from src.infrastructure.constants import N_SCALER, PRUNED_MODELS_PATH, CONV2D_LAYER, FULLY_CONNECTED_LAYER, \
    BATCH_NORM_2D_LAYER

RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES = [
    # Initial convolutional layer
    {"name": "conv1", "type": CONV2D_LAYER, "in_channels": 3, "out_channels": 64,
     "kernel_size": 7, "stride": 2, "padding": 3, "bias_enabled": False},

    # Layer 1 - Block 1
    {"name": "layer1_block1_conv1", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer1_block1_conv2", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer1_block1_conv3", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer1_block1_downsample", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},

    # Layer 1 - Block 2
    {"name": "layer1_block2_conv1", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer1_block2_conv2", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer1_block2_conv3", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},

    # Layer 1 - Block 3
    {"name": "layer1_block3_conv1", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer1_block3_conv2", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "layer1_block3_conv3", "type": CONV2D_LAYER, "in_channels": 64,
     "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
]

# Layer 2 - Blocks 1
RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
    {"name": "layer2_block1_conv1", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer2_block1_conv2", "type": CONV2D_LAYER, "in_channels": 128,
     "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer2_block1_conv3", "type": CONV2D_LAYER, "in_channels": 128,
     "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer2_block1_downsample", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 512, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
])

# Layer 2 - Blocks 2 to 4
for i in range(2, 5):
    RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
        {"name": f"layer2_block{i}_conv1", "type": CONV2D_LAYER, "in_channels": 512,
         "out_channels": 128, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
        {"name": f"layer2_block{i}_conv2", "type": CONV2D_LAYER, "in_channels": 128,
         "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
        {"name": f"layer2_block{i}_conv3", "type": CONV2D_LAYER, "in_channels": 128,
         "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    ])

# Layer 3 - Blocks 1
RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
    # Layer 3 - Block 1 (Modified Stride Placement)
    {"name": "layer3_block1_conv1", "type": CONV2D_LAYER, "in_channels": 512,
     "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer3_block1_conv2", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer3_block1_conv3", "type": CONV2D_LAYER, "in_channels": 256,
     "out_channels": 1024, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer3_block1_downsample", "type": CONV2D_LAYER, "in_channels": 512,
     "out_channels": 1024, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
])

# Layer 3 - Blocks 2 to 6
for i in range(2, 7):
    RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
        {"name": f"layer3_block{i}_conv1", "type": CONV2D_LAYER, "in_channels": 1024,
         "out_channels": 256, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
        {"name": f"layer3_block{i}_conv2", "type": CONV2D_LAYER, "in_channels": 256,
         "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
        {"name": f"layer3_block{i}_conv3", "type": CONV2D_LAYER, "in_channels": 256,
         "out_channels": 1024, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    ])

# Layer 4 - Blocks 1
RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
    {"name": "layer4_block1_conv1", "type": CONV2D_LAYER, "in_channels": 1024,
     "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer4_block1_conv2", "type": CONV2D_LAYER, "in_channels": 512,
     "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1, "bias_enabled": False},
    {"name": "layer4_block1_conv3", "type": CONV2D_LAYER, "in_channels": 512,
     "out_channels": 2048, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    {"name": "layer4_block1_downsample", "type": CONV2D_LAYER, "in_channels": 1024,
     "out_channels": 2048, "kernel_size": 1, "stride": 2, "padding": 0, "bias_enabled": False},
])

# Layer 4 - Blocks 2 to 3
for i in range(2, 4):
    RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.extend([
        {"name": f"layer4_block{i}_conv1", "type": CONV2D_LAYER, "in_channels": 2048,
         "out_channels": 512, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
        {"name": f"layer4_block{i}_conv2", "type": CONV2D_LAYER, "in_channels": 512,
         "out_channels": 512, "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
        {"name": f"layer4_block{i}_conv3", "type": CONV2D_LAYER, "in_channels": 512,
         "out_channels": 2048, "kernel_size": 1, "stride": 1, "padding": 0, "bias_enabled": False},
    ])

# Fully connected layer
RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES.append(
    {"name": "fc", "type": FULLY_CONNECTED_LAYER, "in_features": 2048, "out_features": 1000, "bias_enabled": True}
)


# Unregistered layers (e.g., batch norms, activations, pooling)
RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES = [
    # Initial batch normalization and pooling
    {"name": "bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "maxpool1", "type": "MaxPool2d", "kernel_size": 3, "stride": 2, "padding": 1},

    # Layer 1 - Block 1
    {"name": "layer1_block1_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block1_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block1_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
    {"name": "layer1_block1_downsample_bn", "type": BATCH_NORM_2D_LAYER, "num_features": 256},

    # Layer 1 - Block 2
    {"name": "layer1_block2_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block2_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block2_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 256},

    # Layer 1 - Block 3
    {"name": "layer1_block3_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block3_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 64},
    {"name": "layer1_block3_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
]

# Layer 2 - Blocks 1 to 4
for i in range(1, 5):
    if i == 1:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer2_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 128},
            {"name": f"layer2_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 128},
            {"name": f"layer2_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
            {"name": f"layer2_block{i}_downsample_bn", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
        ])
    else:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer2_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 128},
            {"name": f"layer2_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 128},
            {"name": f"layer2_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
        ])

# Layer 3 - Blocks 1 to 6
for i in range(1, 7):
    if i == 1:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer3_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
            {"name": f"layer3_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
            {"name": f"layer3_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 1024},
            {"name": f"layer3_block{i}_downsample_bn", "type": BATCH_NORM_2D_LAYER, "num_features": 1024},
        ])
    else:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer3_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
            {"name": f"layer3_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 256},
            {"name": f"layer3_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 1024},
        ])

# Layer 4 - Blocks 1 to 3
for i in range(1, 4):
    if i == 1:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer4_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
            {"name": f"layer4_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
            {"name": f"layer4_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 2048},
            {"name": f"layer4_block{i}_downsample_bn", "type": BATCH_NORM_2D_LAYER, "num_features": 2048},
        ])
    else:
        RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.extend([
            {"name": f"layer4_block{i}_bn1", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
            {"name": f"layer4_block{i}_bn2", "type": BATCH_NORM_2D_LAYER, "num_features": 512},
            {"name": f"layer4_block{i}_bn3", "type": BATCH_NORM_2D_LAYER, "num_features": 2048},
        ])

# Average pooling layer
RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES.append(
    {"name": "avgpool", "type": "AdaptiveAvgPool2d", "output_size": (1, 1)}
)


# Mapping between custom and standard layer names
RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = [
    # Initial layers
    {'custom_name': 'conv1', 'standard_name': 'conv1.weight'},
    {'custom_name': 'bn1', 'standard_name': 'bn1'},

    # Layer 1 - Blocks 1 to 3
    *[
        {
            'custom_name': f'layer1_block{i}_conv{j}',
            'standard_name': f'layer1.{i-1}.conv{j}.weight'
        }
        for i in range(1, 4) for j in range(1, 4)
    ],
    *[
        {
            'custom_name': f'layer1_block{i}_bn{j}',
            'standard_name': f'layer1.{i-1}.bn{j}'
        }
        for i in range(1, 4) for j in range(1, 4)
    ],
    {
        'custom_name': 'layer1_block1_downsample',
        'standard_name': 'layer1.0.downsample.0.weight'
    },
    {
        'custom_name': 'layer1_block1_downsample_bn',
        'standard_name': 'layer1.0.downsample.1'
    },

    # Layer 2 - Blocks 1 to 4
    *[
        {
            'custom_name': f'layer2_block{i}_conv{j}',
            'standard_name': f'layer2.{i-1}.conv{j}.weight'
        }
        for i in range(1, 5) for j in range(1, 4)
    ],
    *[
        {
            'custom_name': f'layer2_block{i}_bn{j}',
            'standard_name': f'layer2.{i-1}.bn{j}'
        }
        for i in range(1, 5) for j in range(1, 4)
    ],
    {
        'custom_name': 'layer2_block1_downsample',
        'standard_name': 'layer2.0.downsample.0.weight'
    },
    {
        'custom_name': 'layer2_block1_downsample_bn',
        'standard_name': 'layer2.0.downsample.1'
    },

    # Layer 3 - Blocks 1 to 6
    *[
        {
            'custom_name': f'layer3_block{i}_conv{j}',
            'standard_name': f'layer3.{i-1}.conv{j}.weight'
        }
        for i in range(1, 7) for j in range(1, 4)
    ],
    *[
        {
            'custom_name': f'layer3_block{i}_bn{j}',
            'standard_name': f'layer3.{i-1}.bn{j}'
        }
        for i in range(1, 7) for j in range(1, 4)
    ],
    {
        'custom_name': 'layer3_block1_downsample',
        'standard_name': 'layer3.0.downsample.0.weight'
    },
    {
        'custom_name': 'layer3_block1_downsample_bn',
        'standard_name': 'layer3.0.downsample.1'
    },

    # Layer 4 - Blocks 1 to 3
    *[
        {
            'custom_name': f'layer4_block{i}_conv{j}',
            'standard_name': f'layer4.{i-1}.conv{j}.weight'
        }
        for i in range(1, 4) for j in range(1, 4)
    ],
    *[
        {
            'custom_name': f'layer4_block{i}_bn{j}',
            'standard_name': f'layer4.{i-1}.bn{j}'
        }
        for i in range(1, 4) for j in range(1, 4)
    ],
    {
        'custom_name': 'layer4_block1_downsample',
        'standard_name': 'layer4.0.downsample.0.weight'
    },
    {
        'custom_name': 'layer4_block1_downsample_bn',
        'standard_name': 'layer4.0.downsample.1'
    },

    # Fully connected layer
    {'custom_name': 'fc', 'standard_name': 'fc.weight'},
]

# Create the STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING by swapping keys and values
RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = [
    {'standard_name': mapping['standard_name'], 'custom_name': mapping['custom_name']}
    for mapping in RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
]
