from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

# Registered layers for VGG19_bn
VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES = [
    # Block 1
    {"name": "conv1_1", "type": CONV2D_LAYER, "in_channels": 3, "out_channels": 64,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv1_2", "type": CONV2D_LAYER, "in_channels": 64, "out_channels": 64,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Block 2
    {"name": "conv2_1", "type": CONV2D_LAYER, "in_channels": 64, "out_channels": 128,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv2_2", "type": CONV2D_LAYER, "in_channels": 128, "out_channels": 128,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Block 3
    {"name": "conv3_1", "type": CONV2D_LAYER, "in_channels": 128, "out_channels": 256,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv3_2", "type": CONV2D_LAYER, "in_channels": 256, "out_channels": 256,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv3_3", "type": CONV2D_LAYER, "in_channels": 256, "out_channels": 256,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv3_4", "type": CONV2D_LAYER, "in_channels": 256, "out_channels": 256,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Block 4
    {"name": "conv4_1", "type": CONV2D_LAYER, "in_channels": 256, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv4_2", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv4_3", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv4_4", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Block 5
    {"name": "conv5_1", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv5_2", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv5_3", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},
    {"name": "conv5_4", "type": CONV2D_LAYER, "in_channels": 512, "out_channels": 512,
     "kernel_size": 3, "stride": 1, "padding": 1, "bias_enabled": False},

    # Fully connected layers
    {"name": "fc1", "type": FULLY_CONNECTED_LAYER, "in_features": 25088, "out_features": 4096, "bias_enabled": True},
    {"name": "fc2", "type": FULLY_CONNECTED_LAYER, "in_features": 4096, "out_features": 4096, "bias_enabled": True},
    {"name": "fc3", "type": FULLY_CONNECTED_LAYER, "in_features": 4096, "out_features": 1000, "bias_enabled": True},
]

# Unregistered layers for VGG19_bn (BatchNorm layers)
VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES = [
    # BatchNorm layers for Block 1
    {"name": "bn1_1", "type": "BatchNorm2d", "num_features": 64},
    {"name": "bn1_2", "type": "BatchNorm2d", "num_features": 64},

    # BatchNorm layers for Block 2
    {"name": "bn2_1", "type": "BatchNorm2d", "num_features": 128},
    {"name": "bn2_2", "type": "BatchNorm2d", "num_features": 128},

    # BatchNorm layers for Block 3
    {"name": "bn3_1", "type": "BatchNorm2d", "num_features": 256},
    {"name": "bn3_2", "type": "BatchNorm2d", "num_features": 256},
    {"name": "bn3_3", "type": "BatchNorm2d", "num_features": 256},
    {"name": "bn3_4", "type": "BatchNorm2d", "num_features": 256},

    # BatchNorm layers for Block 4
    {"name": "bn4_1", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn4_2", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn4_3", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn4_4", "type": "BatchNorm2d", "num_features": 512},

    # BatchNorm layers for Block 5
    {"name": "bn5_1", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn5_2", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn5_3", "type": "BatchNorm2d", "num_features": 512},
    {"name": "bn5_4", "type": "BatchNorm2d", "num_features": 512},
]

# Custom to Standard layer name mapping for VGG19_bn
VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = [
    # Block 1
    {'custom_name': 'conv1_1', 'standard_name': 'features.0.weight'},
    {'custom_name': 'bn1_1', 'standard_name': 'features.1.weight'},
    {'custom_name': 'conv1_2', 'standard_name': 'features.3.weight'},
    {'custom_name': 'bn1_2', 'standard_name': 'features.4.weight'},

    # Block 2
    {'custom_name': 'conv2_1', 'standard_name': 'features.7.weight'},
    {'custom_name': 'bn2_1', 'standard_name': 'features.8.weight'},
    {'custom_name': 'conv2_2', 'standard_name': 'features.10.weight'},
    {'custom_name': 'bn2_2', 'standard_name': 'features.11.weight'},

    # Block 3
    {'custom_name': 'conv3_1', 'standard_name': 'features.14.weight'},
    {'custom_name': 'bn3_1', 'standard_name': 'features.15.weight'},
    {'custom_name': 'conv3_2', 'standard_name': 'features.17.weight'},
    {'custom_name': 'bn3_2', 'standard_name': 'features.18.weight'},
    {'custom_name': 'conv3_3', 'standard_name': 'features.20.weight'},
    {'custom_name': 'bn3_3', 'standard_name': 'features.21.weight'},
    {'custom_name': 'conv3_4', 'standard_name': 'features.23.weight'},
    {'custom_name': 'bn3_4', 'standard_name': 'features.24.weight'},

    # Block 4
    {'custom_name': 'conv4_1', 'standard_name': 'features.27.weight'},
    {'custom_name': 'bn4_1', 'standard_name': 'features.28.weight'},
    {'custom_name': 'conv4_2', 'standard_name': 'features.30.weight'},
    {'custom_name': 'bn4_2', 'standard_name': 'features.31.weight'},
    {'custom_name': 'conv4_3', 'standard_name': 'features.33.weight'},
    {'custom_name': 'bn4_3', 'standard_name': 'features.34.weight'},
    {'custom_name': 'conv4_4', 'standard_name': 'features.36.weight'},
    {'custom_name': 'bn4_4', 'standard_name': 'features.37.weight'},

    # Block 5
    {'custom_name': 'conv5_1', 'standard_name': 'features.40.weight'},
    {'custom_name': 'bn5_1', 'standard_name': 'features.41.weight'},
    {'custom_name': 'conv5_2', 'standard_name': 'features.43.weight'},
    {'custom_name': 'bn5_2', 'standard_name': 'features.44.weight'},
    {'custom_name': 'conv5_3', 'standard_name': 'features.46.weight'},
    {'custom_name': 'bn5_3', 'standard_name': 'features.47.weight'},
    {'custom_name': 'conv5_4', 'standard_name': 'features.49.weight'},
    {'custom_name': 'bn5_4', 'standard_name': 'features.50.weight'},

    # Fully connected layers
    {'custom_name': 'fc1', 'standard_name': 'classifier.0.weight'},
    {'custom_name': 'fc2', 'standard_name': 'classifier.3.weight'},
    {'custom_name': 'fc3', 'standard_name': 'classifier.6.weight'},
]

VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = [
    {'standard_name': mapping['standard_name'], 'custom_name': mapping['custom_name']}
    for mapping in VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
]

