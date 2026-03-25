from src.infrastructure.constants import FULLY_CONNECTED_LAYER

LENET300_MNIST_REGISTERED_LAYERS_ATTRIBUTES = [
    {
        "name": "fc1",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 28 * 28,  # 784
        "out_features": 300,
        "bias_enabled": True
    },

    {
        "name": "fc2",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 300,
        "out_features": 100,
        "bias_enabled": True
    },

    {
        "name": "fc3",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 100,
        "out_features": 10,
        "bias_enabled": True
    }
]

LENET300_MNIST_UNREGISTERED_LAYERS_ATTRIBUTES = [
]

LENET300_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = [
    {'custom_name': 'fc1', 'standard_name': 'fc1.weight'},

    {'custom_name': 'fc2', 'standard_name': 'fc2.weight'},

    {'custom_name': 'fc3', 'standard_name': 'fc3.weight'},
]

LENET300_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = [
    {'standard_name': 'fc1.weight', 'custom_name': 'fc1'},

    {'standard_name': 'fc2.weight', 'custom_name': 'fc2'},

    {'standard_name': 'fc3.weight', 'custom_name': 'fc3'},
]
