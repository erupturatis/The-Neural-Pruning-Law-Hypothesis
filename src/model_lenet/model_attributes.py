from src.infrastructure.constants import FULLY_CONNECTED_LAYER

def get_lenet_variable_attributes(alpha: float):
    if alpha < 10 ** -2:
        raise Exception("Alpha is too small for this network")

    h1 = max(1, int(300 * alpha))
    h2 = max(1, int(100 * alpha))

    return [
        {
            "name": "fc1",
            "type": FULLY_CONNECTED_LAYER,
            "in_features": 28 * 28,
            "out_features": h1,
            "bias_enabled": True,
            "save": True,
        },
        {
            "name": "fc2",
            "type": FULLY_CONNECTED_LAYER,
            "in_features": h1,
            "out_features": h2,
            "bias_enabled": True,
            "save": True,
        },
        {
            "name": "fc3",
            "type": FULLY_CONNECTED_LAYER,
            "in_features": h2,
            "out_features": 10,
            "bias_enabled": True,
            "save": True,
        },
    ]
