from src.infrastructure.layers import ConfigsNetworkMask
from src.model_lenet.model_lenetVariable_class import ModelLenetVariable


class ModelLenet300(ModelLenetVariable):
    def __init__(self, config_network_mask: ConfigsNetworkMask):
        super(ModelLenet300, self).__init__(alpha=1.0, config_network_mask=config_network_mask)
