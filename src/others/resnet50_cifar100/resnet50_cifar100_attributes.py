from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_resnet50 import (
    RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

replace_conv1 = Mutation(
    field_identified='name',
    value_in_field='conv1',
    action='replace',
    replacement_dict={
        "name": "conv1",
        "type": CONV2D_LAYER,
        "in_channels": 3,
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias_enabled": False
    }
)

replace_fc = Mutation(
    field_identified='name',
    value_in_field='fc',
    action='replace',
    replacement_dict={
        "name": "fc",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 2048,
        "out_features": 100,
        "bias_enabled": True
    }
)

remove_maxpool1 = Mutation(
    field_identified='name',
    value_in_field='maxpool1',
    action='remove',
    replacement_dict={}
)

cifar100_registered_mutations = [
    replace_conv1,
    replace_fc
]

cifar10_unregistered_layers_mutations = [
    remove_maxpool1,
]

RESNET50_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET50_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

RESNET50_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=RESNET50_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar10_unregistered_layers_mutations  # No mutations to apply
)

RESNET50_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET50_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)

RESNET50_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=RESNET50_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=[]  # No mutations to apply
)
