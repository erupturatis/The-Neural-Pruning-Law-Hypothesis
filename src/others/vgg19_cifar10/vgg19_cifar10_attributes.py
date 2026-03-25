from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_vgg19 import (
    VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

replace_fc1_cifar10 = Mutation(
    field_identified='name',
    value_in_field='fc1',
    action='replace',
    replacement_dict={
        "name": "fc1",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 512,
        "out_features": 10,
        "bias_enabled": True
    }
)

remove_fc2_cifar10 = Mutation(
    field_identified='name',
    value_in_field='fc2',
    action='remove',
)

remove_fc3_cifar10 = Mutation(
    field_identified='name',
    value_in_field='fc3',
    action='remove',
)

cifar10_registered_mutations = [
    replace_fc1_cifar10,
    remove_fc2_cifar10,
    remove_fc3_cifar10,
]

cifar10_removed_mutations = [
    remove_fc2_cifar10,
    remove_fc3_cifar10,
]


remove_fc2_cifar10_mapping = Mutation(
    field_identified='custom_name',
    value_in_field='fc2',
    action='remove',
)

remove_fc3_cifar10_mapping = Mutation(
    field_identified='custom_name',
    value_in_field='fc3',
    action='remove',
)

cifar10_remove_from_mappings = [
    remove_fc2_cifar10_mapping,
    remove_fc3_cifar10_mapping
]

# Apply mutations to the registered layers for CIFAR-10
VGG19_CIFAR10_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar10_registered_mutations
)

# Unregistered layers remain unchanged for CIFAR-10
VGG19_CIFAR10_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

# Layer name mappings remain unchanged for CIFAR-10
VGG19_CIFAR10_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=cifar10_remove_from_mappings  # No mutations to apply
)

VGG19_CIFAR10_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=cifar10_remove_from_mappings  # No mutations to apply
)
