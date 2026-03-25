from src.common_files_experiments.attributes_mutations import Mutation, mutate_attributes
from src.common_files_experiments.vanilla_attributes_vgg19 import (
    VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING
)
from src.infrastructure.constants import CONV2D_LAYER, FULLY_CONNECTED_LAYER

replace_fc1_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc1',
    action='replace',
    replacement_dict={
        "name": "fc1",
        "type": FULLY_CONNECTED_LAYER,
        "in_features": 512,
        "out_features": 100,
        "bias_enabled": True
    }
)

remove_fc2_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc2',
    action='remove',
)

remove_fc3_cifar100 = Mutation(
    field_identified='name',
    value_in_field='fc3',
    action='remove',
)

cifar100_registered_mutations = [
    replace_fc1_cifar100,
    remove_fc2_cifar100,
    remove_fc3_cifar100,
]

cifar100_removed_mutations = [
    remove_fc2_cifar100,
    remove_fc3_cifar100,
]


remove_fc2_cifar100_mapping = Mutation(
    field_identified='custom_name',
    value_in_field='fc2',
    action='remove',
)

remove_fc3_cifar100_mapping = Mutation(
    field_identified='custom_name',
    value_in_field='fc3',
    action='remove',
)

cifar100_remove_from_mappings = [
    remove_fc2_cifar100_mapping,
    remove_fc3_cifar100_mapping
]

VGG19_CIFAR100_REGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_REGISTERED_LAYERS_ATTRIBUTES,
    mutations=cifar100_registered_mutations
)

VGG19_CIFAR100_UNREGISTERED_LAYERS_ATTRIBUTES = mutate_attributes(
    attributes=VGG19_VANILLA_UNREGISTERED_LAYERS_ATTRIBUTES,
    mutations=[]  # No mutations to apply
)

VGG19_CIFAR100_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_STANDARD_TO_CUSTOM_LAYER_NAME_MAPPING,
    mutations=cifar100_remove_from_mappings
)

VGG19_CIFAR100_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING = mutate_attributes(
    attributes=VGG19_VANILLA_CUSTOM_TO_STANDARD_LAYER_NAME_MAPPING,
    mutations=cifar100_remove_from_mappings
)
