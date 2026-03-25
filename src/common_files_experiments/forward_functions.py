from types import SimpleNamespace
import torch
from src.infrastructure.layers import LayerComposite

def forward_pass_vgg19_cifars_version2(
        self: 'LayerComposite',
        x: torch.Tensor,
        registered_layers_attributes,
        unregistered_layers_attributes
) -> torch.Tensor:
    # Create local shortcuts for registered/unregistered layers
    registered_layers_object = SimpleNamespace()
    for layer in registered_layers_attributes:
        name = layer['name']
        layer_instance = getattr(self, name)
        setattr(registered_layers_object, name, layer_instance)

    unregistered_layers_object = SimpleNamespace()
    for layer in unregistered_layers_attributes:
        name = layer['name']
        layer_instance = getattr(self, name)
        setattr(unregistered_layers_object, name, layer_instance)

    # ------------------ Block 1 ------------------
    x = registered_layers_object.conv1_1(x)
    x = unregistered_layers_object.bn1_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv1_2(x)
    x = unregistered_layers_object.bn1_2(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # ------------------ Block 2 ------------------
    x = registered_layers_object.conv2_1(x)
    x = unregistered_layers_object.bn2_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv2_2(x)
    x = unregistered_layers_object.bn2_2(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # ------------------ Block 3 ------------------
    x = registered_layers_object.conv3_1(x)
    x = unregistered_layers_object.bn3_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_2(x)
    x = unregistered_layers_object.bn3_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_3(x)
    x = unregistered_layers_object.bn3_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv3_4(x)
    x = unregistered_layers_object.bn3_4(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # ------------------ Block 4 ------------------
    x = registered_layers_object.conv4_1(x)
    x = unregistered_layers_object.bn4_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_2(x)
    x = unregistered_layers_object.bn4_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_3(x)
    x = unregistered_layers_object.bn4_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv4_4(x)
    x = unregistered_layers_object.bn4_4(x)
    x = torch.relu(x)
    x = self.maxpool(x)

    # ------------------ Block 5 ------------------
    x = registered_layers_object.conv5_1(x)
    x = unregistered_layers_object.bn5_1(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_2(x)
    x = unregistered_layers_object.bn5_2(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_3(x)
    x = unregistered_layers_object.bn5_3(x)
    x = torch.relu(x)
    x = registered_layers_object.conv5_4(x)
    x = unregistered_layers_object.bn5_4(x)
    x = torch.relu(x)
    # grasp repo applies a avg pool not maxpool
    # x = self.maxpool(x)

    # ------------------ Final Average Pool ------------------
    x = torch.nn.functional.avg_pool2d(x, kernel_size=2)

    # Flatten
    x = torch.flatten(x, 1)

    # Single FC classifier
    x = registered_layers_object.fc1(x)

    return x


def forward_pass_resnet50_imagenet(self: 'LayerComposite', x: torch.Tensor, registered_layer_attributes, unregistered_layer_attributes) -> torch.Tensor:
    registered_layers_object = SimpleNamespace()
    for layer_attr in registered_layer_attributes:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer_attr in unregistered_layer_attributes:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(unregistered_layers_object, name, layer)

    x = registered_layers_object.conv1(x)
    x = unregistered_layers_object.bn1(x)
    x = self.relu(x)
    x = unregistered_layers_object.maxpool1(x)

    layers_config = [
        {'layer_num': 1, 'num_blocks': 3},
        {'layer_num': 2, 'num_blocks': 4},
        {'layer_num': 3, 'num_blocks': 6},
        {'layer_num': 4, 'num_blocks': 3},
    ]

    # Process each layer
    for layer_info in layers_config:
        layer_num = layer_info['layer_num']
        num_blocks = layer_info['num_blocks']
        x = _forward_layer_resnet50(self, x, layer_num, num_blocks, registered_layers_object, unregistered_layers_object)

    x = unregistered_layers_object.avgpool(x)
    x = torch.flatten(x, 1)
    x = registered_layers_object.fc(x)

    return x

def forward_pass_resnet50_cifars(self: 'LayerComposite', x: torch.Tensor, registered_layer_attributes, unregistered_layer_attributes) -> torch.Tensor:
    registered_layers_object = SimpleNamespace()
    for layer_attr in registered_layer_attributes:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer_attr in unregistered_layer_attributes:
        name = layer_attr['name']
        layer = getattr(self, name)
        setattr(unregistered_layers_object, name, layer)

    # Initial layers without Max Pooling
    x = registered_layers_object.conv1(x)
    x = unregistered_layers_object.bn1(x)
    x = self.relu(x)

    layers_config = [
        {'layer_num': 1, 'num_blocks': 3},
        {'layer_num': 2, 'num_blocks': 4},
        {'layer_num': 3, 'num_blocks': 6},
        {'layer_num': 4, 'num_blocks': 3},
    ]

    # Process each layer
    for layer_info in layers_config:
        layer_num = layer_info['layer_num']
        num_blocks = layer_info['num_blocks']
        x = _forward_layer_resnet50(self, x, layer_num, num_blocks, registered_layers_object, unregistered_layers_object)

    x = unregistered_layers_object.avgpool(x)
    x = torch.flatten(x, 1)
    x = registered_layers_object.fc(x)

    return x



def _forward_layer_resnet50(self, x, layer_num, num_blocks, registered_layers_object, unregistered_layers_object):
    for block_num in range(1, num_blocks + 1):
        identity = x

        downsample_name = f'layer{layer_num}_block{block_num}_downsample'
        if hasattr(registered_layers_object, downsample_name):
            downsample_conv = getattr(registered_layers_object, downsample_name)
            downsample_bn = getattr(unregistered_layers_object, f'{downsample_name}_bn')
            identity = downsample_bn(downsample_conv(x))

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv1')(x)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn1')(out)
        out = self.relu(out)

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv2')(out)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn2')(out)
        out = self.relu(out)

        out = getattr(registered_layers_object, f'layer{layer_num}_block{block_num}_conv3')(out)
        out = getattr(unregistered_layers_object, f'layer{layer_num}_block{block_num}_bn3')(out)

        out += identity
        out = self.relu(out)

        x = out

    return x


def forward_pass_resnet18_cifars(self: 'LayerComposite', x: torch.Tensor, registered_layers, unregistered_layers) -> torch.Tensor:
    # Ensures all layers used in forward are registered in these 2 arrays
    registered_layers_object = SimpleNamespace()
    for layer in registered_layers:
        name = layer['name']
        layer = getattr(self, name)
        setattr(registered_layers_object, name, layer)

    unregistered_layers_object = SimpleNamespace()
    for layer in unregistered_layers:
        name = layer['name']
        layer = getattr(self, name)
        setattr(unregistered_layers_object, name, layer)

    # Initial layers
    x = registered_layers_object.conv1(x)
    x = unregistered_layers_object.bn1(x)
    x = self.relu(x)

    # Layer 1
    # Block 1
    identity = x
    out = registered_layers_object.layer1_block1_conv1(x)
    out = unregistered_layers_object.layer1_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer1_block1_conv2(out)
    out = unregistered_layers_object.layer1_block1_bn2(out)
    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer1_block2_conv1(out)
    out = unregistered_layers_object.layer1_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer1_block2_conv2(out)
    out = unregistered_layers_object.layer1_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 2
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer2_block1_conv1(out)
    out = unregistered_layers_object.layer2_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer2_block1_conv2(out)
    out = unregistered_layers_object.layer2_block1_bn2(out)

    identity = registered_layers_object.layer2_block1_downsample(identity)
    identity = unregistered_layers_object.layer2_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer2_block2_conv1(out)
    out = unregistered_layers_object.layer2_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer2_block2_conv2(out)
    out = unregistered_layers_object.layer2_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 3
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer3_block1_conv1(out)
    out = unregistered_layers_object.layer3_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer3_block1_conv2(out)
    out = unregistered_layers_object.layer3_block1_bn2(out)

    identity = registered_layers_object.layer3_block1_downsample(identity)
    identity = unregistered_layers_object.layer3_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer3_block2_conv1(out)
    out = unregistered_layers_object.layer3_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer3_block2_conv2(out)
    out = unregistered_layers_object.layer3_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Layer 4
    # Block 1 with downsampling
    identity = out
    out = registered_layers_object.layer4_block1_conv1(out)
    out = unregistered_layers_object.layer4_block1_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer4_block1_conv2(out)
    out = unregistered_layers_object.layer4_block1_bn2(out)

    identity = registered_layers_object.layer4_block1_downsample(identity)
    identity = unregistered_layers_object.layer4_block1_downsample_bn(identity)

    out += identity
    out = self.relu(out)

    # Block 2
    identity = out
    out = registered_layers_object.layer4_block2_conv1(out)
    out = unregistered_layers_object.layer4_block2_bn1(out)
    out = self.relu(out)
    out = registered_layers_object.layer4_block2_conv2(out)
    out = unregistered_layers_object.layer4_block2_bn2(out)
    out += identity
    out = self.relu(out)

    # Final layers
    out = self.avgpool(out)
    out = torch.flatten(out, 1)
    out = registered_layers_object.fc(out)

    return out

