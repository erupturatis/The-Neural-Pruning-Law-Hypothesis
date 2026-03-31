import torch
import torch.nn as nn


def _forward_bottleneck_layer(self, x: torch.Tensor, layer_num: int, num_blocks: int) -> torch.Tensor:
    for block_num in range(1, num_blocks + 1):
        identity = x

        downsample_name = f"layer{layer_num}_block{block_num}_downsample"
        if hasattr(self, downsample_name):
            identity = getattr(self, f"{downsample_name}_bn")(
                getattr(self, downsample_name)(x)
            )

        out = getattr(self, f"layer{layer_num}_block{block_num}_conv1")(x)
        out = getattr(self, f"layer{layer_num}_block{block_num}_bn1")(out)
        out = self.relu(out)

        out = getattr(self, f"layer{layer_num}_block{block_num}_conv2")(out)
        out = getattr(self, f"layer{layer_num}_block{block_num}_bn2")(out)
        out = self.relu(out)

        out = getattr(self, f"layer{layer_num}_block{block_num}_conv3")(out)
        out = getattr(self, f"layer{layer_num}_block{block_num}_bn3")(out)

        out = out + identity
        out = self.relu(out)
        x = out

    return x


def forward_pass_resnet50_cifar10(self, x: torch.Tensor) -> torch.Tensor:
    # Stem — CIFAR-10: 3×3 conv, stride=1, no maxpool
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = _forward_bottleneck_layer(self, x, 1, 3)
    x = _forward_bottleneck_layer(self, x, 2, 4)
    x = _forward_bottleneck_layer(self, x, 3, 6)
    x = _forward_bottleneck_layer(self, x, 4, 3)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
