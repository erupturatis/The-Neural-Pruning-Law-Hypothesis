import torch
import torch.nn.functional as F


def forward_pass_vgg19_cifar(self, x: torch.Tensor) -> torch.Tensor:
    # Block 1
    x = self.relu(self.bn1_1(self.conv1_1(x)))
    x = self.relu(self.bn1_2(self.conv1_2(x)))
    x = self.maxpool(x)

    # Block 2
    x = self.relu(self.bn2_1(self.conv2_1(x)))
    x = self.relu(self.bn2_2(self.conv2_2(x)))
    x = self.maxpool(x)

    # Block 3
    x = self.relu(self.bn3_1(self.conv3_1(x)))
    x = self.relu(self.bn3_2(self.conv3_2(x)))
    x = self.relu(self.bn3_3(self.conv3_3(x)))
    x = self.relu(self.bn3_4(self.conv3_4(x)))
    x = self.maxpool(x)

    # Block 4
    x = self.relu(self.bn4_1(self.conv4_1(x)))
    x = self.relu(self.bn4_2(self.conv4_2(x)))
    x = self.relu(self.bn4_3(self.conv4_3(x)))
    x = self.relu(self.bn4_4(self.conv4_4(x)))
    x = self.maxpool(x)

    # Block 5
    x = self.relu(self.bn5_1(self.conv5_1(x)))
    x = self.relu(self.bn5_2(self.conv5_2(x)))
    x = self.relu(self.bn5_3(self.conv5_3(x)))
    x = self.relu(self.bn5_4(self.conv5_4(x)))

    # Final Average Pool
    x = F.avg_pool2d(x, kernel_size=2)

    # Flatten
    x = torch.flatten(x, 1)

    # Classifier
    x = self.fc1(x)

    return x
