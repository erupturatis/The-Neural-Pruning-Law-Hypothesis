import torch
from src.infrastructure.layers import LayerComposite
import torch.nn.functional as F

def forward_pass_lenet(self: 'LayerComposite', x: torch.Tensor) -> torch.Tensor:
    names = [attr['name'] for attr in self._layer_attributes]
    x = x.view(-1, 28 * 28)
    x = F.relu(getattr(self, names[0])(x))
    x = F.relu(getattr(self, names[1])(x))
    x = getattr(self, names[2])(x)
    return x
