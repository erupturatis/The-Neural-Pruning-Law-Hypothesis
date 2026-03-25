from typing import List, Optional
import math
import torch
import torch.nn as nn
from src.infrastructure.layers import (
    LayerComposite,
    ConfigsNetworkMasksImportance,
    LayerConv2MaskImportance,
    ConfigsLayerConv2,
    LayerLinearMaskImportance,
    ConfigsLayerLinear,
    LayerPrimitive,
    get_flow_params_loss,
    get_layers_primitive,
    get_layer_composite_flow_params_statistics,
)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        pass
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class PatchEmbed(nn.Module):
    """Image to patch embedding via Conv2d(kernel=stride=patch)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        self.patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        grid = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = grid[0] * grid[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, (
            f"Input size ({H}x{W}) != model ({self.img_size[0]}x{self.img_size[1]})"
        )
        x = self.proj(x)  # [B, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class PrunableAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        configs_network_masks,
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = LayerLinearMaskImportance(
            ConfigsLayerLinear(
                in_features=dim, out_features=3 * dim, bias_enabled=qkv_bias
            ),
            configs_network_masks,
        )
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = LayerLinearMaskImportance(
            ConfigsLayerLinear(in_features=dim, out_features=dim, bias_enabled=True),
            configs_network_masks,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, Dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, heads, N, Dh]
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PrunableMlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float, configs_network_masks):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = LayerLinearMaskImportance(
            ConfigsLayerLinear(in_features=dim, out_features=hidden, bias_enabled=True),
            configs_network_masks,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = LayerLinearMaskImportance(
            ConfigsLayerLinear(in_features=hidden, out_features=dim, bias_enabled=True),
            configs_network_masks,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PrunableBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer,
        configs_network_masks,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrunableAttention(
            dim, num_heads, qkv_bias, attn_drop, drop, configs_network_masks
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = PrunableMlp(dim, mlp_ratio, drop, configs_network_masks)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerPrunable(LayerComposite):
    def __init__(
        self,
        configs_network_masks,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6),
    ):
        super().__init__()
        self.registered_layers: List[LayerPrimitive] = []

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self._patch_as_module = True
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                PrunableBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    configs_network_masks=configs_network_masks,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        for li, blk in enumerate(self.blocks):
            # attention qkv/proj
            self.registered_layers.append(blk.attn.qkv)
            self.registered_layers.append(blk.attn.proj)
            # mlp fc1/fc2
            self.registered_layers.append(blk.mlp.fc1)
            self.registered_layers.append(blk.mlp.fc2)
        # NOTE: head/classifier kept dense (not registered). Register it if you want to prune head:

    def get_remaining_parameters_loss(self) -> torch.Tensor:
        total, sigmoid = get_flow_params_loss(self)
        return sigmoid / total

    def get_parameters_pruning_statistics(self):
        return get_layer_composite_flow_params_statistics(self)

    def get_layers_primitive(self) -> List[LayerPrimitive]:
        return get_layers_primitive(self)

    # ---- init helpers
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    # ---- forward
    def _patch_forward(self, x):
        if self._patch_as_module:
            return self.patch_embed(x)  # [B, N, D]
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_features(self, x):
        B = x.size(0)
        x = self._patch_forward(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Not used at this point - here for reference
def deit_tiny_prunable(configs_network_masks, **kwargs):
    return VisionTransformerPrunable(
        configs_network_masks=configs_network_masks,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    )


def deit_small_prunable(configs_network_masks, **kwargs):
    return VisionTransformerPrunable(
        configs_network_masks=configs_network_masks,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    )


def deit_base_prunable(configs_network_masks, **kwargs):
    return VisionTransformerPrunable(
        configs_network_masks=configs_network_masks,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,
    )
