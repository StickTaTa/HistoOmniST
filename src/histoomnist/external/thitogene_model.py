from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def squash(x: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return (1.0 - 1.0 / (torch.exp(norm) + eps)) * (x / (norm + eps))


class ODConvAttention(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        *,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        min_channel: int = 16,
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), int(min_channel))
        self.kernel_size = int(kernel_size)
        self.kernel_num = int(kernel_num)
        self.temperature = 1.0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU()
        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.filter_fc = None if in_planes == groups and in_planes == out_planes else nn.Conv2d(
            attention_channel,
            out_planes,
            1,
            bias=True,
        )
        self.spatial_fc = None if kernel_size == 1 else nn.Conv2d(
            attention_channel,
            kernel_size * kernel_size,
            1,
            bias=True,
        )
        self.kernel_fc = None if kernel_num == 1 else nn.Conv2d(attention_channel, kernel_num, 1, bias=True)

    @staticmethod
    def _skip(_: torch.Tensor) -> float:
        return 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]:
        h = self.relu(self.bn(self.fc(self.avgpool(x))))
        channel = torch.sigmoid(self.channel_fc(h).view(x.size(0), -1, 1, 1) / self.temperature)
        filt: torch.Tensor | float
        spatial: torch.Tensor | float
        kernel: torch.Tensor | float
        filt = 1.0 if self.filter_fc is None else torch.sigmoid(self.filter_fc(h).view(x.size(0), -1, 1, 1))
        if self.spatial_fc is None:
            spatial = 1.0
        else:
            spatial = torch.sigmoid(
                self.spatial_fc(h).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) / self.temperature
            )
        if self.kernel_fc is None:
            kernel = 1.0
        else:
            kernel = F.softmax(self.kernel_fc(h).view(x.size(0), -1, 1, 1, 1, 1) / self.temperature, dim=1)
        return channel, filt, spatial, kernel


class ODConv2d(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        kernel_num: int = 4,
    ):
        super().__init__()
        self.in_planes = int(in_planes)
        self.out_planes = int(out_planes)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.groups = int(groups)
        self.kernel_num = int(kernel_num)
        self.attention = ODConvAttention(
            self.in_planes,
            self.out_planes,
            self.kernel_size,
            groups=self.groups,
            kernel_num=self.kernel_num,
        )
        self.weight = nn.Parameter(
            torch.randn(self.kernel_num, self.out_planes, self.in_planes // self.groups, self.kernel_size, self.kernel_size)
        )
        for idx in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[idx], mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, _, height, width = x.shape
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            -1,
            self.in_planes // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        out = F.conv2d(
            x,
            weight=aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups * batch_size,
        )
        out = out.view(batch_size, self.out_planes, out.size(-2), out.size(-1))
        return out * filter_attention


class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, num_capsules: int, dim_capsules: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            padding=0,
        )
        self.num_capsules = int(num_capsules)
        self.dim_capsules = int(dim_capsules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise_conv(x)
        expected = self.num_capsules * self.dim_capsules
        if out.shape[1] * out.shape[2] * out.shape[3] != expected:
            raise ValueError(f"Capsule feature map shape {tuple(out.shape)} cannot be viewed as {expected} values.")
        out = out.view(out.size(0), self.num_capsules, self.dim_capsules)
        return squash(out)


class RoutingLayer(nn.Module):
    def __init__(self, num_capsules: int, dim_capsules: int):
        super().__init__()
        self.num_capsules = int(num_capsules)
        self.dim_capsules = int(dim_capsules)
        self.weight = nn.Parameter(torch.empty(self.num_capsules, 16, 8, self.dim_capsules))
        self.bias = nn.Parameter(torch.zeros(self.num_capsules, 16, 1))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.einsum("...ji,kjiz->...kjz", x, self.weight)
        c = torch.einsum("...ij,...kj->...i", u, u)[..., None]
        c = c / math.sqrt(float(self.dim_capsules))
        c = torch.softmax(c, dim=1) + self.bias
        return squash(torch.sum(u * c, dim=-2))


class EfficientCapsNet(nn.Module):
    def __init__(self, rout_capsules: int, route_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.primary_caps = PrimaryCapsLayer(128, kernel_size=9, num_capsules=16, dim_capsules=8)
        self.digit_caps = RoutingLayer(num_capsules=rout_capsules, dim_capsules=route_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = torch.relu(self.batch_norm4(self.conv4(x)))
        return self.digit_caps(self.primary_caps(x))


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, dropout: float = 0.2, alpha: float = 0.01, concat: bool = True):
        super().__init__()
        self.dropout = float(dropout)
        self.out_features = int(out_features)
        self.concat = bool(concat)
        self.weight = nn.Parameter(torch.empty(int(in_features), int(out_features)))
        self.attention = nn.Parameter(torch.empty(2 * int(out_features), 1))
        self.leakyrelu = nn.LeakyReLU(float(alpha))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        wh = torch.mm(h, self.weight)
        wh1 = torch.matmul(wh, self.attention[: self.out_features])
        wh2 = torch.matmul(wh, self.attention[self.out_features :])
        e = self.leakyrelu(wh1 + wh2.T)
        zero_vec = torch.full_like(e, -9.0e15)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        out = torch.matmul(attention, wh)
        return F.elu(out) if self.concat else out


class MultiHeadGAT(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.01,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.attentions = nn.ModuleList(
            [
                GraphAttentionLayer(
                    in_features,
                    hidden_features,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(int(heads))
            ]
        )
        self.out_att = GraphAttentionLayer(
            hidden_features * int(heads),
            out_features,
            dropout=dropout,
            alpha=alpha,
            concat=False,
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.elu(self.out_att(x, adj))


class THItoGenePatchRegressor(nn.Module):
    """Patch-H5 adaptation of THItoGene's ODConv/capsule/Transformer/GAT stack."""

    def __init__(
        self,
        *,
        n_genes: int,
        patch_size: int = 112,
        n_layers: int = 2,
        transformer_heads: int = 4,
        gat_heads: int = 2,
        dim: int = 512,
        dropout: float = 0.2,
        n_pos: int = 64,
        caps: int = 20,
        route_dim: int = 64,
        gat_hidden: int = 128,
        gat_out: int = 256,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.n_pos = int(n_pos)
        self.caps = int(caps)
        self.route_dim = int(route_dim)
        caps_out = (self.caps + 2) * self.route_dim

        self.odconv2d = ODConv2d(3, 16, kernel_size=4, stride=4)
        self.caps_layer = EfficientCapsNet(rout_capsules=self.caps, route_dim=self.route_dim)
        self.x_embed = nn.Embedding(self.n_pos, self.route_dim)
        self.y_embed = nn.Embedding(self.n_pos, self.route_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=caps_out,
            nhead=int(transformer_heads),
            dim_feedforward=int(dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.gat = MultiHeadGAT(
            caps_out,
            int(gat_hidden),
            int(gat_out),
            heads=int(gat_heads),
            dropout=float(dropout),
            alpha=0.01,
        )
        self.gene_head = nn.Sequential(
            nn.Linear(int(gat_out), int(dim)),
            nn.ReLU(),
            nn.LayerNorm(int(dim)),
            nn.Linear(int(dim), int(n_genes)),
        )

    def forward(self, patches: torch.Tensor, positions: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"patches must be shaped (batch, spots, channels, height, width), got {tuple(patches.shape)}")
        if patches.shape[0] != 1:
            raise ValueError("THItoGene patch adapter currently expects batch_size=1 slide chunks.")
        if adj.ndim == 3:
            adj = adj[0]
        batch, spots, channels, height, width = patches.shape
        x = patches.reshape(batch * spots, channels, height, width)
        if height != self.patch_size or width != self.patch_size:
            x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)
        x = torch.relu(self.odconv2d(x))
        x = self.caps_layer(x).reshape(spots, self.caps, self.route_dim)

        bins = torch.clamp(positions.long(), min=0, max=self.n_pos - 1)
        x_pos = self.x_embed(bins[0, :, 0]).unsqueeze(1)
        y_pos = self.y_embed(bins[0, :, 1]).unsqueeze(1)
        x = torch.cat((x, x_pos, y_pos), dim=1).reshape(1, spots, -1)
        x = self.transformer(x).reshape(spots, -1)
        x = self.gat(x, adj.to(dtype=x.dtype, device=x.device))
        return self.gene_head(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, expression_mask: torch.Tensor) -> torch.Tensor:
    if target.ndim == 3:
        target = target[0]
    if expression_mask.ndim == 2:
        expression_mask = expression_mask[0]
    valid = expression_mask.bool().unsqueeze(0).expand_as(target)
    if not torch.any(valid):
        raise ValueError("No valid gene values for masked MSE.")
    return (pred - target).pow(2)[valid].mean()
