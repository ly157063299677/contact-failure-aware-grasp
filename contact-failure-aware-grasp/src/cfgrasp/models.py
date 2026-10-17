"""Subset outcome prediction corresponding to manuscript Equations (6)--(9).

The manuscript fixes the interaction pattern, dimensions and number of heads.
MLP widths, GELU activations and the PointNet-style encoder below are explicit
reference choices because their exact layouts are not specified in the paper.
No pretrained model or experimentally validated weights are bundled.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .subsets import MASKS


class PointEncoder(nn.Module):
    """Shared point MLP, with max pooling for the object-level descriptor."""

    def __init__(self, point_dim: int, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, dim), nn.GELU(),
        )

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        point_features = self.mlp(points)
        return point_features, point_features.amax(dim=1)


class GeometryBiasedCrossAttention(nn.Module):
    """Five finger queries attend to shared object points, for every subset.

    The first bias layer is factored into geometry, diagnostics and role terms.
    Their sum is exactly a linear layer on their concatenation, while avoiding
    replication of the raw geometry/diagnostic inputs across all 31 masks.
    """

    def __init__(self, dim: int, heads: int, diagnostic_dim: int,
                 bias_hidden_dim: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("network.dim must be divisible by network.heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        # rho = normalized displacement (3), separation (1), alignment (1).
        self.geometry_bias = nn.Linear(5, bias_hidden_dim)
        self.diagnostic_bias = nn.Linear(diagnostic_dim, bias_hidden_dim, bias=False)
        self.role_bias = nn.Linear(1, bias_hidden_dim, bias=False)
        self.bias_output = nn.Linear(bias_hidden_dim, heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, points: Tensor, geometry: Tensor,
                diagnostics: Tensor, masks: Tensor) -> Tensor:
        batch, subsets, fingers, dim = queries.shape
        point_count = points.shape[1]
        q = self.query(queries).reshape(
            batch, subsets, fingers, self.heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        k = self.key(points).reshape(
            batch, point_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.value(points).reshape(
            batch, point_count, self.heads, self.head_dim
        ).permute(0, 2, 1, 3)
        logits = torch.einsum("bshid,bhjd->bshij", q, k) / self.head_dim ** 0.5
        shared_bias = self.geometry_bias(geometry)
        shared_bias = shared_bias + self.diagnostic_bias(diagnostics).unsqueeze(2)
        role_bias = self.role_bias(masks.unsqueeze(-1)).unsqueeze(0).unsqueeze(3)
        bias = self.bias_output(F.gelu(shared_bias.unsqueeze(1) + role_bias))
        logits = logits + bias.permute(0, 1, 4, 2, 3)
        weights = self.dropout(logits.softmax(dim=-1))
        context = torch.einsum("bshij,bhjd->bshid", weights, v)
        context = context.permute(0, 1, 3, 2, 4).reshape(batch, subsets, fingers, dim)
        return self.output(context)


class InteractionBlock(nn.Module):
    """Pre-LN cross attention, five-finger self attention, then FFN (Eq. 8)."""

    def __init__(self, dim: int, heads: int, diagnostic_dim: int,
                 bias_hidden_dim: int, ffn_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = GeometryBiasedCrossAttention(
            dim, heads, diagnostic_dim, bias_hidden_dim, dropout
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_multiplier), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_multiplier, dim), nn.Dropout(dropout),
        )

    def forward(self, fingers: Tensor, points: Tensor, geometry: Tensor,
                diagnostics: Tensor, masks: Tensor) -> Tensor:
        fingers = fingers + self.cross_attention(
            self.cross_norm(fingers), points, geometry, diagnostics, masks
        )
        batch, subsets, count, dim = fingers.shape
        normed = self.self_norm(fingers).reshape(batch * subsets, count, dim)
        # No selected-finger attention mask: all five fingers remain visible.
        attended, _ = self.self_attention(normed, normed, normed, need_weights=False)
        fingers = fingers + attended.reshape(batch, subsets, count, dim)
        return fingers + self.ffn(self.ffn_norm(fingers))


def pool_fingers(fingers: Tensor, masks: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pool selected/unselected fingers; report the empty unselected group.

    All candidate masks are nonempty. The full-hand mask has an unselected mean
    of exactly zero and an empty-group indicator of one.
    """
    selected = masks.to(dtype=fingers.dtype).unsqueeze(0).unsqueeze(-1)
    unselected = 1.0 - selected
    selected_count = selected.sum(dim=2)
    unselected_count = unselected.sum(dim=2)
    selected_mean = (fingers * selected).sum(dim=2) / selected_count.clamp_min(1.0)
    unselected_mean = (fingers * unselected).sum(dim=2) / unselected_count.clamp_min(1.0)
    empty = (unselected_count == 0).to(fingers.dtype).expand(fingers.shape[0], -1, -1)
    return selected_mean, unselected_mean, empty


class SubsetOutcomeNetwork(nn.Module):
    """Evaluate requested masks (all 31 by default) in a state-grouped batch.

    Inputs are the normalized tensors supplied by ``StateDataset``. State and
    object identifiers are deliberately unused. Set ``interaction=False`` for
    the PointNet--Finger MLP architecture: it omits role/mask conditioning and
    both interaction blocks, retaining shared encoders, pooling and mask fusion.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        config = dict(config or {})
        self.config = config
        dim = int(config.get("dim", 128))
        heads = int(config.get("heads", 4))
        diagnostic_dim = int(config.get("diagnostic_dim", 12))
        finger_dim = int(config.get("finger_dim", 35))
        context_dim = int(config.get("context_dim", 23))
        point_dim = int(config.get("point_dim", 3))
        blocks = int(config.get("blocks", 2))
        dropout = float(config.get("dropout", 0.0))
        self.interaction = bool(config.get("interaction", True))
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("Positive feature dimensions and compatible attention heads required")
        if self.interaction and blocks < 1:
            raise ValueError("At least one block is required when interaction=True")
        self.register_buffer("masks", torch.as_tensor(MASKS.copy(), dtype=torch.float32))
        if self.masks.shape != (31, 5):
            raise ValueError("MASKS must contain all 31 nonempty five-finger masks")
        self.point_encoder = PointEncoder(point_dim, dim)
        self.finger_encoder = nn.Sequential(
            # The feature contract already includes diagnostics in this vector;
            # its separate batch field is used only for the attention bias.
            nn.Linear(finger_dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )
        self.finger_identity = nn.Embedding(5, dim)
        if self.interaction:
            self.control_role = nn.Embedding(2, dim)
            self.subset_encoder = nn.Sequential(nn.Linear(5, dim), nn.GELU())
            self.condition = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU())
            self.blocks = nn.ModuleList([
                InteractionBlock(
                    dim, heads, diagnostic_dim,
                    int(config.get("bias_hidden_dim", 32)),
                    int(config.get("ffn_multiplier", 4)), dropout,
                ) for _ in range(blocks)
            ])
        # g, selected mean, unselected mean, t, full mask, empty indicator.
        self.fusion = nn.Sequential(
            nn.Linear(3 * dim + context_dim + 5 + 1, 2 * dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim), nn.GELU(),
        )
        self.success_head = nn.Linear(dim, 1)
        self.deviation_head = nn.Linear(dim, 1)

    def forward(self, batch: Mapping[str, Tensor],
                subset_indices: Sequence[int] | Tensor | None = None) -> dict[str, Tensor]:
        """Predict masks selected by zero-based indices in integer-mask order.

        Training leaves ``subset_indices=None`` to supervise all 31 outcomes.
        Singleton-based inference can pass ``[0,1,3,7,15]`` and compute only five
        conditioned interactions. The returned columns follow the requested
        order; outputs are identical to the corresponding full-set predictions.
        """
        points = batch["points"]
        finger_features = batch["finger_features"]
        diagnostics = batch["diagnostics"]
        context = batch["context"]
        if points.ndim != 3 or points.shape[1] == 0:
            raise ValueError("points must have shape [batch, positive point count, point_dim]")
        if finger_features.shape[1] != 5 or diagnostics.shape[1] != 5:
            raise ValueError("Finger features and diagnostics must contain all five fingers")
        point_features, global_object = self.point_encoder(points)
        fingers = self.finger_encoder(finger_features)
        identities = torch.arange(5, device=fingers.device)
        fingers = fingers + self.finger_identity(identities).unsqueeze(0)
        batch_size, _, dim = fingers.shape
        masks = self.masks
        if subset_indices is not None:
            indices = torch.as_tensor(subset_indices, device=self.masks.device)
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError("subset_indices must be a nonempty one-dimensional index sequence")
            if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("subset_indices must contain integers")
            if bool(((indices < 0) | (indices >= len(self.masks))).any()):
                raise ValueError("subset_indices must be in [0,30]")
            masks = masks.index_select(0, indices.long())
        masks = masks.to(dtype=fingers.dtype)
        subset_count = masks.shape[0]
        fingers = fingers.unsqueeze(1).expand(-1, subset_count, -1, -1)
        if self.interaction:
            role = self.control_role(masks.long()).unsqueeze(0).expand(batch_size, -1, -1, -1)
            subset = self.subset_encoder(masks).view(1, subset_count, 1, dim)
            subset = subset.expand(batch_size, -1, 5, -1)
            fingers = self.condition(torch.cat((fingers, role, subset), dim=-1))
            geometry = batch["geometry"]
            if geometry.shape != (batch_size, 5, points.shape[1], 5):
                raise ValueError("geometry must have shape [batch, 5, point_count, 5]")
            for block in self.blocks:
                fingers = block(fingers, point_features, geometry, diagnostics, masks)
        selected, unselected, empty = pool_fingers(fingers, masks)
        representation = torch.cat((
            global_object.unsqueeze(1).expand(-1, subset_count, -1), selected, unselected,
            context.unsqueeze(1).expand(-1, subset_count, -1),
            masks.unsqueeze(0).expand(batch_size, -1, -1), empty,
        ), dim=-1)
        representation = self.fusion(representation)
        logits = self.success_head(representation).squeeze(-1)
        deviation = F.softplus(self.deviation_head(representation).squeeze(-1))
        return {"success_logits": logits, "success": logits.sigmoid(), "deviation": deviation}


def build_model(config: Mapping[str, Any]) -> SubsetOutcomeNetwork:
    """Accept either the complete configuration or its ``network`` section."""
    return SubsetOutcomeNetwork(config.get("network", config))
