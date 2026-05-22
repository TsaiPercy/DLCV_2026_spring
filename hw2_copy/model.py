"""DETR with ResNet-50 backbone for digit detection (HW2)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2-D sine/cosine positional encoding (DETR-style)."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x, mask):
        # x: [B, C, H, W]  mask: [B, H, W] True = padding
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, D/2]
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  # [B, d_model, H, W]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor (stride-32, 2048-channel output)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        self.body = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.num_channels = 2048

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# Custom Transformer layers (add positional encoding inside each layer)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.act = nn.ReLU()

    @staticmethod
    def _with_pos(t, pos):
        return t if pos is None else t + pos

    def forward(self, src, src_key_padding_mask=None, pos=None):
        q = k = self._with_pos(src, pos)
        src2, _ = self.self_attn(
            q, k, src, key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.drop1(src2))
        src2 = self.ff2(self.drop2(self.act(self.ff1(src))))
        src = self.norm2(src + self.drop3(src2))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.drop4 = nn.Dropout(dropout)
        self.act = nn.ReLU()

    @staticmethod
    def _with_pos(t, pos):
        return t if pos is None else t + pos

    def forward(self, tgt, memory, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        q = k = self._with_pos(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = self.norm1(tgt + self.drop1(tgt2))

        tgt2, _ = self.cross_attn(
            self._with_pos(tgt, query_pos),
            self._with_pos(memory, pos),
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.drop2(tgt2))

        tgt2 = self.ff2(self.drop3(self.act(self.ff1(tgt))))
        tgt = self.norm3(tgt + self.drop4(tgt2))
        return tgt


class Transformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # src: [B, C, H, W]   mask: [B, H, W]
        # query_embed: [N, d_model]   pos_embed: [B, d_model, H, W]
        B, C, H, W = src.shape

        # Flatten spatial dims: [H*W, B, C]
        src_seq = src.flatten(2).permute(2, 0, 1)
        pos_seq = pos_embed.flatten(2).permute(2, 0, 1)
        mask_flat = mask.flatten(1)  # [B, H*W]

        # Encoder
        memory = src_seq
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=mask_flat, pos=pos_seq)

        # Decoder
        tgt = torch.zeros(query_embed.shape[0], B, self.d_model, device=src.device)
        query_pos = query_embed.unsqueeze(1).expand(-1, B, -1)  # [N, B, d_model]

        hs = tgt
        intermediates = []
        for layer in self.decoder_layers:
            hs = layer(
                hs, memory,
                memory_key_padding_mask=mask_flat,
                pos=pos_seq,
                query_pos=query_pos,
            )
            intermediates.append(hs.permute(1, 0, 2))  # [B, N, d_model]

        # [num_decoder_layers, B, N, d_model]
        return torch.stack(intermediates)


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# DETR
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """Detection Transformer with ResNet-50 backbone."""

    def __init__(
        self,
        num_classes=10,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pretrained_backbone=True,
    ):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        # Project backbone channels to d_model
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, d_model, kernel_size=1
        )
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Prediction heads
        # num_classes real classes (0-indexed) + 1 "no-object" class (last)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, num_layers=3)

        self.num_classes = num_classes
        self.num_queries = num_queries

    def forward(self, images, masks):
        # images: [B, 3, H, W]   masks: [B, H, W] True = padding
        features = self.backbone(images)  # [B, 2048, H', W']

        # Downsample mask to feature-map size
        feat_h, feat_w = features.shape[-2:]
        feat_mask = (
            F.interpolate(
                masks.float().unsqueeze(1), size=(feat_h, feat_w)
            )
            .squeeze(1)
            .bool()
        )

        src = self.input_proj(features)  # [B, d_model, H', W']
        pos = self.pos_encoding(src, feat_mask)  # [B, d_model, H', W']

        # hs: [num_decoder_layers, B, N, d_model]
        hs = self.transformer(
            src, feat_mask, self.query_embed.weight, pos
        )

        # Final layer predictions
        pred_logits = self.class_embed(hs[-1])          # [B, N, num_classes+1]
        pred_boxes = self.bbox_embed(hs[-1]).sigmoid()   # [B, N, 4]

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        # Auxiliary outputs from intermediate decoder layers (shared heads)
        if self.training:
            out["aux_outputs"] = [
                {
                    "pred_logits": self.class_embed(hs[i]),
                    "pred_boxes": self.bbox_embed(hs[i]).sigmoid(),
                }
                for i in range(hs.shape[0] - 1)
            ]

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    num_classes=10,
    num_queries=100,
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    pretrained_backbone=True,
):
    model = DETR(
        num_classes=num_classes,
        num_queries=num_queries,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pretrained_backbone=pretrained_backbone,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model
