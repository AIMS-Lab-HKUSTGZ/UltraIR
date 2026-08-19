"""UltraIR encoder and task head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SavGolConv1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        coefficients = torch.tensor(
            [[-0.0952381, 0.14285714, 0.28571429, 0.33333333,
              0.28571429, 0.14285714, -0.0952381]]
        )
        self.register_buffer("weight", coefficients.view(1, 1, -1))
        self.padding = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x, (self.padding, self.padding), mode="reflect"), self.weight)


class LearnableDerivative(nn.Module):
    def __init__(self, length: int) -> None:
        super().__init__()
        self.smoother = SavGolConv1d()
        self.grad1_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.grad2_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.grad1_conv.weight.copy_(torch.tensor([[[-0.5, 0.0, 0.5]]]))
            self.grad2_conv.weight.copy_(torch.tensor([[[1.0, -2.0, 1.0]]]))
        self.norm_d1 = nn.LayerNorm([1, length])
        self.norm_d2 = nn.LayerNorm([1, length])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        smoothed = self.smoother(x)
        return self.norm_d1(self.grad1_conv(smoothed)), self.norm_d2(self.grad2_conv(smoothed))


class GatedInputFusion(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 16) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x_multi: torch.Tensor) -> torch.Tensor:
        return x_multi * self.gate(x_multi)


class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        se_hidden = max(1, out_channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, se_hidden, 1),
            nn.ReLU(),
            nn.Conv1d(se_hidden, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)) + residual)
        return out * self.se(out)


class MultiChannelIrViT(nn.Module):
    def __init__(
        self,
        num_classes: int = 17,
        in_channels: int = 3,
        patch_size: int = 8,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 2048,
        cnn_channels: int = 64,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.channel_scale = nn.Parameter(torch.tensor([1.0, 0.5, 0.5]).view(1, 3, 1))
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
        )
        self.layer1 = ConvResBlock(cnn_channels, cnn_channels * 2, stride=2)
        self.layer2 = ConvResBlock(cnn_channels * 2, cnn_channels * 4, stride=2)
        self.layer3 = ConvResBlock(cnn_channels * 4, cnn_channels * 4, stride=2)
        self.layer4 = ConvResBlock(cnn_channels * 4, cnn_channels * 4)
        output_channels = cnn_channels * 4
        self.proj_x1 = nn.Conv1d(cnn_channels * 2, output_channels, 1)
        self.cnn_fc = nn.Sequential(
            nn.Linear(output_channels * 3, output_channels * 2),
            nn.GELU(),
            nn.Linear(output_channels * 2, output_channels),
        )
        self.patch_embedding = nn.Conv1d(
            output_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size // 2,
            padding=patch_size // 2,
        )
        # ======================== Not actually enabled. ========================
        self.patch_embedding_small = nn.Conv1d(
            in_channels=output_channels,
            out_channels=embedding_dim,
            kernel_size=4,
            stride=2,
            padding=2,
        )
        self.patch_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, int(embedding_dim * 1.5)),
            nn.GELU(),
            nn.Linear(int(embedding_dim * 1.5), embedding_dim),
        )
        # ======================================================================
        self.max_position_embeddings = 16384
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_position_embeddings + 1, embedding_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Linear(mlp_dim // 2, mlp_dim // 4),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_dim // 4, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _stem_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x * self.channel_scale)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x4 = self.layer4(self.layer3(x2))
        x2 = F.interpolate(x2, size=x4.shape[2], mode="nearest")
        x1 = self.proj_x1(F.interpolate(x1, size=x4.shape[2], mode="nearest"))
        return self.cnn_fc(torch.cat([x4, x2, x1], dim=1).transpose(1, 2)).transpose(1, 2)

    def _encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        features = self._stem_features(x)
        patches = self.patch_embedding(features).transpose(1, 2)
        batch_size, num_patches, _ = patches.shape
        if num_patches + 1 > self.max_position_embeddings:
            patches = patches[:, : self.max_position_embeddings - 1]
            num_patches = self.max_position_embeddings - 1
        tokens = torch.cat([self.cls_token.expand(batch_size, -1, -1), patches], dim=1)
        tokens = self.pos_drop(tokens + self.position_embedding[:, : num_patches + 1])
        return self.transformer_encoder(tokens)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_patches(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


class UltraIRClassifier(nn.Module):
    def __init__(
        self,
        num_fgroups: int = 17,
        d_model: int = 1024,
        signal_size: int = 1792,
        patch_len: int = 16,
        n_heads: int = 16,
        num_global_layers: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        input_fusion_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.derivative_module = LearnableDerivative(signal_size)
        self.input_fusion = GatedInputFusion(in_ch=3, hidden=input_fusion_hidden)
        self.backbone = MultiChannelIrViT(
            num_classes=num_fgroups,
            in_channels=3,
            patch_size=patch_len,
            embedding_dim=d_model,
            num_heads=n_heads,
            num_layers=num_global_layers,
            mlp_dim=d_model * 4,
            cnn_channels=64,
            dropout=dropout,
            head_dropout=head_dropout,
        )

    def _fused_input(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.derivative_module(x)
        return self.input_fusion(torch.cat([x, first, second], dim=1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(self._fused_input(x))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.forward_features(x))
