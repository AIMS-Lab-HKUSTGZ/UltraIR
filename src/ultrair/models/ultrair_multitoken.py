import torch
import torch.nn as nn

from ultrair.models.ultrair import LearnableDerivative, GatedInputFusion, MultiChannelIrViT


class MultiChannelIrViTMultiToken(MultiChannelIrViT):
    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode_patches(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_tokens(x)
        return encoded[:, 0, :]

    def forward_features_with_tokens(self, x: torch.Tensor):
        encoded = self._encode_tokens(x)
        return encoded[:, 0, :], encoded[:, 1:, :]


class UltraIRClassifierMultiToken(nn.Module):
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
    ):
        super().__init__()
        self.derivative_module = LearnableDerivative(length=signal_size)
        self.input_fusion = GatedInputFusion(in_ch=3, hidden=input_fusion_hidden)
        self.backbone = MultiChannelIrViTMultiToken(
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        d1, d2 = self.derivative_module(x)
        x_multi = torch.cat([x, d1, d2], dim=1)
        x_multi = self.input_fusion(x_multi)
        return self.backbone.forward_features(x_multi)

    def forward_features_with_tokens(self, x: torch.Tensor):
        d1, d2 = self.derivative_module(x)
        x_multi = torch.cat([x, d1, d2], dim=1)
        x_multi = self.input_fusion(x_multi)
        return self.backbone.forward_features_with_tokens(x_multi)

    def forward(self, x):
        d1, d2 = self.derivative_module(x)
        x_multi = torch.cat([x, d1, d2], dim=1)
        x_multi = self.input_fusion(x_multi)
        return self.backbone(x_multi)
