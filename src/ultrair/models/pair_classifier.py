import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultrair.models.ultrair import UltraIRClassifier


class SpectralPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class UltraPairClassifier(nn.Module):
    """
    Official pair classifier for mixture component identification.
    """

    def __init__(
        self,
        signal_size: int = 1792,
        d_model: int = 1024,
        patch_len: int = 16,
        n_heads: int = 16,
        num_global_layers: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        input_fusion_hidden: int = 16,
        pair_hidden_dim: int = 384,
        pair_dropout: float = 0.1,
        joint_hidden_dim: int = 512,
        joint_num_heads: int = 8,
        joint_num_layers: int = 2,
        freq_hidden_dim: int = 160,
    ):
        super().__init__()
        self.feature_dim = int(d_model)
        self.pair_hidden_dim = int(pair_hidden_dim)

        self.encoder = UltraIRClassifier(
            num_fgroups=2,
            d_model=d_model,
            signal_size=signal_size,
            patch_len=patch_len,
            n_heads=n_heads,
            num_global_layers=num_global_layers,
            dropout=dropout,
            head_dropout=head_dropout,
            input_fusion_hidden=input_fusion_hidden,
        )

        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.pair_hidden_dim),
            nn.GELU(),
        )
        self.cls_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.pair_hidden_dim),
            nn.GELU(),
        )
        self.token_pool = nn.Sequential(
            nn.LayerNorm(self.pair_hidden_dim),
            nn.Linear(self.pair_hidden_dim, 1),
        )

        self.joint_conv = nn.Sequential(
            nn.Conv1d(3, joint_hidden_dim // 4, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(joint_hidden_dim // 4),
            nn.GELU(),
            nn.Conv1d(joint_hidden_dim // 4, joint_hidden_dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(joint_hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(joint_hidden_dim // 2, joint_hidden_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(joint_hidden_dim),
            nn.GELU(),
        )
        self.joint_pos = SpectralPositionalEncoding(joint_hidden_dim, dropout=pair_dropout, max_len=1024)
        joint_layer = nn.TransformerEncoderLayer(
            d_model=joint_hidden_dim,
            nhead=joint_num_heads,
            dropout=pair_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.joint_transformer = nn.TransformerEncoder(joint_layer, num_layers=joint_num_layers)
        self.joint_pool = nn.Sequential(
            nn.LayerNorm(joint_hidden_dim),
            nn.Linear(joint_hidden_dim, 1),
        )
        self.joint_proj = nn.Sequential(
            nn.LayerNorm(joint_hidden_dim),
            nn.Linear(joint_hidden_dim, self.pair_hidden_dim),
            nn.GELU(),
        )

        self.freq_branch_short = nn.Sequential(
            nn.Conv1d(8, freq_hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(freq_hidden_dim),
            nn.GELU(),
        )
        self.freq_branch_mid = nn.Sequential(
            nn.Conv1d(8, freq_hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(freq_hidden_dim),
            nn.GELU(),
        )
        self.freq_branch_long = nn.Sequential(
            nn.Conv1d(8, freq_hidden_dim, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(freq_hidden_dim),
            nn.GELU(),
        )
        self.freq_merge = nn.Sequential(
            nn.Conv1d(freq_hidden_dim * 3, freq_hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(freq_hidden_dim * 2),
            nn.GELU(),
            nn.Conv1d(freq_hidden_dim * 2, freq_hidden_dim * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(freq_hidden_dim * 2),
            nn.GELU(),
        )
        self.freq_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(freq_hidden_dim * 2),
            nn.Linear(freq_hidden_dim * 2, self.pair_hidden_dim),
            nn.GELU(),
        )

        pair_stats_dim = 10
        total_pair_repr_dim = self.pair_hidden_dim * 5 + pair_stats_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_pair_repr_dim),
            nn.Linear(total_pair_repr_dim, self.pair_hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.pair_hidden_dim * 3, self.pair_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.pair_hidden_dim * 2, 1),
        )

    def _encode_single_spectrum(self, x_single: torch.Tensor):
        base = self.encoder.backbone
        d1, d2 = self.encoder.derivative_module(x_single)
        x_multi = torch.cat([x_single, d1, d2], dim=1)
        x_multi = self.encoder.input_fusion(x_multi)
        features = base._stem_features(x_multi)
        patches = base.patch_embedding(features).transpose(1, 2)

        batch_size, num_tokens, _ = patches.shape
        if num_tokens + 1 > base.max_position_embeddings:
            patches = patches[:, : base.max_position_embeddings - 1, :]
            num_tokens = base.max_position_embeddings - 1

        cls_tokens = base.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        tokens = tokens + base.position_embedding[:, : num_tokens + 1, :]
        encoded = base.transformer_encoder(base.pos_drop(tokens))
        return encoded[:, 0, :], encoded[:, 1:, :]

    def _attentive_pool(self, tokens: torch.Tensor, pool_head: nn.Module) -> torch.Tensor:
        score = pool_head(tokens).squeeze(-1)
        attn = torch.softmax(score, dim=-1).unsqueeze(-1)
        pooled = torch.sum(tokens * attn, dim=1)
        max_pooled = torch.amax(tokens, dim=1)
        return 0.5 * (pooled + max_pooled)

    def _build_frequency_channels(self, ref: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        ref_fft = torch.fft.rfft(ref, dim=-1)
        mix_fft = torch.fft.rfft(mix, dim=-1)

        ref_mag = torch.log1p(torch.abs(ref_fft))
        mix_mag = torch.log1p(torch.abs(mix_fft))
        sum_mag = ref_mag + mix_mag
        diff_mag = torch.abs(mix_mag - ref_mag)

        cross = ref_fft * torch.conj(mix_fft)
        coherence = torch.abs(cross) / (torch.abs(ref_fft) * torch.abs(mix_fft) + eps)
        phase = torch.angle(cross) / math.pi
        power_ratio = torch.log1p((torch.abs(mix_fft) + eps) / (torch.abs(ref_fft) + eps))
        power_product = torch.log1p(torch.abs(ref_fft) * torch.abs(mix_fft))

        return torch.cat(
            [ref_mag, mix_mag, sum_mag, diff_mag, coherence, phase, power_ratio, power_product],
            dim=1,
        )

    def _frequency_features(self, ref: torch.Tensor, mix: torch.Tensor):
        freq_input = self._build_frequency_channels(ref, mix)
        freq_feat = torch.cat(
            [
                self.freq_branch_short(freq_input),
                self.freq_branch_mid(freq_input),
                self.freq_branch_long(freq_input),
            ],
            dim=1,
        )
        freq_feat = self.freq_proj(self.freq_merge(freq_feat))

        ref_mag = freq_input[:, 0, :]
        mix_mag = freq_input[:, 1, :]
        diff_mag = freq_input[:, 3, :]
        coherence = freq_input[:, 4, :]
        phase = freq_input[:, 5, :]
        pair_stats = torch.cat(
            [
                F.cosine_similarity(ref_mag, mix_mag, dim=-1).unsqueeze(-1),
                torch.mean(diff_mag, dim=-1, keepdim=True),
                torch.amax(diff_mag, dim=-1, keepdim=True),
                torch.mean(coherence, dim=-1, keepdim=True),
                torch.amax(coherence, dim=-1, keepdim=True),
                torch.mean(torch.abs(phase), dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return freq_feat, pair_stats

    def forward_pair_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected [B, 2, L], got {tuple(x.shape)}")

        ref = x[:, 0:1, :]
        mix = x[:, 1:2, :]

        cls_ref, tok_ref = self._encode_single_spectrum(ref)
        cls_mix, tok_mix = self._encode_single_spectrum(mix)

        tok_ref = self.token_proj(tok_ref)
        tok_mix = self.token_proj(tok_mix)
        cls_ref = self.cls_proj(cls_ref)
        cls_mix = self.cls_proj(cls_mix)
        pooled_ref = self._attentive_pool(tok_ref, self.token_pool)
        pooled_mix = self._attentive_pool(tok_mix, self.token_pool)

        raw_pair = torch.cat([ref, mix, torch.abs(mix - ref)], dim=1)
        joint_tokens = self.joint_conv(raw_pair).transpose(1, 2)
        joint_tokens = self.joint_pos(joint_tokens)
        joint_tokens = self.joint_transformer(joint_tokens)
        joint_feat = self.joint_proj(self._attentive_pool(joint_tokens, self.joint_pool))

        freq_feat, freq_stats = self._frequency_features(ref, mix)

        cls_diff = torch.abs(cls_ref - cls_mix)
        pooled_diff = torch.abs(pooled_ref - pooled_mix)

        pair_stats = torch.cat(
            [
                F.cosine_similarity(cls_ref, cls_mix, dim=-1).unsqueeze(-1),
                F.cosine_similarity(pooled_ref, pooled_mix, dim=-1).unsqueeze(-1),
                F.cosine_similarity(joint_feat, freq_feat, dim=-1).unsqueeze(-1),
                torch.mean(joint_tokens * joint_tokens, dim=(1, 2), keepdim=False).unsqueeze(-1),
                freq_stats,
            ],
            dim=-1,
        )

        pair_repr = torch.cat(
            [
                cls_ref,
                cls_mix,
                cls_diff,
                pooled_diff,
                freq_feat,
            ],
            dim=-1,
        )
        return torch.cat([pair_repr, pair_stats], dim=-1)

    def forward(self, x):
        ref_idx = None
        has_targets = False
        if isinstance(x, dict):
            ref_idx = x.get("ref_idx", None)
            has_targets = "targets" in x
            x = x.get("ir", None)
        if x is None:
            raise ValueError("UltraPairClassifier requires 'ir' input.")

        pair_repr = self.forward_pair_features(x)
        logits = self.classifier(pair_repr)
        if ref_idx is None or not has_targets:
            return logits
        return {
            "logits": logits,
            "pair_repr": pair_repr,
            "ref_idx": ref_idx,
        }
