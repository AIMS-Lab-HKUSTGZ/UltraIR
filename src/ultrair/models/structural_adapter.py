from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ultrair.models.ultrair_multitoken import UltraIRClassifierMultiToken
from ultrair.utils.string_tokenizer import build_formula_tokenizer, build_smiles_tokenizer


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)


class FormulaTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int = 0,
        max_len: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


class IRResampler(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        num_latents: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(d_model),
                        "ln_kv": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=d_model,
                            num_heads=n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "ffn": nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, 4 * d_model),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(4 * d_model, d_model),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, ir_tokens: torch.Tensor) -> torch.Tensor:
        x = self.latents.expand(ir_tokens.size(0), -1, -1)
        for blk in self.layers:
            q = blk["ln_q"](x)
            kv = blk["ln_kv"](ir_tokens)
            attn_out, _ = blk["cross_attn"](q, kv, kv, need_weights=False)
            x = x + attn_out
            x = x + blk["ffn"](x)
        return x


class SmilesTransformerDecoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_idx: int,
        max_len: int,
    ):
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        x = self.embedding(decoder_input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        tgt_key_padding_mask = decoder_input_ids.eq(self.pad_idx)
        tgt_mask = self._causal_mask(decoder_input_ids.size(1), decoder_input_ids.device)
        hidden = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula_counts(formula: str) -> Dict[str, int]:
    formula = "" if formula is None else str(formula).replace(" ", "")
    counts: Dict[str, int] = {}
    for elem, count in _FORMULA_RE.findall(formula):
        counts[elem] = counts.get(elem, 0) + (int(count) if count else 1)
    return counts


def _formula_exact_match(pred_formula: str, target_formula: str) -> bool:
    pred_counts = _parse_formula_counts(pred_formula)
    target_counts = _parse_formula_counts(target_formula)
    if not pred_counts or not target_counts:
        return False
    return pred_counts == target_counts


class UltraIRStructuralAdapter(nn.Module):
    def __init__(
        self,
        *,
        signal_size: int,
        d_model: int = 512,
        patch_len: int = 16,
        n_heads: int = 8,
        num_global_layers: int = 6,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
        input_fusion_hidden: int = 16,
        formula_num_layers: int = 2,
        decoder_num_layers: int = 4,
        formula_max_len: int = 64,
        smiles_max_len: int = 128,
        beam_size: int = 1,
        num_return_sequences: int = 1,
        length_penalty: float = 0.7,
        use_ir_vocab_prompt: bool = False,
        ir_prompt_mode: str = "soft",
        ir_prompt_num_tokens: int = 4,
        ir_vocab_prompt_temperature: float = 1.0,
        ir_prompt_dropout: float = 0.0,
        use_multi_token_ir_memory: bool = False,
        ir_num_latents: int = 8,
        ir_resampler_layers: int = 2,
        contrastive_dim: int = 256,
        contrastive_temperature: float = 0.1,
        lambda_ir_formula_contrastive: float = 0.0,
        lambda_ir_target_contrastive: float = 0.0,
        use_formula_rerank: bool = False,
    ):
        super().__init__()
        self.tokenizer = build_smiles_tokenizer()
        self.formula_tokenizer = build_formula_tokenizer()
        self.pad_id = int(self.tokenizer.pad_id)
        self.unk_id = int(self.tokenizer.unk_id)
        self.bos_id = int(self.tokenizer.bos_id)
        self.eos_id = int(self.tokenizer.eos_id)
        self.smiles_max_len = int(smiles_max_len)
        self.default_beam_size = int(beam_size)
        self.default_num_return_sequences = int(num_return_sequences)
        self.length_penalty = float(length_penalty)
        self.use_ir_vocab_prompt = bool(use_ir_vocab_prompt)
        self.ir_prompt_mode = str(ir_prompt_mode).lower()
        valid_prompt_modes = {"soft", "topk", "hybrid"}
        if self.use_ir_vocab_prompt and self.ir_prompt_mode not in valid_prompt_modes:
            raise ValueError(
                f"Unsupported ir_prompt_mode={ir_prompt_mode!r}. "
                f"Expected one of {sorted(valid_prompt_modes)}."
            )
        self.ir_prompt_num_tokens = max(1, int(ir_prompt_num_tokens))
        self.ir_vocab_prompt_temperature = float(ir_vocab_prompt_temperature)
        self.use_multi_token_ir_memory = bool(use_multi_token_ir_memory)
        self.ir_num_latents = max(1, int(ir_num_latents))
        self.contrastive_temperature = max(1e-4, float(contrastive_temperature))
        self.lambda_ir_formula_contrastive = float(lambda_ir_formula_contrastive)
        self.lambda_ir_target_contrastive = float(lambda_ir_target_contrastive)
        self.use_formula_rerank = bool(use_formula_rerank)
        ff_dim = d_model * 4

        self.ir_backbone = UltraIRClassifierMultiToken(
            num_fgroups=self.tokenizer.vocab_size,
            d_model=d_model,
            signal_size=signal_size,
            patch_len=patch_len,
            n_heads=n_heads,
            num_global_layers=num_global_layers,
            dropout=dropout,
            head_dropout=head_dropout,
            input_fusion_hidden=input_fusion_hidden,
        )
        self.ir_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.ir_resampler = None
        if self.use_multi_token_ir_memory:
            self.ir_resampler = IRResampler(
                d_model=d_model,
                n_heads=n_heads,
                num_latents=self.ir_num_latents,
                num_layers=max(1, int(ir_resampler_layers)),
                dropout=dropout,
            )

        self.formula_encoder = FormulaTransformerEncoder(
            vocab_size=self.formula_tokenizer.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=formula_num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            pad_idx=self.formula_tokenizer.pad_id,
            max_len=formula_max_len,
        )
        self.decoder = SmilesTransformerDecoder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=decoder_num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            pad_idx=self.pad_id,
            max_len=smiles_max_len,
        )

        if self.use_ir_vocab_prompt:
            self.ir_soft_prompt_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            self.ir_feature_prompt_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            self.ir_topk_prompt_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            self.ir_prompt_dropout = nn.Dropout(ir_prompt_dropout)
            self.ir_soft_prompt_gate = nn.Parameter(torch.tensor(1.0))
            self.ir_feature_prompt_gate = nn.Parameter(torch.tensor(1.0))
            self.ir_topk_prompt_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.ir_soft_prompt_proj = None
            self.ir_feature_prompt_proj = None
            self.ir_topk_prompt_proj = None
            self.ir_prompt_dropout = None
            self.ir_soft_prompt_gate = None
            self.ir_feature_prompt_gate = None
            self.ir_topk_prompt_gate = None

        self.ir_formula_proj_ir = None
        self.ir_formula_proj_formula = None
        if self.lambda_ir_formula_contrastive > 0.0:
            self.ir_formula_proj_ir = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, contrastive_dim),
                nn.GELU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
            self.ir_formula_proj_formula = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, contrastive_dim),
                nn.GELU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )

        self.ir_target_proj_ir = None
        self.ir_target_proj_target = None
        if self.lambda_ir_target_contrastive > 0.0:
            self.ir_target_proj_ir = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, contrastive_dim),
                nn.GELU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
            self.ir_target_proj_target = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, contrastive_dim),
                nn.GELU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=x.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def _decode_formula(self, formula: Dict[str, torch.Tensor], index: int) -> str:
        ids = formula["input_ids"][index]
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        return self.formula_tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _build_ir_prompt_tokens(self, ir_feat: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.use_ir_vocab_prompt:
            return None

        ir_logits = self.ir_backbone.backbone.classifier(ir_feat)
        temperature = max(1e-4, self.ir_vocab_prompt_temperature)
        prompt_tokens: List[torch.Tensor] = []

        if self.ir_prompt_mode in {"soft", "hybrid"}:
            probs = F.softmax(ir_logits / temperature, dim=-1)
            soft_token = probs @ self.decoder.embedding.weight
            soft_token = self.ir_soft_prompt_proj(soft_token).unsqueeze(1)
            prompt_tokens.append(self.ir_soft_prompt_gate * soft_token)

        if self.ir_prompt_mode == "topk":
            topk = min(self.ir_prompt_num_tokens, ir_logits.size(-1))
            top_logits, top_ids = torch.topk(ir_logits, k=topk, dim=-1)
            top_probs = F.softmax(top_logits / temperature, dim=-1)
            top_tokens = self.decoder.embedding(top_ids)
            top_tokens = top_tokens * top_probs.unsqueeze(-1)
            top_tokens = self.ir_topk_prompt_proj(top_tokens)
            prompt_tokens.append(self.ir_topk_prompt_gate * top_tokens)

        if self.ir_prompt_mode == "hybrid":
            feature_token = self.ir_feature_prompt_proj(ir_feat).unsqueeze(1)
            prompt_tokens.append(self.ir_feature_prompt_gate * feature_token)

        if not prompt_tokens:
            return None

        prompt = torch.cat(prompt_tokens, dim=1)
        return self.ir_prompt_dropout(prompt)

    def _contrastive_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        proj_x: nn.Module,
        proj_y: nn.Module,
    ) -> torch.Tensor:
        if x.size(0) < 2:
            return x.new_zeros(())
        x = F.normalize(proj_x(x), dim=-1)
        y = F.normalize(proj_y(y), dim=-1)
        logits = (x @ y.transpose(0, 1)) / self.contrastive_temperature
        labels = torch.arange(x.size(0), device=x.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def _beam_rank_score(self, seq: List[int], score: float) -> float:
        length = max(1, len(seq) - 1)
        return float(score) / (length ** self.length_penalty)

    def _apply_generation_token_constraints(self, step_log_probs: torch.Tensor) -> torch.Tensor:
        step_log_probs = step_log_probs.clone()
        blocked_ids = [self.pad_id, self.unk_id, self.bos_id]
        for token_id in blocked_ids:
            step_log_probs[:, token_id] = float("-inf")
        return step_log_probs

    def _rerank_beams(
        self,
        beams: List[Tuple[List[int], float, bool]],
        formula_str: str,
    ) -> List[Tuple[List[int], float, bool]]:
        if not self.use_formula_rerank:
            return sorted(beams, key=lambda item: self._beam_rank_score(item[0], item[1]), reverse=True)

        ranked = []
        for seq, score, finished in beams:
            smiles = self.tokenizer.decode(seq, skip_special_tokens=True)
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            beam_score = self._beam_rank_score(seq, score)
            finished_tier = 0 if finished else 1

            if mol is None:
                tier = 2
            else:
                formula_match = True
                if formula_str:
                    pred_formula = rdMolDescriptors.CalcMolFormula(mol)
                    formula_match = _formula_exact_match(pred_formula, formula_str)
                tier = 0 if formula_match else 1

            ranked.append(((finished_tier, tier, -beam_score), (seq, score, finished)))

        ranked.sort(key=lambda item: item[0])
        return [item[1] for item in ranked]

    def encode_memory(
        self,
        ir: torch.Tensor,
        formula: Dict[str, torch.Tensor],
        return_context: bool = False,
    ):
        if self.use_multi_token_ir_memory:
            ir_global, ir_tokens = self.ir_backbone.forward_features_with_tokens(ir)
            ir_token = self.ir_proj(ir_global).unsqueeze(1)
            ir_latents = self.ir_resampler(ir_tokens)
        else:
            ir_global = self.ir_backbone.forward_features(ir)
            ir_token = self.ir_proj(ir_global).unsqueeze(1)
            ir_latents = None

        formula_ids = formula["input_ids"]
        formula_mask = formula["attention_mask"].bool()
        formula_hidden = self.formula_encoder(formula_ids, formula_mask)
        prompt_tokens = self._build_ir_prompt_tokens(ir_global)

        pieces = [ir_token]
        masks = [torch.ones((ir_token.size(0), ir_token.size(1)), device=ir_token.device, dtype=torch.bool)]
        if ir_latents is not None:
            pieces.append(ir_latents)
            masks.append(torch.ones((ir_latents.size(0), ir_latents.size(1)), device=ir_latents.device, dtype=torch.bool))
        if prompt_tokens is not None:
            pieces.append(prompt_tokens)
            masks.append(torch.ones((prompt_tokens.size(0), prompt_tokens.size(1)), device=prompt_tokens.device, dtype=torch.bool))
        pieces.append(formula_hidden)
        masks.append(formula_mask)

        memory = torch.cat(pieces, dim=1)
        memory_valid_mask = torch.cat(masks, dim=1)
        memory_key_padding_mask = ~memory_valid_mask

        if not return_context:
            return memory, memory_key_padding_mask

        context = {
            "ir_global": ir_global,
            "formula_hidden": formula_hidden,
            "formula_mask": formula_mask,
        }
        return memory, memory_key_padding_mask, context

    def forward_train(self, ir: torch.Tensor, formula: Dict[str, torch.Tensor], smiles: Dict[str, torch.Tensor]):
        memory, memory_key_padding_mask, context = self.encode_memory(ir, formula, return_context=True)
        input_ids = smiles["input_ids"]
        if input_ids.size(1) < 2:
            raise ValueError("SMILES target length must be at least 2 when using BOS/EOS.")

        decoder_input_ids = input_ids[:, :-1]
        need_hidden = self.lambda_ir_target_contrastive > 0.0
        if need_hidden:
            logits, hidden = self.decoder(decoder_input_ids, memory, memory_key_padding_mask, return_hidden=True)
        else:
            logits = self.decoder(decoder_input_ids, memory, memory_key_padding_mask)
            hidden = None

        out: Dict[str, Any] = {"logits": logits}

        if self.lambda_ir_formula_contrastive > 0.0 and self.ir_formula_proj_ir is not None and self.ir_formula_proj_formula is not None:
            formula_summary = self._masked_mean(context["formula_hidden"], context["formula_mask"])
            out["loss_ir_formula_contrastive"] = self.lambda_ir_formula_contrastive * self._contrastive_loss(
                context["ir_global"],
                formula_summary,
                self.ir_formula_proj_ir,
                self.ir_formula_proj_formula,
            )

        if self.lambda_ir_target_contrastive > 0.0 and hidden is not None and self.ir_target_proj_ir is not None and self.ir_target_proj_target is not None:
            if "attention_mask" in smiles:
                decoder_mask = smiles["attention_mask"][:, :-1].bool()
            else:
                decoder_mask = decoder_input_ids.ne(self.pad_id)
            target_summary = self._masked_mean(hidden, decoder_mask)
            out["loss_ir_target_contrastive"] = self.lambda_ir_target_contrastive * self._contrastive_loss(
                context["ir_global"],
                target_summary,
                self.ir_target_proj_ir,
                self.ir_target_proj_target,
            )

        return out

    @torch.no_grad()
    def generate(
        self,
        ir: torch.Tensor,
        formula: Dict[str, torch.Tensor],
        *,
        max_len: Optional[int] = None,
        beam_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
    ) -> Dict[str, List[List[str]]]:
        max_len = int(max_len or self.smiles_max_len)
        beam_size = int(beam_size or self.default_beam_size)
        num_return_sequences = int(num_return_sequences or self.default_num_return_sequences)
        beam_size = max(1, beam_size)
        num_return_sequences = max(1, min(num_return_sequences, beam_size))

        memory, memory_key_padding_mask = self.encode_memory(ir, formula)
        outputs: List[List[str]] = []
        batch_size = memory.size(0)

        for i in range(batch_size):
            mem_i = memory[i : i + 1]
            pad_i = memory_key_padding_mask[i : i + 1] if memory_key_padding_mask is not None else None
            beams: List[Tuple[List[int], float, bool]] = [([self.bos_id], 0.0, False)]

            for _ in range(max_len - 1):
                finished_beams = [beam for beam in beams if beam[2]]
                active_beams = [beam for beam in beams if not beam[2]]
                if not active_beams:
                    break

                seq_lens = [len(seq) for seq, _, _ in active_beams]
                max_seq_len = max(seq_lens)
                decoder_ids = torch.full(
                    (len(active_beams), max_seq_len),
                    fill_value=self.pad_id,
                    device=mem_i.device,
                    dtype=torch.long,
                )
                for row_idx, (seq, _, _) in enumerate(active_beams):
                    decoder_ids[row_idx, : len(seq)] = torch.as_tensor(seq, device=mem_i.device, dtype=torch.long)

                mem_batch = mem_i.expand(len(active_beams), -1, -1)
                pad_batch = pad_i.expand(len(active_beams), -1) if pad_i is not None else None
                step_logits = self.decoder(decoder_ids, mem_batch, pad_batch)
                row_idx = torch.arange(len(active_beams), device=mem_i.device)
                last_pos = torch.as_tensor([length - 1 for length in seq_lens], device=mem_i.device, dtype=torch.long)
                step_logits = step_logits[row_idx, last_pos, :]
                step_log_probs = F.log_softmax(step_logits, dim=-1)
                step_log_probs = self._apply_generation_token_constraints(step_log_probs)
                top_log_probs, top_ids = torch.topk(step_log_probs, k=beam_size, dim=-1)

                candidates = list(finished_beams)
                for beam_idx, (seq, score, _) in enumerate(active_beams):
                    for logp, token_id in zip(top_log_probs[beam_idx].tolist(), top_ids[beam_idx].tolist()):
                        token_id = int(token_id)
                        new_seq = seq + [token_id]
                        new_finished = token_id == self.eos_id
                        candidates.append((new_seq, score + float(logp), new_finished))

                candidates.sort(key=lambda item: self._beam_rank_score(item[0], item[1]), reverse=True)
                beams = candidates[:beam_size]
                if all(finished for _, _, finished in beams):
                    break

            formula_str = self._decode_formula(formula, i)
            beams = self._rerank_beams(beams, formula_str)
            smiles_list = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq, _, _ in beams[:num_return_sequences]]
            outputs.append(smiles_list)

        return {"topk_smiles": outputs}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        ir = batch["ir"]
        formula = batch["formula"]
        smiles = batch.get("smiles", None)

        if smiles is not None:
            return self.forward_train(ir, formula, smiles)
        return self.generate(ir, formula)

