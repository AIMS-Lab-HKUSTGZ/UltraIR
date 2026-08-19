from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ultrair.utils.string_tokenizer import build_formula_tokenizer, build_smiles_tokenizer


RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def _safe_mol(smiles: str) -> Optional["Chem.Mol"]:
    if smiles is None:
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def _canonicalize_smiles(smiles: str) -> str:
    mol = _safe_mol(smiles)
    if mol is None:
        return "" if smiles is None else str(smiles).strip()
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return str(smiles).strip()


def _morgan_fp(mol: "Chem.Mol", radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _tanimoto_smiles(smi_a: str, smi_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    mol_a = _safe_mol(smi_a)
    mol_b = _safe_mol(smi_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = _morgan_fp(mol_a, radius=radius, n_bits=n_bits)
    fp_b = _morgan_fp(mol_b, radius=radius, n_bits=n_bits)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _scaffold_smiles(smi: str) -> str:
    mol = _safe_mol(smi)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return ""


class SmilesGenerationLoss(nn.Module):
    def __init__(self, pad_id: int):
        super().__init__()
        self.pad_id = int(pad_id)

    def forward(self, outputs: Any, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(outputs, dict):
            logits = outputs.get("logits", None)
        else:
            logits = outputs
        if logits is None:
            raise ValueError("Generation model outputs must contain 'logits' during training.")
        if not isinstance(targets, dict) or "input_ids" not in targets:
            raise TypeError("SMILES generation targets must be a token dict with 'input_ids'.")

        target_ids = targets["input_ids"].long()
        if target_ids.size(1) < 2:
            raise ValueError("Tokenized SMILES must include at least BOS and EOS.")

        target_out = target_ids[:, 1:]
        if logits.size(1) != target_out.size(1):
            raise ValueError(
                "SMILES logits and targets are misaligned: "
                f"logits.shape={tuple(logits.shape)}, target_out.shape={tuple(target_out.shape)}. "
                "This usually indicates inconsistent BOS/EOS handling or decoder length handling."
            )

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_out.reshape(-1),
            ignore_index=self.pad_id,
        )

        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if key.startswith("loss_") and torch.is_tensor(value):
                    loss = loss + value

        return loss


@dataclass
class MolecularStructureElucidationTask:
    ir_key: str = "ir"
    formula_key: str = "formula"
    target_key: str = "smiles"

    smiles_filename: str = "smiles.npy"
    formula_filename: str = "formula.npy"

    fp_radius: int = 2
    fp_n_bits: int = 2048
    eval_topk: Sequence[int] = (1,)
    generation_max_len: int = 128
    beam_size: int = 1
    num_return_sequences: int = 1

    tokenize_extra_keys: Optional[Sequence[str]] = ("formula",)
    prediction_is_object: bool = True
    requires_ckpt: bool = True
    forward_needs_targets: bool = True

    _formula_tokenizer: Any = field(default_factory=build_formula_tokenizer, init=False, repr=False)
    _smiles_tokenizer: Any = field(default_factory=build_smiles_tokenizer, init=False, repr=False)

    def __post_init__(self):
        ks = []
        for value in (self.eval_topk or (1,)):
            try:
                k = int(value)
            except Exception:
                continue
            if k > 0:
                ks.append(k)

        if not ks:
            ks = [1]

        self.eval_topk = tuple(sorted(set(ks)))

        self.beam_size = max(1, int(self.beam_size))
        self.num_return_sequences = max(1, int(self.num_return_sequences))

        required_k = int(max(self.eval_topk))
        self.num_return_sequences = max(self.num_return_sequences, required_k)
        self.beam_size = max(self.beam_size, self.num_return_sequences)

    @property
    def name(self) -> str:
        return "molecular_structure_elucidation"

    @property
    def label_filename(self) -> str:
        return self.smiles_filename

    @property
    def extra_filenames(self) -> Dict[str, str]:
        return {self.formula_key: self.formula_filename}

    def class_names(self) -> Optional[List[str]]:
        return None

    def build_label_tokenizer(self):
        return self._smiles_tokenizer

    def build_extra_tokenizer(self, key: str):
        if key == self.formula_key:
            return self._formula_tokenizer
        return None

    def build_criterion(self) -> nn.Module:
        return SmilesGenerationLoss(pad_id=self._smiles_tokenizer.pad_id)

    def prepare_targets(self, y: Any):
        if isinstance(y, dict):
            out = {}
            for k, v in y.items():
                if torch.is_tensor(v):
                    if k == "attention_mask":
                        out[k] = v.bool()
                    else:
                        out[k] = v.long()
                else:
                    out[k] = v
            return out
        return y

    def _decode_smiles_batch(self, values: Any) -> List[str]:
        if isinstance(values, dict):
            input_ids = values.get("input_ids", None)
            if input_ids is None:
                raise ValueError("Token dict must contain input_ids.")
            rows = input_ids.detach().cpu().tolist() if torch.is_tensor(input_ids) else np.asarray(input_ids).tolist()
            return [self._smiles_tokenizer.decode(row, skip_special_tokens=True) for row in rows]
        if isinstance(values, (list, tuple)):
            return [str(x) for x in values]
        arr = np.asarray(values)
        if arr.ndim == 0:
            return [str(arr.item())]
        return [str(x) for x in arr.tolist()]

    def forward_model(self, model: nn.Module, x_or_batch: Union[torch.Tensor, Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(x_or_batch, dict):
            raise TypeError(
                "MolecularStructuralElucidationTask.forward_model requires a batch dict "
                f"with keys '{self.ir_key}' and '{self.formula_key}'."
            )

        if self.ir_key not in x_or_batch or self.formula_key not in x_or_batch:
            raise KeyError(
                f"Expected keys '{self.ir_key}' and '{self.formula_key}' in the batch, got {list(x_or_batch.keys())}"
            )

        ir = x_or_batch[self.ir_key]
        formula = x_or_batch[self.formula_key]
        smiles = x_or_batch.get(self.target_key, None)

        if not torch.is_tensor(ir):
            ir = torch.as_tensor(ir, dtype=torch.float32)
        if ir.ndim == 2:
            ir = ir.unsqueeze(1)

        batch = {
            self.ir_key: ir,
            self.formula_key: formula,
        }
        if smiles is not None:
            batch[self.target_key] = smiles
            return model(batch)

        return model.generate(
            ir=ir,
            formula=formula,
            max_len=int(self.generation_max_len),
            beam_size=int(self.beam_size),
            num_return_sequences=int(self.num_return_sequences),
        )

    @torch.no_grad()
    def eval_from_logits_and_targets(self, logits: Any, targets: Any, sample_indices: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out = logits
        if isinstance(out, dict) and "topk_smiles" in out:
            topk_smiles = out["topk_smiles"]
        else:
            topk_smiles = out

        if not isinstance(topk_smiles, list):
            raise ValueError("Expected generated outputs as List[List[str]]")

        true_smiles = self._decode_smiles_batch(targets)
        true_smiles_canonical = [_canonicalize_smiles(s) for s in true_smiles]
        batch_size = len(true_smiles_canonical)

        ks = [int(k) for k in self.eval_topk] if self.eval_topk is not None else [1]
        ks = sorted(set([k for k in ks if k > 0]))

        def acc_at(k: int) -> float:
            hits = 0
            for i in range(batch_size):
                preds_i = topk_smiles[i] if i < len(topk_smiles) else []
                preds_i = preds_i[:k] if isinstance(preds_i, list) else []
                preds_i = [_canonicalize_smiles(p) for p in preds_i]
                hits += int(any(pred == true_smiles_canonical[i] for pred in preds_i))
            return hits / max(batch_size, 1)

        topk_acc = {str(k): float(acc_at(k)) for k in ks}

        tani_list: List[float] = []
        scaf_list: List[float] = []
        valid_list: List[float] = []

        for i in range(batch_size):
            preds_i = topk_smiles[i] if i < len(topk_smiles) else []
            pred1 = str(preds_i[0]) if isinstance(preds_i, list) and len(preds_i) > 0 else ""
            valid_list.append(1.0 if _safe_mol(pred1) is not None else 0.0)

            tani = _tanimoto_smiles(true_smiles[i], pred1, radius=int(self.fp_radius), n_bits=int(self.fp_n_bits))
            tani_list.append(tani)

            sc_true = _scaffold_smiles(true_smiles[i])
            sc_pred = _scaffold_smiles(pred1)
            scaf_list.append(1.0 if (sc_true != "" and sc_true == sc_pred) else 0.0)

        return {
            "overall": {
                "is_structural": True,
                "num_samples": int(batch_size),
                "valid_top1_rate": float(np.mean(valid_list) if valid_list else 0.0),
            },
            "topk": topk_acc,
            "similarity": {
                "tanimoto_mean": float(np.mean(tani_list) if tani_list else 0.0),
                "scaffold_mean": float(np.mean(scaf_list) if scaf_list else 0.0),
            },
        }

