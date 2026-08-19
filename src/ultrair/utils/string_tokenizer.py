"""Small deterministic character tokenizers for molecular strings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class BatchEncoding:
    input_ids: torch.LongTensor
    attention_mask: torch.BoolTensor


class CharTokenizer:
    def __init__(
        self,
        vocab: str | Sequence[str],
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        characters = list(dict.fromkeys(list(vocab)))
        specials = [pad_token, unk_token]
        specials += [token for token in (bos_token, eos_token) if token is not None]
        self.itos = specials + characters
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_token, self.unk_token = pad_token, unk_token
        self.bos_token, self.eos_token = bos_token, eos_token
        self.pad_id, self.unk_id = self.stoi[pad_token], self.stoi[unk_token]
        self.bos_id = self.stoi[bos_token] if bos_token else None
        self.eos_id = self.stoi[eos_token] if eos_token else None
        self.special_token_ids = {
            index for index in (self.pad_id, self.unk_id, self.bos_id, self.eos_id)
            if index is not None
        }

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(
        self,
        value: str,
        max_len: Optional[int] = None,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos and self.bos_id is not None:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(char, self.unk_id) for char in str(value or ""))
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids if max_len is None else ids[: int(max_len)]

    def batch_encode(
        self,
        values: Sequence[str],
        max_len: Optional[int] = None,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> BatchEncoding:
        encoded = [self.encode(v, max_len, add_bos=add_bos, add_eos=add_eos) for v in values]
        width = int(max_len) if max_len is not None else max((len(row) for row in encoded), default=0)
        input_ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for row_index, row in enumerate(encoded):
            if row:
                input_ids[row_index, : len(row)] = torch.tensor(row)
                attention_mask[row_index, : len(row)] = True
        return BatchEncoding(input_ids, attention_mask)

    def batch_encode_dict(self, values: Sequence[str], **kwargs) -> dict[str, torch.Tensor]:
        encoding = self.batch_encode(values, **kwargs)
        return {"input_ids": encoding.input_ids, "attention_mask": encoding.attention_mask}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for raw_index in ids:
            index = int(raw_index)
            if skip_special_tokens and index in self.special_token_ids:
                continue
            if 0 <= index < len(self.itos):
                tokens.append(self.itos[index])
        return "".join(tokens)


FORMULA_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=().[]#@/\\,:;%* "
SMILES_CHARSET = "#%()+-./0123456789=@ABCFGHIKLMNOPRSTVWYZ[]\\abcdeghilnorsu"


def build_formula_tokenizer() -> CharTokenizer:
    return CharTokenizer(FORMULA_CHARSET)


def build_smiles_tokenizer() -> CharTokenizer:
    return CharTokenizer(SMILES_CHARSET, bos_token="<bos>", eos_token="<eos>")
