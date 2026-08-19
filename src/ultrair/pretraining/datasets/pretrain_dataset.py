"""NPY datasets and batching for UltraIR multitask pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except OSError:
        return np.load(path)


class ModuloSplitIndices:
    """Lazy global indices for a deterministic modulo train/valid/test split."""

    def __init__(self, total_size: int, split: str, modulo: int, valid_mod: int, test_mod: int):
        self.total_size = int(total_size)
        self.modulo = int(modulo)
        if self.modulo < 3:
            raise ValueError("data.split_modulo must be at least 3")
        if not 0 <= valid_mod < self.modulo or not 0 <= test_mod < self.modulo:
            raise ValueError("valid_mod and test_mod must be within split_modulo")
        if valid_mod == test_mod:
            raise ValueError("valid_mod and test_mod must differ")
        if split == "valid":
            self.allowed = np.asarray([valid_mod], dtype=np.int64)
        elif split == "test":
            self.allowed = np.asarray([test_mod], dtype=np.int64)
        elif split == "train":
            self.allowed = np.asarray(
                [value for value in range(self.modulo) if value not in {valid_mod, test_mod}],
                dtype=np.int64,
            )
        else:
            raise ValueError(f"Unknown split: {split}")
        self._counts = [self._count_remainder(int(remainder)) for remainder in self.allowed]
        self._length = sum(self._counts)
        self.ndim = 1

    def _count_remainder(self, remainder: int) -> int:
        if remainder >= self.total_size:
            return 0
        return 1 + (self.total_size - 1 - remainder) // self.modulo

    def __len__(self) -> int:
        return self._length

    @property
    def shape(self) -> tuple[int]:
        return (self._length,)

    def _one(self, position: int) -> int:
        if position < 0:
            position += self._length
        if not 0 <= position < self._length:
            raise IndexError(position)
        if len(self.allowed) == 1:
            return int(self.allowed[0]) + position * self.modulo
        cycle, offset = divmod(position, len(self.allowed))
        rotated = np.concatenate((self.allowed[offset:], self.allowed[:offset]))
        for allowed in rotated:
            value = cycle * self.modulo + int(allowed)
            if value < self.total_size:
                return value
        raise IndexError(position)

    def __getitem__(self, index: int | slice | Sequence[int] | np.ndarray):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._length)
            index = np.arange(start, stop, step, dtype=np.int64)
        if isinstance(index, (list, tuple, np.ndarray)):
            positions = np.asarray(index, dtype=np.int64)
            normalized = np.where(positions < 0, positions + self._length, positions)
            if np.any((normalized < 0) | (normalized >= self._length)):
                raise IndexError(index)
            if len(self.allowed) == 1:
                return self.allowed[0] + normalized * self.modulo
            cycles, offsets = np.divmod(normalized, len(self.allowed))
            values = cycles * self.modulo + self.allowed[offsets]
            if np.any(values >= self.total_size):
                values = np.asarray([self._one(int(position)) for position in normalized], dtype=np.int64)
            return values.astype(np.int64, copy=False)
        return self._one(int(index))

    def __array__(self, dtype=None):
        values = self[:]
        return values.astype(dtype, copy=False) if dtype is not None else values


def _resize_signal(signal: np.ndarray, size: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == size:
        return signal.copy()
    source = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float64)
    target = np.linspace(0.0, 1.0, num=size, dtype=np.float64)
    return np.interp(target, source, signal).astype(np.float32)


def _sample(real_index: int, signal: np.ndarray, fingerprint: np.ndarray, groups: np.ndarray):
    return {
        "index": torch.tensor(real_index, dtype=torch.long),
        "signal": torch.from_numpy(np.asarray(signal, dtype=np.float32).copy()),
        "fingerprint": torch.from_numpy(np.asarray(fingerprint, dtype=np.float32).copy()),
        "functional_groups": torch.from_numpy(np.asarray(groups, dtype=np.float32).copy()),
    }


def _samples(
    real_indices: np.ndarray,
    signals: np.ndarray,
    fingerprints: np.ndarray,
    groups: np.ndarray,
    signal_size: int,
) -> list[dict[str, torch.Tensor]]:
    return [
        _sample(
            int(real_index),
            _resize_signal(signals[index], signal_size),
            fingerprints[index],
            groups[index],
        )
        for index, real_index in enumerate(real_indices)
    ]


class NpyPretrainingDataset(Dataset):
    """Read one directory containing the three pretraining NPY arrays."""

    def __init__(self, root: Path, indices: Any, signal_size: int):
        self.root = root
        self.indices = indices
        self.ir = _load_npy(root / "ir_norm.npy")
        self.fingerprints = _load_npy(root / "fingerprint.npy")
        self.functional_groups = _load_npy(root / "functional_groups.npy")
        self._validate_arrays(self.ir, self.fingerprints, self.functional_groups)
        self.signal_size = int(signal_size)
        self.fingerprint_bits = int(self.fingerprints.shape[1])
        self.num_fgroups = int(self.functional_groups.shape[1])

    @staticmethod
    def _validate_arrays(ir: np.ndarray, fingerprints: np.ndarray, groups: np.ndarray) -> None:
        if ir.ndim != 2:
            raise ValueError(f"Expected IR array [N, L], got {ir.shape}")
        if fingerprints.ndim != 2 or fingerprints.shape[1] != 2048:
            raise ValueError(f"Expected fingerprint array [N, 2048], got {fingerprints.shape}")
        if groups.ndim != 2 or groups.shape[1] != 17:
            raise ValueError(f"Expected functional-group array [N, 17], got {groups.shape}")
        if not (ir.shape[0] == fingerprints.shape[0] == groups.shape[0]):
            raise ValueError("The pretraining arrays have different row counts")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        real_index = int(self.indices[position])
        return _sample(
            real_index,
            _resize_signal(self.ir[real_index], self.signal_size),
            self.fingerprints[real_index],
            self.functional_groups[real_index],
        )

    def __getitems__(self, positions: Sequence[int]):
        real_indices = np.asarray(self.indices[np.asarray(positions, dtype=np.int64)], dtype=np.int64)
        return _samples(
            real_indices,
            self.ir[real_indices],
            self.fingerprints[real_indices],
            self.functional_groups[real_indices],
            self.signal_size,
        )

    def fingerprint_batch(self, real_indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.fingerprints[real_indices], dtype=np.float32)


def _total_size(root: Path) -> int:
    return int(_load_npy(root / "ir_norm.npy").shape[0])


def _load_split_indices(root: Path, filename: str | None, total_size: int):
    if not filename:
        return None
    path = Path(filename).expanduser()
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        return None
    indices = _load_npy(path)
    if indices.ndim != 1:
        raise ValueError(f"Split indices must be one-dimensional: {path}")
    return indices


def _build_bucket_keys(
    dataset: NpyPretrainingDataset,
    path: Path,
    seed: int,
    num_keys: int,
    bitcount_bin_size: int,
    chunk_size: int = 8192,
) -> np.ndarray:
    num_keys = max(1, min(int(num_keys), int(dataset.fingerprint_bits)))
    bitcount_bin_size = max(1, int(bitcount_bin_size))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    memory_backed = False
    try:
        keys = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.int32, shape=(len(dataset), num_keys)
        )
    except OSError:
        keys = np.empty((len(dataset), num_keys), dtype=np.int32)
        memory_backed = True
    priority = np.random.default_rng(seed).permutation(dataset.fingerprint_bits).astype(np.int32)
    sentinel = np.iinfo(np.int32).max
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        real_indices = np.asarray(dataset.indices[start:end], dtype=np.int64)
        fingerprints = dataset.fingerprint_batch(real_indices) > 0.5
        bit_bins = np.minimum(fingerprints.sum(1) // bitcount_bin_size, 127).astype(np.int32)
        scores = np.where(fingerprints, priority[None, :], sentinel)
        selected = np.argpartition(scores, kth=num_keys - 1, axis=1)[:, :num_keys]
        selected_scores = np.take_along_axis(scores, selected, axis=1)
        order = np.argsort(selected_scores, axis=1)
        selected = np.take_along_axis(selected, order, axis=1).astype(np.int32)
        selected[selected_scores[np.arange(end - start)[:, None], order] == sentinel] = -1
        keys[start:end] = selected + bit_bins[:, None] * dataset.fingerprint_bits
    if memory_backed:
        with temporary.open("wb") as handle:
            np.save(handle, keys)
    else:
        keys.flush()
    del keys
    temporary.replace(path)
    return _load_npy(path)


class FingerprintSimilarityBatchSampler(Sampler[list[int]]):
    def __init__(self, bucket_keys: np.ndarray, batch_size: int, seed: int, drop_last: bool = True):
        self.bucket_keys = bucket_keys if bucket_keys.ndim == 2 else bucket_keys[:, None]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        size = len(self.bucket_keys)
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        column = self.epoch % self.bucket_keys.shape[1]
        order = np.argsort(self.bucket_keys[:, column], kind="stable")
        boundaries = np.flatnonzero(np.diff(self.bucket_keys[order, column])) + 1
        groups = np.split(order, boundaries)
        rng.shuffle(groups)
        carry: list[int] = []
        for group in groups:
            rng.shuffle(group)
            carry.extend(group.tolist())
            while len(carry) >= self.batch_size:
                yield carry[: self.batch_size]
                del carry[: self.batch_size]
        if carry and not self.drop_last:
            yield carry


@dataclass(frozen=True)
class DatasetInfo:
    signal_size: int
    num_fgroups: int
    fingerprint_bits: int
    train_size: int


def build_pretraining_loaders(config: dict[str, Any]):
    data_cfg, loader_cfg = config["data"], config.get("loader", {})
    root = Path(data_cfg["root"]).expanduser()
    dataset_format = str(data_cfg.get("format", "npy"))
    if dataset_format != "npy":
        raise ValueError("data.format must be 'npy'")
    total_size = _total_size(root)
    split_names = data_cfg.get("splits", {})
    loaded = {
        split: _load_split_indices(root, split_names.get(split), total_size)
        for split in ("train", "valid", "test")
    }
    if all(value is not None for value in loaded.values()):
        split_indices = loaded
    elif str(data_cfg.get("split_strategy", "modulo")) == "modulo":
        modulo = int(data_cfg.get("split_modulo", 62))
        valid_mod, test_mod = int(data_cfg.get("valid_mod", 0)), int(data_cfg.get("test_mod", 1))
        split_indices = {
            split: ModuloSplitIndices(total_size, split, modulo, valid_mod, test_mod)
            for split in ("train", "valid", "test")
        }
    else:
        missing = [split for split, value in loaded.items() if value is None]
        raise FileNotFoundError(f"Missing split index files for: {', '.join(missing)}")

    signal_size = int(data_cfg.get("signal_size", 1792))
    datasets = {
        split: NpyPretrainingDataset(root, split_indices[split], signal_size)
        for split in ("train", "valid", "test")
    }
    batch_size = int(loader_cfg.get("batch_size", 128))
    workers = int(loader_cfg.get("num_workers", 0))
    common = {
        "num_workers": workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", True)),
        "persistent_workers": workers > 0 and bool(loader_cfg.get("persistent_workers", True)),
    }

    sampler = None
    if str(loader_cfg.get("batch_mode", "random")) == "similarity":
        cache_value = loader_cfg.get("similarity_bucket_keys")
        cache_path = Path(cache_value).expanduser() if cache_value else root / ".cache" / "train_similarity_keys.npy"
        if cache_path.is_file():
            bucket_keys = _load_npy(cache_path)
        else:
            bucket_keys = _build_bucket_keys(
                datasets["train"],
                cache_path,
                seed=int(loader_cfg.get("similarity_seed", config.get("seed", 42))),
                num_keys=int(loader_cfg.get("similarity_num_keys", 3)),
                bitcount_bin_size=int(loader_cfg.get("similarity_bitcount_bin_size", 8)),
            )
        if len(bucket_keys) != len(datasets["train"]):
            raise ValueError("Similarity bucket-key rows do not match the training split")
        sampler = FingerprintSimilarityBatchSampler(
            bucket_keys,
            batch_size,
            seed=int(loader_cfg.get("similarity_seed", config.get("seed", 42))),
        )
        train_loader = DataLoader(datasets["train"], batch_sampler=sampler, **common)
    else:
        train_loader = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True, **common
        )

    eval_common = dict(common)
    valid_loader = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, **eval_common)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, **eval_common)
    info = DatasetInfo(signal_size, datasets["train"].num_fgroups, datasets["train"].fingerprint_bits, len(datasets["train"]))
    return train_loader, valid_loader, test_loader, info, sampler
