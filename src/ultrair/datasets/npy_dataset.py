"""NPY datasets and DataLoader construction for all downstream tasks."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ultrair.utils.string_tokenizer import (
    CharTokenizer,
    build_formula_tokenizer,
    build_smiles_tokenizer,
)


def _load_array(path: Path, mmap: bool) -> np.ndarray:
    array = np.load(path, allow_pickle=True)
    if mmap and array.dtype != object:
        try:
            return np.load(path, allow_pickle=True, mmap_mode="r")
        except OSError:
            pass
    return array


def _is_numeric(value: Any) -> bool:
    return (
        isinstance(value, np.ndarray) and value.dtype != object
    ) or isinstance(value, (np.number, numbers.Number))


def _as_output(value: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if _is_numeric(value):
        return torch.from_numpy(np.asarray(value, dtype=np.float32))
    return str(value)


class NpyIRDataset(Dataset):
    def __init__(
        self,
        ir_path: str | Path,
        label_path: str | Path,
        *,
        mmap: bool = True,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        label_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        return_index: bool = False,
        extra_paths: Optional[dict[str, str | Path]] = None,
    ) -> None:
        self.x = _load_array(Path(ir_path), mmap)
        self.y = _load_array(Path(label_path), mmap)
        self.extras = {
            key: _load_array(Path(path), mmap) for key, path in (extra_paths or {}).items()
        }
        if self.x.ndim == 3 and self.x.shape[1] == 1:
            self.x = self.x[:, 0]
        if self.x.ndim not in {2, 3}:
            raise ValueError(f"Expected IR shape [N,L] or [N,C,L], got {self.x.shape}")
        if self.y.ndim not in {1, 2}:
            raise ValueError(f"Expected label shape [N] or [N,K], got {self.y.shape}")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"Sample count differs: ir={len(self.x)}, labels={len(self.y)}")
        for key, array in self.extras.items():
            if len(array) != len(self.x):
                raise ValueError(f"Sample count differs for extra {key}: {len(array)} != {len(self.x)}")

        self.signal_size = int(self.x.shape[-1])
        self.num_classes = -1
        if self.y.dtype != object:
            if self.y.ndim == 2:
                self.num_classes = int(self.y.shape[1])
            elif self.y.size:
                self.num_classes = int(np.max(self.y)) + 1
        self.transform = transform
        self.label_transform = label_transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        signal = np.asarray(self.x[index], dtype=np.float32).copy()
        label = self.y[index]
        if _is_numeric(label):
            label = np.asarray(label, dtype=np.float32).copy()
            if self.label_transform is not None:
                label = self.label_transform(label)
        else:
            label = str(label)
        extras = {
            key: (
                np.asarray(array[index], dtype=np.float32).copy()
                if _is_numeric(array[index]) else str(array[index])
            )
            for key, array in self.extras.items()
        }
        sample = {
            "signal": signal[None] if signal.ndim == 1 else signal,
            "label": label,
            **extras,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        signal_out = torch.as_tensor(sample["signal"], dtype=torch.float32)
        if signal_out.ndim == 2 and signal_out.shape[0] == 1:
            signal_out = signal_out.squeeze(0)
        label_out = _as_output(sample["label"])
        extras_out = {key: _as_output(sample[key]) for key in extras}
        values: list[Any] = [signal_out, label_out]
        if self.return_index:
            values.append(index)
        if extras_out:
            values.append(extras_out)
        return tuple(values)


class NpyIRPredictionDataset(Dataset):
    """Unlabeled spectra loaded from an arbitrary NPY file."""

    def __init__(
        self,
        ir_path: str | Path,
        *,
        pair_input: bool = False,
        mmap: bool = True,
        transform: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        extra_inputs: Optional[dict[str, Any]] = None,
    ) -> None:
        values = _load_array(Path(ir_path), mmap)
        if pair_input:
            if values.ndim == 2:
                if values.shape[0] != 2:
                    raise ValueError(
                        "A single pair must have shape [2,L]; "
                        f"got {values.shape}"
                    )
                values = values[None]
            if values.ndim != 3 or values.shape[1] != 2:
                raise ValueError(
                    "Pair-task input must have shape [2,L] or [N,2,L]; "
                    f"got {values.shape}"
                )
        else:
            if values.ndim == 1:
                values = values[None]
            if values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 2:
                raise ValueError(
                    "Spectrum input must have shape [L], [N,L], or [N,1,L]; "
                    f"got {values.shape}"
                )
        if len(values) == 0:
            raise ValueError("Prediction input contains no spectra")

        self.x = values
        self.signal_size = int(values.shape[-1])
        self.transform = transform
        self.extras: dict[str, np.ndarray] = {}
        for key, raw in (extra_inputs or {}).items():
            array = np.asarray(raw)
            if array.ndim == 0:
                array = np.repeat(array.reshape(1), len(values), axis=0)
            elif len(array) == 1 and len(values) != 1:
                array = np.repeat(array, len(values), axis=0)
            if len(array) != len(values):
                raise ValueError(
                    f"Sample count differs for extra {key}: "
                    f"{len(array)} != {len(values)}"
                )
            self.extras[key] = array

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        signal = np.asarray(self.x[index], dtype=np.float32).copy()
        extras = {
            key: (
                np.asarray(array[index], dtype=np.float32).copy()
                if _is_numeric(array[index])
                else str(array[index])
            )
            for key, array in self.extras.items()
        }
        sample = {
            "signal": signal[None] if signal.ndim == 1 else signal,
            "label": None,
            **extras,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        signal_out = torch.as_tensor(sample["signal"], dtype=torch.float32)
        if signal_out.ndim == 2 and signal_out.shape[0] == 1:
            signal_out = signal_out.squeeze(0)
        extras_out = {key: _as_output(sample[key]) for key in extras}
        return (signal_out, index, extras_out) if extras_out else (signal_out, index)


@dataclass(frozen=True)
class DatasetInfo:
    signal_size: int
    num_classes: int


class GroupedIndexSampler(Sampler[int]):
    """Shuffle complete reference groups while retaining adjacency within each group."""

    def __init__(self, group_ids: np.ndarray, batch_size: int, shuffle: bool, seed: int = 42) -> None:
        ids = np.asarray(group_ids).reshape(-1)
        boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        self.groups = [
            np.arange(start, end, dtype=np.int64)
            for start, end in zip(
                np.concatenate(([0], boundaries)), np.concatenate((boundaries, [len(ids)]))
            )
        ]
        if not self.groups:
            raise ValueError("group_ids is empty")
        group_size = len(self.groups[0])
        if any(len(group) != group_size for group in self.groups):
            raise ValueError("All groups must have the same size")
        if batch_size < group_size or batch_size % group_size:
            raise ValueError("batch_size must be a multiple of the reference group size")
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self.size = len(ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        return iter([int(index) for group in order for index in self.groups[int(group)]])

    def __len__(self) -> int:
        return self.size


class TransformChain:
    def __init__(self, *transforms: Any) -> None:
        self.transforms = [transform for transform in transforms if transform is not None]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class SpectralPreprocessor:
    def __init__(self, steps: list[str], eps: float = 1e-6) -> None:
        self.steps, self.eps = [step.lower() for step in steps], float(eps)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = np.asarray(sample["signal"], dtype=np.float32)
        for step in self.steps:
            if step in {"", "none", "identity"}:
                continue
            if step in {"transmission_percent_to_absorbance", "percent_transmission_to_absorbance"}:
                output = -np.log10(np.clip(output, self.eps, 100.0) / 100.0)
            elif step in {"per_spectrum_minmax", "minmax"}:
                low = output.min(axis=-1, keepdims=True)
                output = (output - low) / np.maximum(output.max(axis=-1, keepdims=True) - low, self.eps)
            elif step in {"divide_100", "percent_to_fraction"}:
                output = output / 100.0
            else:
                raise ValueError(f"Unknown spectral preprocessing step: {step}")
        return {**sample, "signal": output.astype(np.float32, copy=False)}


class SpectralStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.maximum(np.asarray(std, dtype=np.float32), float(eps))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal = np.asarray(sample["signal"], dtype=np.float32)
        if signal.shape[-1] != self.mean.shape[-1]:
            raise ValueError(
                "Spectral standardization requires the training spectral grid "
                f"with {self.mean.shape[-1]} points; got {signal.shape[-1]}"
            )
        return {
            **sample,
            "signal": ((signal - self.mean) / self.std).astype(np.float32, copy=False),
        }


def _build_spectral_standardizer(
    cfg: dict[str, Any], train_ir_path: Path, preprocessor: Optional[SpectralPreprocessor]
) -> Optional[SpectralStandardizer]:
    standardize_cfg = cfg["data"].get("spectral_standardize", {}) or {}
    if not standardize_cfg.get("enable", False):
        return None
    spectra = _load_array(train_ir_path, mmap=True)
    if spectra.ndim == 3 and spectra.shape[1] == 1:
        spectra = spectra[:, 0]
    if spectra.ndim != 2:
        raise ValueError(
            "Spectral standardization expects [N,L] or [N,1,L], "
            f"got {spectra.shape}"
        )
    if preprocessor is not None and not standardize_cfg.get(
        "legacy_before_preprocess", False
    ):
        spectra = preprocessor({"signal": spectra})["signal"]
    return SpectralStandardizer(
        spectra.mean(axis=0),
        spectra.std(axis=0),
        eps=float(standardize_cfg.get("eps", 1e-6)),
    )


def _collate_builder(task: Any, cfg: dict[str, Any]):
    tokenize_keys = list(getattr(task, "tokenize_extra_keys", None) or [])
    tokenize_label = bool((cfg.get("tokenizer", {}) or {}).get("tokenize_label", False))
    if not tokenize_keys and not tokenize_label:
        return None
    token_cfg = cfg.get("tokenizer", {}) or {}
    default_tokenizer = build_formula_tokenizer()
    label_tokenizer = task.build_label_tokenizer() if hasattr(task, "build_label_tokenizer") else build_smiles_tokenizer()

    def collate(batch: list[tuple[Any, ...]]):
        width = len(batch[0])
        signals = torch.stack([row[0] for row in batch]).float()
        labels = [row[1] for row in batch]
        label_out: Any
        if torch.is_tensor(labels[0]):
            label_out = torch.stack(labels)
        elif tokenize_label:
            options = token_cfg.get("label", {}) or {}
            label_out = label_tokenizer.batch_encode_dict(
                labels,
                max_len=int(options.get("max_len", 128)),
                add_bos=bool(options.get("add_bos", True)),
                add_eos=bool(options.get("add_eos", True)),
            )
        else:
            label_out = labels
        has_index = width in {3, 4} and not isinstance(batch[0][2], dict)
        has_extras = isinstance(batch[0][-1], dict)
        extras_out = None
        if has_extras:
            extras_out = {}
            for key in batch[0][-1]:
                values = [row[-1][key] for row in batch]
                if key in tokenize_keys:
                    builder = getattr(task, "build_extra_tokenizer", None)
                    tokenizer: CharTokenizer = builder(key) if callable(builder) else default_tokenizer
                    options = (token_cfg.get("extras", {}) or {}).get(key, {}) or {}
                    extras_out[key] = tokenizer.batch_encode_dict(
                        values,
                        max_len=int(options.get("max_len", 128)),
                        add_bos=bool(options.get("add_bos", False)),
                        add_eos=bool(options.get("add_eos", False)),
                    )
                elif torch.is_tensor(values[0]):
                    extras_out[key] = torch.stack(values)
                else:
                    extras_out[key] = values
        result: list[Any] = [signals, label_out]
        if has_index:
            result.append(torch.as_tensor([row[2] for row in batch], dtype=torch.long))
        if has_extras:
            result.append(extras_out)
        return tuple(result)

    return collate


def _prediction_collate_builder(task: Any, cfg: dict[str, Any]):
    tokenize_keys = list(getattr(task, "tokenize_extra_keys", None) or [])
    token_cfg = cfg.get("tokenizer", {}) or {}
    default_tokenizer = build_formula_tokenizer()

    def collate(batch: list[tuple[Any, ...]]):
        signals = torch.stack([row[0] for row in batch]).float()
        indices = torch.as_tensor([row[1] for row in batch], dtype=torch.long)
        if len(batch[0]) == 2:
            return signals, indices

        extras_out: dict[str, Any] = {}
        for key in batch[0][2]:
            values = [row[2][key] for row in batch]
            if key in tokenize_keys:
                builder = getattr(task, "build_extra_tokenizer", None)
                tokenizer: CharTokenizer = (
                    builder(key) if callable(builder) else default_tokenizer
                )
                options = (token_cfg.get("extras", {}) or {}).get(key, {}) or {}
                extras_out[key] = tokenizer.batch_encode_dict(
                    values,
                    max_len=int(options.get("max_len", 128)),
                    add_bos=bool(options.get("add_bos", False)),
                    add_eos=bool(options.get("add_eos", False)),
                )
            elif torch.is_tensor(values[0]):
                extras_out[key] = torch.stack(values)
            else:
                extras_out[key] = values
        return signals, indices, extras_out

    return collate


def _split_dir(cfg: dict[str, Any], fold: int | str, split: str) -> Path:
    data = cfg["data"]
    if str(data.get("layout", "fold_directories")).lower() == "flat":
        return Path(data["root"]).expanduser()
    return Path(data["root"]).expanduser() / f"fold-{fold}" / data["splits"][split]


def _split_file(
    cfg: dict[str, Any], fold: int | str, split: str, filename: str
) -> Path:
    data = cfg["data"]
    directory = _split_dir(cfg, fold, split)
    if str(data.get("layout", "fold_directories")).lower() != "flat":
        return directory / filename
    pattern = str(
        data.get(
            "file_pattern",
            "{dataset}_fold-{fold}_{split}_{filename}",
        )
    )
    return directory / pattern.format(
        dataset=data.get("dataset", ""),
        fold=fold,
        split=data["splits"][split],
        filename=filename,
    )


def build_loaders_for_task(cfg: dict[str, Any], task: Any, fold: int | str):
    directories = {split: _split_dir(cfg, fold, split) for split in ("train", "valid", "test")}
    ir_paths = {
        split: _split_file(cfg, fold, split, cfg["data"].get("ir_name", "ir.npy"))
        for split in directories
    }
    label_paths = {
        split: _split_file(cfg, fold, split, task.label_filename)
        for split in directories
    }
    configure = getattr(task, "configure_data_context", None)
    if callable(configure):
        configure(
            cfg=cfg,
            fold=fold,
            **{f"{key}_dir": str(value) for key, value in directories.items()},
            **{f"{key}_ir_path": str(value) for key, value in ir_paths.items()},
            **{f"{key}_label_path": str(value) for key, value in label_paths.items()},
        )
    extras = getattr(task, "extra_filenames", {})
    extras = extras() if callable(extras) else extras
    extras = extras or {}
    data_cfg, aug_cfg = cfg["data"], cfg.get("augment", {}) or {}
    train_base = aug_cfg.get("train_transform")
    eval_base = aug_cfg.get("eval_transform")
    pre_cfg = data_cfg.get("spectral_preprocess", {}) or {}
    preprocessor = None
    if pre_cfg.get("enable", False):
        steps = pre_cfg.get("steps", [pre_cfg.get("mode", "identity")])
        preprocessor = SpectralPreprocessor([steps] if isinstance(steps, str) else list(steps))
    standardizer = _build_spectral_standardizer(cfg, ir_paths["train"], preprocessor)
    standardize_cfg = data_cfg.get("spectral_standardize", {}) or {}
    if standardize_cfg.get("legacy_before_preprocess", False):
        train_transform = TransformChain(standardizer, preprocessor, train_base)
        eval_transform = TransformChain(standardizer, preprocessor, eval_base)
    else:
        train_transform = TransformChain(preprocessor, standardizer, train_base)
        eval_transform = TransformChain(preprocessor, standardizer, eval_base)
    wrap = getattr(task, "wrap_transforms", None)
    if callable(wrap):
        train_transform, eval_transform = wrap(cfg, fold, train_transform, eval_transform)

    def build_dataset(split: str) -> NpyIRDataset:
        return NpyIRDataset(
            ir_paths[split],
            label_paths[split],
            transform=train_transform if split == "train" else eval_transform,
            label_transform=getattr(task, "normalize", None),
            return_index=split == "test",
            extra_paths={
                key: _split_file(cfg, fold, split, name)
                for key, name in extras.items()
            },
        )

    datasets = {split: build_dataset(split) for split in directories}
    loader_cfg = cfg.get("loader", {}) or {}
    batch_size = int(loader_cfg.get("batch_size", 128))
    num_workers = int(loader_cfg.get("num_workers", 8))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    collate_fn = _collate_builder(task, cfg)
    sampler_builder = getattr(task, "build_sampler", None)

    def build_loader(split: str) -> DataLoader:
        dataset = datasets[split]
        shuffle = split == "train"
        sampler = sampler_builder(dataset, split, batch_size, shuffle) if callable(sampler_builder) else None
        kwargs: dict[str, Any] = {}
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", False))
            kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=bool(loader_cfg.get("drop_last", False)) if split == "train" else False,
            collate_fn=collate_fn,
            **kwargs,
        )

    sample_signal = datasets["train"][0][0]
    outputs = getattr(task, "num_outputs", None)
    info = DatasetInfo(
        signal_size=int(sample_signal.shape[-1]),
        num_classes=int(outputs if outputs is not None else datasets["train"].num_classes),
    )
    return build_loader("train"), build_loader("valid"), build_loader("test"), info


def build_prediction_loader(
    cfg: dict[str, Any],
    task: Any,
    input_path: str | Path,
    *,
    stats_fold: int | str,
    extra_inputs: Optional[dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DatasetInfo]:
    """Build an unlabeled loader while reusing training-time preprocessing."""

    train_dir = _split_dir(cfg, stats_fold, "train")
    train_ir_path = _split_file(
        cfg, stats_fold, "train", cfg["data"].get("ir_name", "ir.npy")
    )
    train_label_path = _split_file(
        cfg, stats_fold, "train", task.label_filename
    )
    configure = getattr(task, "configure_data_context", None)
    if callable(configure):
        configure(
            cfg=cfg,
            fold=stats_fold,
            train_dir=str(train_dir),
            train_ir_path=str(train_ir_path),
            train_label_path=str(train_label_path),
        )

    data_cfg, aug_cfg = cfg["data"], cfg.get("augment", {}) or {}
    pre_cfg = data_cfg.get("spectral_preprocess", {}) or {}
    preprocessor = None
    if pre_cfg.get("enable", False):
        steps = pre_cfg.get("steps", [pre_cfg.get("mode", "identity")])
        preprocessor = SpectralPreprocessor(
            [steps] if isinstance(steps, str) else list(steps)
        )
    standardizer = _build_spectral_standardizer(cfg, train_ir_path, preprocessor)
    standardize_cfg = data_cfg.get("spectral_standardize", {}) or {}
    eval_base = aug_cfg.get("eval_transform")
    if standardize_cfg.get("legacy_before_preprocess", False):
        eval_transform = TransformChain(standardizer, preprocessor, eval_base)
    else:
        eval_transform = TransformChain(preprocessor, standardizer, eval_base)
    wrap = getattr(task, "wrap_transforms", None)
    if callable(wrap):
        _, eval_transform = wrap(cfg, stats_fold, None, eval_transform)

    pair_input = task.name in {
        "targeted_component_detection",
        "targeted_fractional_contribution_estimation",
    }
    dataset = NpyIRPredictionDataset(
        input_path,
        pair_input=pair_input,
        transform=eval_transform,
        extra_inputs=extra_inputs,
    )
    loader_cfg = cfg.get("loader", {}) or {}
    resolved_batch_size = int(
        loader_cfg.get("batch_size", 128) if batch_size is None else batch_size
    )
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        collate_fn=_prediction_collate_builder(task, cfg),
    )
    outputs = getattr(task, "num_outputs", None)
    if outputs is None and task.name == "molecular_structure_elucidation":
        outputs = 1  # The structural adapter does not use DatasetInfo.num_classes.
    if outputs is None:
        raise ValueError(
            f"Task {task.name!r} does not define its prediction output width"
        )
    sample_signal = dataset[0][0]
    info = DatasetInfo(
        signal_size=int(sample_signal.shape[-1]),
        num_classes=int(outputs),
    )
    return loader, info
