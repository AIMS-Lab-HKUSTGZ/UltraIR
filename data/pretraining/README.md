# UltraIR Pretraining Data

This directory documents how to prepare the molecular IR data used for UltraIR
pretraining from public source archives and generated spectra.

## Public sources

The four public simulated-data sources are:

1. The newly generated UltraIR molecular-dynamics simulated IR dataset:
   [Hugging Face: `yusentan/UltraIR`](https://huggingface.co/datasets/yusentan/UltraIR)
2. The simulated IR dataset from IRtoMol:
   [Zenodo record 7928396](https://zenodo.org/records/7928396)
3. The simulated IR dataset from the multimodal spectroscopy dataset:
   [Zenodo record 14770232](https://zenodo.org/records/14770232)
4. The simulated IR dataset from QM9S:
   [Figshare QM9S dataset](https://figshare.com/articles/dataset/QM9S_dataset/24235333)

Download the archives from those landing pages (or use the provider's official
CLI), inspect their accompanying metadata, and make one aligned pair of arrays
for each source. For the Hugging Face dataset, for example:

```bash
python -m pip install -U huggingface_hub
hf download yusentan/UltraIR --repo-type dataset \
  --local-dir /path/to/raw/UltraIR
```

For Zenodo and Figshare, use the download controls on the linked record pages.
Then make one aligned pair of arrays for each source:

```text
source/
  ir.npy       # float [N, L], one spectrum per row
  smiles.npy   # string [N], exactly the same row order
```

Convert provider-specific Excel/CSV files using their accompanying metadata.
Keep the original source identifiers and wavenumber metadata alongside the
arrays. Before concatenating sources, put every spectrum on the same ascending
wavenumber grid and retain the exact SMILES/spectrum alignment. The common
model preparation performs the final row-wise min-max normalization.

## Source-independent preparation

For each aligned source, generate the pretraining features with the shared
molecular processor. It validates finite values and row counts, filters
invalid SMILES by default, computes 2,048-bit radius-2 Morgan fingerprints,
and creates the 17 functional-group multi-hot labels:

```bash
python -m data.common.molecular \
  --ir /path/to/source/ir.npy \
  --smiles /path/to/source/smiles.npy \
  --output-dir /path/to/prepared/source \
  --no-folds
```

The processor also writes shared molecular metadata arrays. The pretraining
loader reads `ir_norm.npy`, `fingerprint.npy`, and `functional_groups.npy`;
`--no-folds` writes them at the output root without creating fold directories.
If all sources already share a physical grid but have different point counts,
pass `--target-points M`. Otherwise, resample with the source wavenumber axes
before preparation so the arrays have one common spectral grid.

To combine already prepared sources, concatenate only aligned arrays and
preserve a provenance record. A merge can be performed with:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

roots = [
    Path("/path/to/prepared/ultrair_md"),
    Path("/path/to/prepared/irtomol"),
    Path("/path/to/prepared/multimodal"),
    Path("/path/to/prepared/qm9s"),
]
names = ("ir_norm.npy", "fingerprint.npy", "functional_groups.npy")

def load_array(path):
    try:
        return np.load(path, mmap_mode="r")
    except OSError:
        return np.load(path)

sources = [
    {name: load_array(root / name) for name in names}
    for root in roots
]
for root, arrays in zip(roots, sources):
    if len({len(array) for array in arrays.values()}) != 1:
        raise ValueError(f"unaligned rows under {root}")
for name in names:
    if len({arrays[name].shape[1:] for arrays in sources}) != 1:
        raise ValueError(f"incompatible shapes for {name}")
    if len({arrays[name].dtype for arrays in sources}) != 1:
        raise ValueError(f"incompatible dtypes for {name}")

out = Path("/path/to/prepared/pretraining")
out.mkdir(parents=True, exist_ok=True)
total_rows = sum(len(arrays[names[0]]) for arrays in sources)
for name in names:
    first = sources[0][name]
    header = {
        "descr": np.lib.format.dtype_to_descr(first.dtype),
        "fortran_order": False,
        "shape": (total_rows, *first.shape[1:]),
    }
    with (out / name).open("wb") as handle:
        np.lib.format.write_array_header_2_0(handle, header)
        for arrays in sources:
            values = arrays[name]
            for start in range(0, len(values), 8192):
                block = np.ascontiguousarray(values[start : start + 8192])
                handle.write(block.tobytes(order="C"))
PY
```

The provenance record can be kept outside this loader directory. The final
pretraining directory uses the same three arrays and widths as the demo. The
loader contract is documented in
[`src/ultrair/pretraining/README.md`](../../src/ultrair/pretraining/README.md).

## Recreating generated components

The released UltraIR dataset can be used directly when its files contain
aligned spectra and structures. The following tools recreate the generated
molecular-dynamics and machine-learning components from molecular structures.

### Molecular dynamics

The cleaned public generator is in
[`molecular_dynamics/`](molecular_dynamics/). It performs conformer
generation, geometry relaxation, non-periodic fixed-charge OpenMM dynamics,
dipole-trajectory FFT processing, interpolation to 400--4000 cm^-1, and
maximum normalization. See its [README](molecular_dynamics/README.md) for
dependencies, CPU/GPU commands, resumable output, and failure logs.

```bash
python -m data.pretraining.molecular_dynamics.run_many_ir_openmm \
  --input /path/to/nonoverlapping_smiles.csv \
  --output /path/to/md_ir.csv \
  --gpus 0 --workers-per-gpu 1
```

Convert the generator output and prepare its molecular labels:

```bash
python -m data.pretraining.molecular_dynamics.convert_output \
  --input /path/to/md_ir.csv \
  --output-dir /path/to/md_raw
python -m data.common.molecular \
  --ir /path/to/md_raw/ir.npy \
  --smiles /path/to/md_raw/smiles.npy \
  --output-dir /path/to/prepared/ultrair_md \
  --no-folds
```

### Chemprop-IR machine-learning spectra

The simulator itself is maintained externally. Clone the official
[chemprop-IR repository](https://github.com/gfm-collab/chemprop-IR), install
its documented environment, and download its released checkpoints there. Run
the official `predict.py` twice on the same SMILES CSV, once with `model_1.pt`
and once with `model_2.pt`. This repository supplies a small adapter in
[`chemprop_ir/`](chemprop_ir/) that checks exact row/header alignment and takes
the equal arithmetic mean:

```bash
python -m data.pretraining.chemprop_ir.combine_predictions \
  --predictions /path/to/predictions_model_a.csv /path/to/predictions_model_b.csv \
  --output-dir /path/to/chemprop_ir_raw
python -m data.common.molecular \
  --ir /path/to/chemprop_ir_raw/ir.npy \
  --smiles /path/to/chemprop_ir_raw/smiles.npy \
  --output-dir /path/to/prepared/chemprop_ir \
  --no-folds
```

For partitioned input, keep each prediction pair matched to the same input
part, and concatenate only after all rows and grids have been checked.

## Start pretraining

The demo and the final pretraining directory use the same three inputs:
`ir_norm.npy`, `fingerprint.npy`, and `functional_groups.npy`. Select a
prepared directory explicitly:

```bash
python -m scripts.pretrain \
  --config configs/pretraining/default.yaml \
  --data-root /path/to/prepared/pretraining
```

The plain root must contain those three arrays, with identical first
dimensions and widths 2048 and 17 for the two label arrays.

## Included demo

`demo/` contains 256 prepared rows from the MD source with the same three files
as the full pretraining data. Run it with:

```bash
python -m scripts.pretrain \
  --config configs/pretraining/default.yaml \
  --data-root data/pretraining/demo \
  --epochs 1 \
  --num-workers 0
```

The command above uses the default `loader.batch_size: 128` from
`configs/pretraining/default.yaml`.
