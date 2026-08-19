# Molecular-Dynamics IR Generation

`run_many_ir_openmm.py` is the public, resumable generator for the MD part of
the UltraIR pretraining data. Supply a local CSV whose first column is
`smiles`; an optional header row is accepted.
The output CSV contains the same SMILES followed by the common 400--4000
cm^-1 grid at 2 cm^-1 spacing.

## Environment

The simulation requires RDKit, OpenMM, `openff-toolkit`, `openff-units`, and
NumPy. Install compatible versions from the OpenFF/OpenMM documentation before
starting a simulation.

## Run

For a CPU run:

```bash
python -m data.pretraining.molecular_dynamics.run_many_ir_openmm \
  --input /path/to/smiles.csv \
  --output /path/to/md_ir.csv \
  --platform CPU \
  --workers 1
```

For GPU workers, make the device assignment explicit. Each worker uses one
GPU and the parent process writes all successful rows:

```bash
python -m data.pretraining.molecular_dynamics.run_many_ir_openmm \
  --input /path/to/smiles.csv \
  --output /path/to/md_ir.csv \
  --gpus 0,1 --workers-per-gpu 1 --platform CUDA
```

Rows that fail are written to `<output>.failures.tsv`; rerunning the command
skips rows already present in the output CSV. Use `--failures` to choose a
different failure-log path. The default setup uses RDKit ETKDGv3 plus UFF
relaxation, 300 K Langevin-middle dynamics, a 2 fs step, fixed Gasteiger
charges, mean removal, Hann windowing, FFT power summation, wavenumber
weighting, and maximum normalization on the output grid. All settings are
overridable with `--help`; `--charge-method am1bcc` is available when its
additional OpenFF charge dependencies are installed.

## Convert to UltraIR arrays

Convert the generated CSV to aligned NumPy arrays, then run the
source-independent molecular processor:

```bash
python -m data.pretraining.molecular_dynamics.convert_output \
  --input /path/to/md_ir.csv \
  --output-dir /path/to/md_raw
python -m data.common.molecular \
  --ir /path/to/md_raw/ir.npy \
  --smiles /path/to/md_raw/smiles.npy \
  --output-dir /path/to/prepared/md \
  --no-folds
```

The converter writes `ir.npy`, `smiles.npy`, `wavenumbers.npy`, and
`manifest.json`. The molecular processor creates `ir_norm.npy`,
`fingerprint.npy`, and `functional_groups.npy` for pretraining, together with
the shared molecular metadata arrays.
