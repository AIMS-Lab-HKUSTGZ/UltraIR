# Targeted Fractional Contribution Estimation Data

## Sources

Pure spectra may be prepared from [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/),
[SDBS](https://sdbs.db.aist.go.jp/), or the [simulated USPTO IR
dataset](https://zenodo.org/records/16417648). `fold-demo` contains a small
packaged example.

## Prepared arrays

Each source has a separate prepared root. Every split contains:

```text
mixture_set.npy         float [N, 2, L]
mixture_ref_weight.npy  float [N]       reference-component fraction
mixture_ref_idx.npy     integer [N]     reference-molecule group index
```

`mixture_set[i, 0]` is the pure reference spectrum and `mixture_set[i, 1]` is
the generated mixture or negative mixture. For positive pairs,
`mixture_ref_weight[i]` is the coefficient used for the reference component; a
negative pair has target weight zero. All arrays must have the same first
dimension, and all rows for one `mixture_ref_idx` remain in one split.

## Source conversion and common normalization

Convert NIST, SDBS, or USPTO to an aligned molecular library with
`data.common.molecular` first. That processor supplies finite row-wise min-max
normalized pure spectra in `[0, 1]` and an aligned `smiles.npy`:

```bash
python -m data.common.molecular \
  --ir /path/to/converted/source/ir.npy \
  --smiles /path/to/converted/source/smiles.npy \
  --output-dir /path/to/prepared/molecular/source
```

Use the shared pair generator to produce the regression targets and scaffold
folds:

```bash
python -m data.common.pairs \
  --input /path/to/prepared/molecular/source/ir_norm.npy \
  --smiles /path/to/prepared/molecular/source/smiles.npy \
  --output-dir /path/to/prepared/targeted_fractional_contribution_estimation/source \
  --augmentations 4 --k 5 --valid-fraction 0.1 --seed 42
```

One run writes both `mixture_ref_weight.npy` and `mixture_labels.npy`, so the
generated root can also support targeted component detection.

The generator preserves the normalized component scale, creates convex mixtures
with noise and baseline variation, and clips values to `[0, 1]`. Leave
`--normalize-mixtures` unset for the documented task contract. Use
`--normalize-components` when the input pure library is not already in
`[0, 1]`.

Supply `--smiles` to write five scaffold folds; omit it to write flat arrays
only. `--augmentations` controls the number of positive and negative examples
per reference molecule.

## Run and cross-validation

Run the packaged example data:

```bash
python -m scripts.run \
  --config configs/targeted_fractional_contribution_estimation/nist.yaml \
  --fold demo
```

Run one prepared NIST root:

```bash
python -m scripts.run \
  --config configs/targeted_fractional_contribution_estimation/nist.yaml \
  --data-root /path/to/prepared/targeted_fractional_contribution_estimation/nist \
  --fold 1
```

Run all five folds:

```bash
python -m scripts.run \
  --config configs/targeted_fractional_contribution_estimation/nist.yaml \
  --data-root /path/to/prepared/targeted_fractional_contribution_estimation/nist \
  --kfold
```

Replace `nist.yaml` with `sdbs.yaml` or `uspto.yaml` for another source. The
task reads `mixture_ref_weight.npy`, uses grouped sampling from
`mixture_ref_idx.npy`, and uses the selected YAML for runtime augmentation.
