# Targeted Component Detection Data

## Sources

Pure molecular spectra can come from [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/),
[SDBS](https://sdbs.db.aist.go.jp/), or the [simulated USPTO IR
dataset](https://zenodo.org/records/16417648). `fold-demo` contains a small
packaged example.

## Prepared arrays

Each source must have its own prepared root. Every split contains:

```text
mixture_set.npy       float [N, 2, L]
mixture_labels.npy    integer [N]       1 if the target is present, otherwise 0
mixture_ref_idx.npy   integer [N]       reference-molecule group index
```

For row `i`, `mixture_set[i, 0]` is the pure reference spectrum and
`mixture_set[i, 1]` is the generated mixture or negative mixture. All three
arrays must have the same first dimension. Rows with the same
`mixture_ref_idx` stay together in a split so the grouped sampler sees complete
reference groups.

## Source conversion and common normalization

First prepare a pure molecular library with the molecular pipeline described in
the functional-group or structure-elucidation README. In particular, the input
to the pair generator must be an aligned `ir_norm.npy` and `smiles.npy` from
`data.common.molecular`; this means each pure spectrum is finite and in `[0, 1]`:

```bash
python -m data.common.molecular \
  --ir /path/to/converted/source/ir.npy \
  --smiles /path/to/converted/source/smiles.npy \
  --output-dir /path/to/prepared/molecular/source
```

Then generate both detection and fractional-contribution targets in one pass:

```bash
python -m data.common.pairs \
  --input /path/to/prepared/molecular/source/ir_norm.npy \
  --smiles /path/to/prepared/molecular/source/smiles.npy \
  --output-dir /path/to/prepared/targeted_component_detection/source \
  --augmentations 4 --k 5 --valid-fraction 0.1 --seed 42
```

The shared generator writes `mixture_labels.npy`, `mixture_ref_weight.npy`, and
`mixture_ref_idx.npy` together, so the output can also be copied or referenced
by the fractional-contribution task.

The pair generator consumes the normalized pure library directly, creates
convex mixtures, adds small noise and baseline variation, and clips to
`[0, 1]`. Leave `--normalize-mixtures` unset for the documented task contract.
For non-molecular pure spectra, `--normalize-components` applies the required
component normalization.

The default `--augmentations 4` creates one positive and one negative pair for
each augmentation and reference molecule. Supply `--smiles` to write the five
scaffold folds; omit it to write flat arrays only.

## Run and cross-validation

Run the packaged example data:

```bash
python -m scripts.run \
  --config configs/targeted_component_detection/nist.yaml \
  --fold demo
```

Run a prepared NIST pair root:

```bash
python -m scripts.run \
  --config configs/targeted_component_detection/nist.yaml \
  --data-root /path/to/prepared/targeted_component_detection/nist \
  --fold 1
```

Run all five scaffold folds:

```bash
python -m scripts.run \
  --config configs/targeted_component_detection/nist.yaml \
  --data-root /path/to/prepared/targeted_component_detection/nist \
  --kfold
```

Use `sdbs.yaml` or `uspto.yaml` with the corresponding pair root. The task uses
`mixture_labels.npy`, thresholds predictions at 0.5, and uses grouped training
sampling based on `mixture_ref_idx.npy`. Runtime augmentation is configured in
the task YAML and is separate from offline pair generation.
