# Bacterial Classification Data

## Source

The benchmark is the green-snow bacterial FTIR dataset from [Zenodo record
4297950](https://zenodo.org/records/4297950). `fold-demo` contains the packaged
example arrays for this task.

## Prepared arrays

Convert the provider files to two aligned NumPy arrays before training:

```text
ir.npy       float [N, L]   one spectrum per row
labels.npy   integer [N]    one class index per row
```

`N` and row order must match exactly. Use one class-to-index mapping across all
folds, following `configs/bacterial_classification/bacterial_classification.yaml`.
All spectra must be finite and have a common sampled length. If the source has
several files, concatenate them while keeping labels in the same order.

## Common preparation and normalization

The public, source-independent processor validates the arrays, applies row-wise
min-max normalization to `[0, 1]`, and creates five stratified train/valid/test
partitions with an approximately 70/10/20 ratio and a configurable seed:

```bash
python -m data.common.labeled \
  --ir /path/to/bacterial/ir.npy \
  --labels /path/to/bacterial/labels.npy \
  --output-dir /path/to/prepared/bacterial_classification \
  --stratify --k 5 --valid-fraction 0.1 --seed 42
```

The output contains the full normalized arrays, `manifest.json`, and:

```text
/path/to/prepared/bacterial_classification/
  fold-1/{train,valid,test}/{ir.npy,labels.npy}
  ...
  fold-5/{train,valid,test}/{ir.npy,labels.npy}
```

Use `--target-points M` only when an explicit common resampling grid is needed.
The training loader already resizes spectra to the model's configured signal
size (1792), so resampling to 1792 during preparation is optional. If the input
is already in the desired final scale, pass `--no-normalize`.

## Run and cross-validation

Run the packaged example data with `--fold demo`:

```bash
python -m scripts.run \
  --task bacterial_classification \
  --fold demo
```

Run one prepared full-data fold by overriding the root:

```bash
python -m scripts.run \
  --task bacterial_classification \
  --data-root /path/to/prepared/bacterial_classification \
  --fold 1
```

Run all five folds:

```bash
python -m scripts.run \
  --task bacterial_classification \
  --data-root /path/to/prepared/bacterial_classification \
  --kfold
```

Training augmentation is applied at runtime according to `configs/aug.yaml`.
