# Microplastics Classification Data

## Source

Download the environmentally sourced microplastics IR dataset from the [project
Google Drive](https://drive.google.com/drive/folders/11MofhjEchgZelWPcHUvIMRPNEQPaLfUO?usp=sharing).
`fold-demo` contains a small packaged example.

## Prepared arrays

After unpacking the source, make one aligned NumPy pair:

```text
ir.npy       float [N, L]   one spectrum per row
labels.npy   integer [N]    one plastic class index per row
```

Use the class-to-index mapping documented by
`configs/microplastics_classification/microplastics_classification.yaml` across
all folds. Preserve row order when combining source files. Spectra must be
finite and sampled on a common grid.

## Common preparation and normalization

`data.common.labeled` consumes the aligned `ir.npy` and `labels.npy`, validates
them, performs row-wise min-max normalization to `[0, 1]`, and writes five
stratified folds using the supplied seed:

```bash
python -m data.common.labeled \
  --ir /path/to/microplastics/ir.npy \
  --labels /path/to/microplastics/labels.npy \
  --output-dir /path/to/prepared/microplastics_classification \
  --stratify --k 5 --valid-fraction 0.1 --seed 42
```

The prepared root contains:

```text
/path/to/prepared/microplastics_classification/
  ir.npy, labels.npy, manifest.json
  fold-1/{train,valid,test}/{ir.npy,labels.npy}
  ...
  fold-5/{train,valid,test}/{ir.npy,labels.npy}
```

The default fold ratio is approximately 70% train, 10% validation, and 20%
test. Use `--target-points M` if the source must be resampled to an explicit
common grid. Otherwise the UltraIR loader resizes to the configured model size
(1792) at runtime. Use `--no-normalize` only when the source's scale is already
the intended final scale.

## Run and cross-validation

Run the packaged example data:

```bash
python -m scripts.run \
  --task microplastics_classification \
  --fold demo
```

Run one full-data fold:

```bash
python -m scripts.run \
  --task microplastics_classification \
  --data-root /path/to/prepared/microplastics_classification \
  --fold 1
```

Run all available folds:

```bash
python -m scripts.run \
  --task microplastics_classification \
  --data-root /path/to/prepared/microplastics_classification \
  --kfold
```
