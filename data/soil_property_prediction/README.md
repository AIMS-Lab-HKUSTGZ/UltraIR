# Soil Property Prediction Data

## Source

Request or download the Open Soil Spectral Library (OSSL) through the [official
database access guide](https://docs.soilspectroscopy.org/db-access.html).
`fold-demo` contains a small packaged example.

## Prepared arrays

The source tables must first be reduced to one strictly row-aligned NumPy pair:

```text
ir.npy       float [N, L]
labels.npy   float [N, 10]
```

The ten target columns and their order are fixed by
`configs/soil_property_prediction/soil_property_prediction.yaml`:

```text
total_carbon, total_nitrogen, total_sulfur, clay,
cation_exchange_capacity, extractable_calcium, extractable_magnesium,
extractable_potassium, extractable_sodium, soil_pH
```

Resolve missing values and source-specific unit choices before saving the
arrays. Every row in `ir.npy` must describe the same soil sample as the
corresponding row in `labels.npy`; both arrays must be finite and the spectra
must have a common sampled length.

## Common preparation and normalization

The shared processor applies row-wise min-max normalization to each IR spectrum
and creates five unstratified folds with an approximately 70/10/20 ratio and a
configurable seed.

```bash
python -m data.common.labeled \
  --ir /path/to/ossl/ir.npy \
  --labels /path/to/ossl/labels.npy \
  --output-dir /path/to/prepared/soil_property_prediction \
  --k 5 --valid-fraction 0.1 --seed 42
```

The output contains full arrays, `manifest.json`, and the files expected by the
training config:

```text
/path/to/prepared/soil_property_prediction/
  fold-1/{train,valid,test}/{ir.npy,labels.npy}
  ...
  fold-5/{train,valid,test}/{ir.npy,labels.npy}
```

Use `--target-points M` for explicit resampling when necessary. The model loader
otherwise resizes to 1792 points at runtime. Use `--no-normalize` only when the
source signal is already in the intended final scale.

The YAML task standardizes the ten regression targets using the training fold.

## Run and cross-validation

Run the packaged example arrays:

```bash
python -m scripts.run \
  --task soil_property_prediction \
  --fold demo
```

Run one prepared fold:

```bash
python -m scripts.run \
  --task soil_property_prediction \
  --data-root /path/to/prepared/soil_property_prediction \
  --fold 1
```

Run all five folds:

```bash
python -m scripts.run \
  --task soil_property_prediction \
  --data-root /path/to/prepared/soil_property_prediction \
  --kfold
```
