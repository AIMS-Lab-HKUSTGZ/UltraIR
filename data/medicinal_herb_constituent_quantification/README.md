# Medicinal-Herb Constituent Quantification Data

## Included data

This directory contains the complete five-fold data for the
Jinyinhua (JYH) and Shanyinhua (SYH) LC constituent-quantification datasets:

```text
fold-<1..5>/<train|valid|test>/
  jyh_lc_ir.npy      # float32 [N_jyh, 1868]
  jyh_lc_labels.npy  # float32 [N_jyh, 6]
  syh_lc_ir.npy      # float32 [N_syh, 1868]
  syh_lc_labels.npy  # float32 [N_syh, 4]
```

Every fold contains all 60 JYH samples as 36/12/12 train/validation/test rows
and all 75 SYH samples as 45/15/15 rows. Spectra and labels are paired row by
row, with regression targets ordered as specified in the corresponding YAML
configuration.

## Runtime processing

Both constituent-quantification YAML configs use the same spectral processing
order. For each fold, UltraIR:

1. converts each spectrum from percent transmission to absorbance;
2. min-max normalizes each spectrum independently;
3. computes a point-wise mean and standard deviation from the resulting
   training spectra only;
4. standardizes the train, validation, and test spectra with those training
   statistics (`eps: 1e-6`); and
5. resizes the signal from 1868 to the model input length of 1792.

The regression targets remain in their original units in the NumPy files.
With `stats_mode: per_fold_train` and `target_normalization: standard`, UltraIR
standardizes every target using its training-fold mean and standard deviation.
The spectral steps above run in the data loader.

Run one packaged fold:

```bash
python -m scripts.run \
  --output-dir runs/medicinal-herb-constituent-quantification \
  --config configs/medicinal_herb_constituent_quantification/jyh_lc.yaml \
  --fold 1

python -m scripts.run \
  --output-dir runs/medicinal-herb-constituent-quantification \
  --config configs/medicinal_herb_constituent_quantification/syh_lc.yaml \
  --fold 1
```

Run all five folds by replacing `--fold 1` with `--kfold`.

## Original data

The original Jinyinhua and Shanyinhua data are available from
[the UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR).
