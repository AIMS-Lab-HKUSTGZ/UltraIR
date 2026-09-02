# Medicinal-Herb Geographic Origin Traceability Data

## Included data

This directory contains the complete stratified five-fold data for
the Jinyinhua (JYH) and Shanyinhua (SYH) geographic-origin datasets:

```text
fold-<1..5>/<train|valid|test>/
  jyh_ir.npy      # float32 [N_jyh, 1868]
  jyh_labels.npy  # int64 [N_jyh]
  syh_ir.npy      # float32 [N_syh, 1868]
  syh_labels.npy  # int64 [N_syh]
```

Every fold contains all 120 JYH samples as 72/24/24 train/validation/test rows
and all 150 SYH samples as 90/30/30 rows. JYH has four origin classes and SYH
has five; spectra and labels are paired row by row, with integer labels following
the class-name order in the corresponding YAML configuration.

## Runtime processing

Both geographic-origin YAML configs use the same spectral pipeline. For each
fold, UltraIR:

1. converts each spectrum from percent transmission to absorbance;
2. min-max normalizes each spectrum independently;
3. computes a point-wise mean and standard deviation from the resulting
   training spectra only;
4. standardizes the train, validation, and test spectra with those training
   statistics (`eps: 1e-6`); and
5. resizes the signal from 1868 to the model input length of 1792.

The JYH and SYH configs select the best validation-accuracy checkpoint for test
evaluation. The packaged spectra retain their original signal representation;
the steps above run in the data loader.

Run one packaged fold:

```bash
python -m scripts.run \
  --config configs/medicinal_herb_geographic_origin_traceability/jyh.yaml \
  --fold 1

python -m scripts.run \
  --config configs/medicinal_herb_geographic_origin_traceability/syh.yaml \
  --fold 1
```

Run all five folds by replacing `--fold 1` with `--kfold`.

## Original data

The original Jinyinhua and Shanyinhua data are available from
[the UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR).
