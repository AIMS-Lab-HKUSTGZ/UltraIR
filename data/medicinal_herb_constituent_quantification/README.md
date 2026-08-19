# Medicinal-Herb Constituent Quantification Data

## Included data

`fold-demo` contains the complete packaged train/validation/test data for both
constituent-quantification datasets and can be passed to the runner directly.

```text
fold-demo/<train|valid|test>/
  jyh_lc_ir.npy      # float32 [N_jyh, 1868]
  jyh_lc_labels.npy  # float32 [N_jyh, 6]
  syh_lc_ir.npy      # float32 [N_syh, 1868]
  syh_lc_labels.npy  # float32 [N_syh, 4]
```

The packaged split contains 60 Jinyinhua samples (36/12/12) and 75 Shanyinhua
samples (45/15/15). IR and label row counts must match within each dataset.
Target columns follow the property order in the corresponding YAML config.

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

Run either complete packaged dataset directly:

```bash
python -m scripts.run \
  --config configs/medicinal_herb_constituent_quantification/jyh_lc.yaml \
  --fold demo

python -m scripts.run \
  --config configs/medicinal_herb_constituent_quantification/syh_lc.yaml \
  --fold demo
```

## Original data

The original Jinyinhua and Shanyinhua data are available from
[the UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR).

## Optional five-fold layout

For custom five-fold experiments, split each dataset independently while
preserving IR/label row alignment. Use this layout under the task data root:

```text
fold-<1..5>/<train|valid|test>/
  jyh_lc_ir.npy
  jyh_lc_labels.npy
  syh_lc_ir.npy
  syh_lc_labels.npy
```

Use the same filenames, dtypes, label-column order, and signal semantics as
`fold-demo`. Run one custom fold with `--fold 1`. Use `--kfold` only after all
five fold directories contain the required arrays.
