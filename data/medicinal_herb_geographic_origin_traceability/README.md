# Medicinal-Herb Geographic Origin Traceability Data

## Included data

`fold-demo` contains the complete packaged train/validation/test data for both
geographic-origin datasets and can be passed to the runner directly.

```text
fold-demo/<train|valid|test>/
  jyh_ir.npy      # float32 [N_jyh, 1868]
  jyh_labels.npy  # int64 [N_jyh]
  syh_ir.npy      # float32 [N_syh, 1868]
  syh_labels.npy  # int64 [N_syh]
```

The packaged split contains 120 Jinyinhua samples (72/24/24) and 150
Shanyinhua samples (90/30/30). Jinyinhua has four origin classes and Shanyinhua
has five; integer labels follow the class-name order in the corresponding YAML
config. IR and label row counts must match within each dataset.

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

Run either complete packaged dataset directly:

```bash
python -m scripts.run \
  --config configs/medicinal_herb_geographic_origin_traceability/jyh.yaml \
  --fold demo

python -m scripts.run \
  --config configs/medicinal_herb_geographic_origin_traceability/syh.yaml \
  --fold demo
```

## Original data

The original Jinyinhua and Shanyinhua data are available from
[the UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR).

## Optional five-fold layout

For custom five-fold experiments, split each dataset independently, preserve
IR/label row alignment, and keep the class distribution represented in every
split where sample counts permit. Use this layout under the task data root:

```text
fold-<1..5>/<train|valid|test>/
  jyh_ir.npy
  jyh_labels.npy
  syh_ir.npy
  syh_labels.npy
```

Use the same filenames, dtypes, class mapping, and signal semantics as
`fold-demo`. Run one custom fold with `--fold 1`. Use `--kfold` only after all
five fold directories contain the required arrays.
