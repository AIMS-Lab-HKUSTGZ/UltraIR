# Pretraining Demo

This demo contains 256 prepared rows from the molecular-dynamics pretraining
source. The spectra use the 400--4000 cm^-1 grid at 2 cm^-1 spacing and are
prepared with the shared molecular processor.

The three arrays read by the pretraining loader are:

```text
ir_norm.npy                 # [256, 1801]
fingerprint.npy             # [256, 2048]
functional_groups.npy       # [256, 17]
```

The directory contains these three loader inputs, matching the logical
pretraining data contract used by the full collection.

Run the example training configuration with:

```bash
python -m scripts.pretrain \
  --config configs/pretraining/default.yaml \
  --data-root data/pretraining/demo \
  --epochs 1 \
  --num-workers 0
```

This keeps `loader.batch_size: 128` from `configs/pretraining/default.yaml`.
The pretraining loader resizes the 1801-point spectra to the configured model
size of 1792 before batching.
