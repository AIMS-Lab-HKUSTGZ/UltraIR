# UltraIR Pretraining

This package contains the encoder pretraining workflow.

## Data layout

The demo and the final collection use the same three-array contract: one IR
array, one 2,048-bit fingerprint array, and one 17-label functional-group
array. The data root must contain these aligned files.

Required files:

```text
data.root/
  ir_norm.npy                 # [N, L]
  fingerprint.npy             # [N, 2048]
  functional_groups.npy       # [N, 17]
```

If all three configured split-index files exist, they are used. Otherwise the
deterministic modulo split is used. Similarity batching creates
`data.root/.cache/train_similarity_keys.npy` on first use; an existing cache can
be selected with `loader.similarity_bucket_keys`.

## Run

The Python environment needs PyTorch, NumPy, PyYAML, and `pytorch-wavelets`.
From the repository root, point the loader at a directory with the three
arrays:

```bash
python -m scripts.pretrain \
  --config configs/pretraining/default.yaml \
  --data-root /path/to/prepared/pretraining
```

Useful local overrides include `--data-root`, `--output-dir`, `--device`,
`--epochs`, `--batch-size`, `--num-workers`, and `--resume`.

The output directory contains `last.pt`, `best.pt`, `history.json`, and the
encoder-only `epoch_<N>_encoder_no_task_head.pt` used by downstream configs.
