# Chemprop-IR Prediction Adapter

The neural-network simulator is maintained in the official external
[chemprop-IR repository](https://github.com/gfm-collab/chemprop-IR). Clone or
download that repository, install its environment, and obtain the released
checkpoints there, then use the commands below to create the prediction CSVs.

Prepare one CSV with a header and a `smiles` column for the molecules to
predict:

```text
smiles
CCO
c1ccccc1
```

Run the official predictor independently for the two released checkpoints
(`model_1.pt` and `model_2.pt` in the repository's model directory):

```bash
cd /path/to/chemprop-IR
python predict.py \
  --test_path /path/to/input_smiles.csv \
  --checkpoint_path /path/to/chemprop-IR/model/model_1.pt \
  --preds_path /path/to/predictions_a.csv \
  --no_cuda
python predict.py \
  --test_path /path/to/input_smiles.csv \
  --checkpoint_path /path/to/chemprop-IR/model/model_2.pt \
  --preds_path /path/to/predictions_b.csv \
  --no_cuda
```

Use `--gpu <index>` instead of `--no_cuda` when appropriate. Both runs must
use the same input file, target grid, and checkpoint family. Chemprop-IR
prediction files contain `smiles` followed by intensity columns; optional
feature files can be supplied with `--features_path` when required by the
checkpoint configuration. Optional uncertainty columns are ignored by the
adapter.

Average the two outputs with the public helper from this repository:

```bash
python -m data.pretraining.chemprop_ir.combine_predictions \
  --predictions /path/to/predictions_a.csv /path/to/predictions_b.csv \
  --output-dir /path/to/chemprop_ir_raw
```

The helper requires exact SMILES row equality and exact intensity-header
equality. It writes `ir.npy`, `smiles.npy`, and a provenance
`manifest.json`; it does not normalize intensities. Finish the standard
UltraIR preparation and feature generation as follows:

```bash
python -m data.common.molecular \
  --ir /path/to/chemprop_ir_raw/ir.npy \
  --smiles /path/to/chemprop_ir_raw/smiles.npy \
  --output-dir /path/to/prepared/chemprop_ir \
  --no-folds
```

For partitioned input, run the external predictor and this adapter independently
for each matching pair of prediction files, then merge only arrays with the
same wavenumber grid and preserved row order.
