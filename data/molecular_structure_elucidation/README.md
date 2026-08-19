# Molecular Structure Elucidation Data

## Sources

The supported sources are [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/),
[SDBS](https://sdbs.db.aist.go.jp/), and the [simulated USPTO IR
release](https://zenodo.org/records/16417648). Prepare each source independently;
`fold-demo` contains a small packaged example.

## Prepared arrays

Prepare NIST, SDBS, and USPTO in separate roots. Each fold must contain:

```text
fold-<1..5>/<train|valid|test>/
  ir_norm.npy   float/string-independent spectrum array [N, L]
  smiles.npy    string/object array [N]
  formula.npy   string/object array [N]
```

Every row must describe the same molecule in all three arrays. SMILES must be
valid RDKit strings. Formula strings use RDKit's `CalcMolFormula`
representation and retain the row order of the spectrum and SMILES arrays.

## Shared molecular preparation and normalization

All source-specific converters first produce an aligned spectrum array and
`smiles.npy`. Then run the one common molecular processor:

```bash
python -m data.common.molecular \
  --ir /path/to/converted/source/ir_norm.npy \
  --smiles /path/to/converted/source/smiles.npy \
  --output-dir /path/to/prepared/molecular_structure_elucidation/source \
  --k 5 --valid-fraction 0.1 --seed 42
```

The processor validates alignment and finiteness, applies row-wise min-max
normalization to `[0, 1]`, generates formulas and the other shared molecular
labels, removes invalid SMILES by default, and writes five scaffold-aware
70/10/20 folds. The manifest records row counts and fold settings.

### NIST conversion

Use an explicit ID list and the public NIST/PubChem helpers, then parse JCAMP-DX:

```bash
python -m data.common.nist_download \
  --ids /path/to/nist_ids.txt --output-dir /path/to/work/nist/jdx
python -m data.common.pubchem jcamp \
  --input-dir /path/to/work/nist/jdx \
  --output-dir /path/to/work/nist/smiles_txt
python -m data.common.jcamp \
  --input-dir /path/to/work/nist/jdx \
  --smiles-dir /path/to/work/nist/smiles_txt \
  --output-dir /path/to/work/nist/converted
python -m data.common.molecular \
  --ir /path/to/work/nist/converted/ir_norm.npy \
  --smiles /path/to/work/nist/converted/smiles.npy \
  --output-dir /path/to/prepared/molecular_structure_elucidation/nist
```

Only requested IDs are downloaded. JCAMP conversion uses the 400-4000 cm-1
range, a 3600-point grid, and row-wise normalization before the final common
validation/labeling step.

### SDBS conversion

Provide a local SDBS metadata CSV and its PNG files. Enrich the CSV, digitize the
PNGs, then run `molecular`:

```bash
python -m data.common.pubchem csv \
  --input-csv /path/to/sdbs/metadata.csv \
  --output-csv /path/to/work/sdbs/metadata_with_smiles.csv
python -m data.common.sdbs_image \
  --metadata-csv /path/to/work/sdbs/metadata_with_smiles.csv \
  --image-root /path/to/sdbs \
  --smiles-field SMILES --skip-invalid \
  --output-dir /path/to/work/sdbs/converted
python -m data.common.molecular \
  --ir /path/to/work/sdbs/converted/ir.npy \
  --smiles /path/to/work/sdbs/converted/smiles.npy \
  --output-dir /path/to/prepared/molecular_structure_elucidation/sdbs
```

The PNG converter records source IDs/files and rejected rows. PubChem enrichment
is performed by the preceding CSV step. The common molecular processor supplies
the final signal normalization and RDKit formula.

### USPTO Parquet conversion

For a local simulated USPTO Parquet release:

```bash
python -m data.common.parquet_ir \
  --input-dir /path/to/uspto/parquet \
  --output-dir /path/to/work/uspto/converted
python -m data.common.molecular \
  --ir /path/to/work/uspto/converted/ir_norm.npy \
  --smiles /path/to/work/uspto/converted/smiles.npy \
  --output-dir /path/to/prepared/molecular_structure_elucidation/uspto
```

`parquet_ir` requires `pyarrow`, validates ID/SMILES/frequency/spectrum columns,
and interpolates each spectrum to the common 400-4000 cm-1, 3600-point grid.

## Run and cross-validation

The three dataset configs share the same prepared layout. For example:

```bash
python -m scripts.run \
  --config configs/molecular_structure_elucidation/nist.yaml \
  --fold demo

python -m scripts.run \
  --config configs/molecular_structure_elucidation/nist.yaml \
  --data-root /path/to/prepared/molecular_structure_elucidation/nist \
  --fold 1

python -m scripts.run \
  --config configs/molecular_structure_elucidation/nist.yaml \
  --data-root /path/to/prepared/molecular_structure_elucidation/nist \
  --kfold
```

Replace `nist.yaml` and the root with `sdbs.yaml` or `uspto.yaml` for the other
sources. Use an explicit `--config` because one task has multiple dataset YAMLs.
The model resizes the normalized input to its configured signal size (1792) at
runtime; that is independent of the offline 3600-point source grid.
