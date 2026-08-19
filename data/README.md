# UltraIR Data

Task directories provide a `fold-demo` with the documented training and
evaluation layout. The two medicinal-herb task directories contain complete
packaged train/validation/test data. The preparation utilities below create
`fold-1` through `fold-5` for cross-validation workflows.

## Task guides

| Task | Data guide |
| --- | --- |
| Functional-group prediction | [`functional_group_prediction/README.md`](functional_group_prediction/README.md) |
| Molecular structure elucidation | [`molecular_structure_elucidation/README.md`](molecular_structure_elucidation/README.md) |
| Physicochemical property prediction | [`physicochemical_property_prediction/README.md`](physicochemical_property_prediction/README.md) |
| Targeted component detection | [`targeted_component_detection/README.md`](targeted_component_detection/README.md) |
| Targeted fractional contribution estimation | [`targeted_fractional_contribution_estimation/README.md`](targeted_fractional_contribution_estimation/README.md) |
| Mixture-level component quantification | [`mixture_level_component_quantification/README.md`](mixture_level_component_quantification/README.md) |
| Bacterial classification | [`bacterial_classification/README.md`](bacterial_classification/README.md) |
| Medicinal-herb geographic origin traceability | [`medicinal_herb_geographic_origin_traceability/README.md`](medicinal_herb_geographic_origin_traceability/README.md) |
| Medicinal-herb constituent quantification | [`medicinal_herb_constituent_quantification/README.md`](medicinal_herb_constituent_quantification/README.md) |
| Microplastics classification | [`microplastics_classification/README.md`](microplastics_classification/README.md) |
| Soil property prediction | [`soil_property_prediction/README.md`](soil_property_prediction/README.md) |
| Pretraining | [`pretraining/README.md`](pretraining/README.md) |

## Data sources

| Dataset | Use in UltraIR | Source |
| --- | --- | --- |
| NIST Chemistry WebBook IR spectra | Molecular-level interpretation and mixture-analysis benchmarks | [NIST Chemistry WebBook](https://webbook.nist.gov/chemistry/) |
| SDBS IR spectra | Molecular-level interpretation and mixture-analysis benchmarks | [SDBS](https://sdbs.db.aist.go.jp/) |
| Simulated USPTO IR spectra | Molecular-level interpretation and mixture-analysis benchmarks | [Zenodo record 16417648](https://zenodo.org/records/16417648) |
| UltraIR pretraining simulated IR spectra | UltraIR encoder pretraining | [UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR), [IRtoMol on Zenodo](https://zenodo.org/records/7928396), [multimodal spectroscopy on Zenodo](https://zenodo.org/records/14770232), [QM9S on Figshare](https://figshare.com/articles/dataset/QM9S_dataset/24235333) |
| Experimental FTIR mixtures | Mixture-level component quantification | [Zenodo record 5498197](https://doi.org/10.5281/zenodo.5498197) |
| Green-snow bacterial FTIR spectra | Bacterial classification | [Zenodo record 4297950](https://zenodo.org/records/4297950) |
| Jinyinhua and Shanyinhua FTIR datasets | Medicinal-herb characterization | [UltraIR dataset on Hugging Face](https://huggingface.co/datasets/yusentan/UltraIR) |
| Environmentally sourced microplastics IR spectra | Microplastics classification | [Google Drive](https://drive.google.com/drive/folders/11MofhjEchgZelWPcHUvIMRPNEQPaLfUO?usp=sharing) |
| OSSL | Soil-property prediction | [Open Soil Spectral Library](https://docs.soilspectroscopy.org/db-access.html) |

## Required prepared layout

Convert external data to NumPy arrays with the filenames and semantics in the
corresponding task guide. Five-fold preparation produces this layout:

```text
<prepared-root>/
  fold-1/{train,valid,test}/...
  fold-2/{train,valid,test}/...
  fold-3/{train,valid,test}/...
  fold-4/{train,valid,test}/...
  fold-5/{train,valid,test}/...
```

All arrays in one split must have the same first dimension and row order. Keep
different source datasets in separate prepared roots when they use the same file
names. Select the matching YAML and pass the prepared root explicitly:

```bash
python -m scripts.run \
  --config configs/<task>/<dataset>.yaml \
  --data-root /path/to/prepared/<task>/<dataset> \
  --fold 1
```

Use the prepared root with the matching YAML. Run the packaged example with
`--fold demo`, one prepared partition with `--fold 1`, or all five prepared
partitions with `--kfold`.

## Normalization policy

Preparation follows these task-specific normalization rules:

- `data.common.molecular` performs row-wise min-max normalization to `[0, 1]`
  after source conversion. It is used by functional-group prediction, molecular
  structure elucidation, physicochemical-property prediction, and as the pure
  library input for targeted-pair generation.
- `data.common.labeled` performs the same row-wise normalization for aligned
  non-molecular `ir.npy` and label arrays. Use it for bacterial, microplastics,
  and soil data after source-specific extraction.
- Targeted pairs retain every generated mixture on the normalized component
  scale without per-mixture normalization.
- FTIRMix component quantification preserves source signal amplitude and uses
  `data.mixture_level_component_quantification.prepare` instead of generic
  per-spectrum min-max normalization.

The packaged medicinal-herb tasks apply percent-transmission-to-absorbance
conversion and per-spectrum min-max normalization, followed by training-fold
point-wise spectral standardization. Constituent-quantification configs also
standardize each regression target from training-fold statistics. Both tasks
resize the 1868-point spectra to 1792 points at runtime; load their included
`fold-demo` arrays directly.

## Local processing code

The repository includes rate-limited acquisition helpers for the NIST and
PubChem endpoints. The source-independent molecular steps are shared by NIST,
SDBS, and USPTO:

```text
data/common/
  spectra.py          # validation, row-wise normalization, resampling
  labeled.py          # normalized spectra, aligned labels, non-molecular folds
  http_client.py      # sequential rate limiting, retry, Retry-After support
  nist_download.py    # requested NIST WebBook IR JCAMP-DX records only
  pubchem.py          # CAS/InChI structure enrichment through PUG REST
  molecular_labels.py # RDKit functional groups, properties, formula, Morgan FP
  molecular.py        # shared molecular validation, labels, and folds
  sdbs_image.py       # digitize SDBS PNG plots without downloading metadata
  mixture_csv.py      # aligned local mixture spectra/target CSV conversion
  scaffold_split.py   # molecular-scaffold five-fold layout
  pairs.py            # grouped positive/negative mixture pairs
  aligned_arrays.py   # filtering and seeded aligned-array subsets
  splits.py           # non-molecular train/valid/test folds
  recipe.py           # explicit YAML dispatch and output validation
```

### Source acquisition

The NIST downloader accepts explicit WebBook IDs as a text file containing one
ID per line or as a directory whose `.txt`, `.npy`, `.jdx`, or `.dx` stems are
IDs. Its default request rate is one per second:

```bash
python -m data.common.nist_download --ids /path/to/nist_ids.txt \
  --output-dir /path/to/nist/jdx
```

Add PubChem ConnectivitySMILES after downloading NIST JCAMP files:

```bash
python -m data.common.pubchem jcamp --input-dir /path/to/nist/jdx \
  --output-dir /path/to/nist/smiles_txt
```

For a locally obtained SDBS metadata CSV, preserve every input row while adding
SMILES and an explicit per-row status:

```bash
python -m data.common.pubchem csv \
  --input-csv /path/to/sdbs/ir_results_deduplicated.csv \
  --output-csv /path/to/sdbs/metadata_with_smiles.csv
```

For local SDBS metadata and PNG files, use `sdbs_image.py`. Acquisition commands
write manifests and skip valid existing outputs by default.

For a converted molecular source containing aligned `ir.npy` and `smiles.npy`:

```bash
python -m data.common.molecular \
  --ir /path/to/ir.npy --smiles /path/to/smiles.npy \
  --output-dir /path/to/prepared/molecular
```

For a non-molecular source that has already been reduced to aligned NumPy
arrays, use the narrow generic processor. Classification uses `--stratify`;
regression does not:

```bash
python -m data.common.labeled \
  --ir /path/to/aligned/ir.npy --labels /path/to/aligned/labels.npy \
  --output-dir /path/to/prepared/task --stratify --k 5 --seed 42
```

Omit `--stratify` for multi-output regression labels. The command writes the
full arrays, `manifest.json`, and `fold-1` through `fold-5` directories.

Source-format converters are kept in `data/common/` because their raw formats
are handled independently from the molecular labels:

```bash
python -m data.common.jcamp --input-dir /path/to/jcamp --output-dir /path/to/prepared
python -m data.common.parquet_ir --input-dir /path/to/parquet \
  --output-dir /path/to/prepared
python -m data.common.sdbs_image \
  --metadata-csv /path/to/sdbs/ir_results_deduplicated.csv \
  --output-dir /path/to/converted/sdbs --smiles-field SMILES --skip-invalid
python -m data.common.mixture_csv --spectra-csv /path/to/spectra.csv \
  --targets-csv /path/to/targets.csv --output-dir /path/to/ftir
```

The SDBS converter uses the `Image File` and `SDBS No` CSV columns and writes
`ir.npy`, `wavenumbers.npy`, source IDs/files, and a rejection manifest. If a
local metadata table already contains molecular structures or phases, preserve
their alignment with `--smiles-field SMILES` and `--phase-field phase`. Optional
`--detect-phase` uses `pytesseract` on the PNG header. PNG conversion requires
Pillow; PubChem enrichment is a separate step.

JCAMP conversion writes source IDs and source filenames. Pass `--smiles-dir`
when each JCAMP stem has a matching `<stem>.txt` file containing one SMILES:

```bash
python -m data.common.jcamp --input-dir /path/to/jcamp \
  --smiles-dir /path/to/smiles_txt --output-dir /path/to/nist
```

Filter or subset any aligned NumPy directory without changing row order:

```bash
python -m data.common.aligned_arrays filter --input-dir /path/to/full \
  --output-dir /path/to/valid --mask valid_mask.npy
python -m data.common.aligned_arrays subset --input-dir /path/to/full \
  --output-dir /path/to/subset --ratio 0.1 --seed 42
```

Targeted tasks use the same pair generator. Supply `--smiles` so that scaffold
folds are written; one output contains both detection labels and fractional
weights:

```bash
python -m data.common.pairs \
  --input /path/to/pure_ir.npy --smiles /path/to/smiles.npy \
  --output-dir /path/to/pairs --augmentations 4 --k 5 --seed 42
```

The specialized FTIRMix task has a separate source layout and processor:

```bash
python -m data.mixture_level_component_quantification.prepare \
  --source-root /path/to/FTIR_and_Machine_Learning \
  --output-dir /path/to/prepared/mixture_level_component_quantification
```

The `scripts.prepare_data` entry point dispatches the `molecular`, `pairs`,
`jcamp`, `parquet_ir`, `sdbs_image`, `mixture_csv`, `ftir_mix`,
`aligned_filter`, and `aligned_subset` processors from a YAML recipe. Source
acquisition and the generic `labeled` processor use their direct commands
shown above. A recipe has this structure:

```yaml
prepare:
  task: functional_group_prediction
  processor: molecular
  input: /path/to/converted/nist
  output: /path/to/prepared/nist
  options:
    ir_name: ir.npy
    smiles_name: smiles.npy
    valid_fraction: 0.1
```

```bash
python -m scripts.prepare_data --config /path/to/recipe.yaml
```

Use `--validate-only` with the same recipe to validate an existing output
directory without preparing it again.

Pretraining reads the three NPY files written by `molecular.py`; see
`pretraining/README.md`.
