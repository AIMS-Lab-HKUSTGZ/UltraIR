# USPTO validation record

Validated on 2026-10-02 against upstream commit `f55743a`. Configuration SHA256 digests and machine-readable validation
results are recorded in [validation.json](validation.json).

## Public input and full preparation

The first IR shard was freshly downloaded from Zenodo and its published MD5
verified. All nine source shards used for full preparation were verified
against the public release sizes and MD5 checksums. The other eight verified
shards were reused from an existing input cache.

All **177,461** rebuilt molecular rows match the existing USPTO full arrays
exactly: IDs, normalized 3,600-point spectra, SMILES, 17 functional-group labels,
2,048-bit fingerprints, 11 properties, and formulas. This comparison uses
RDKit 2025.03.6 and the retained source-grid interpolation bounds.

| Fold | Train | Validation | Test | Nonempty scaffold overlap |
| --- | ---: | ---: | ---: | ---: |
| 1 | 124,134 | 17,834 | 35,493 | 0 |
| 2 | 124,036 | 17,933 | 35,492 | 0 |
| 3 | 124,178 | 17,791 | 35,492 | 0 |
| 4 | 124,036 | 17,933 | 35,492 | 0 |
| 5 | 124,036 | 17,933 | 35,492 | 0 |

## GPU execution

GPU validation used an NVIDIA H200, 16 CPU cores and 64 GiB RAM,
with Python 3.11, CUDA 12.4 PyTorch 2.6.0 and the pinned dependencies.
All five tasks passed training, independent checkpoint-reload evaluation,
standalone unlabeled prediction, and summary generation:

| Task | Training + reload evaluation + prediction |
| --- | ---: |
| Functional groups | 16.8 s |
| Physicochemical properties | 14.1 s |
| Molecular structure | 31.9 s |
| Target detection | 25.1 s |
| Target fraction | 25.2 s |

This execution check used 256 real molecules, fold 1, one epoch, batch size 8,
and shortened structural decoding. It covers all five task execution paths.
Eight focused CPU tests pass, covering interpolation, fixed-size Parquet row alignment, null rejection, disk pair
alignment, scaffold partitioning, corrupt-cache preservation, Unicode SMILES, matching completed-result reuse.

Fold 1 mixture preparation was also checked at full scale: **1,419,688**
pairs across training, validation and test. Every reference index, presence
label and fractional-weight range was checked; 512 spectral pairs per split
were sampled for exact reference-spectrum alignment and finite [0, 1] values.
All checks passed and the temporary audit arrays were removed.

## Full-data benchmark results

The full-data benchmark uses one GPU job per task and five sequential scaffold
folds per job. The following results were recorded on 2026-10-02:

| Task | Evaluated folds | Test metrics (mean over evaluated folds) | Status |
| --- | ---: | --- | --- |
| Functional groups | 5/5 | Micro-F1 0.93835; Macro-F1 0.88807; exact match 0.57138 | Complete |
| Physicochemical properties | 4/5 | R² 0.75995; MAE 11.32230; RMSE 44.78314 | Partial |
| Target detection | 1/5 | Accuracy 0.99866; ROC-AUC 0.99994 | Partial |
| Target fraction | 1/5 | MAE 0.00948; RMSE 0.01305; R² 0.99525 | Partial |
| Molecular structure | 0/5 | Pending | Running |

Each evaluated fold includes independent checkpoint-reload evaluation and
standalone example prediction. The functional-group five-fold summary was
also generated. Per-fold metrics are in [validation.json](validation.json).

The reference functional-group campaign used for paper Table I has
Micro-F1 0.93814, Macro-F1 0.88564 and exact-match rate 0.57030.
Its 17 functional-group mean F1 scores match the published three-decimal
values (maximum absolute difference 0.00048).

## Saved artifacts

Metrics, prediction examples and logs are retained in the experiment workspace.
Inactive training checkpoints were removed after evaluation; active molecular
structure checkpoints and pretrained encoders are retained. Completed-fold
reuse requires the trained checkpoint, evaluation report and example prediction.

## Directory-layout integration check

The revised directory layout was checked using 256 real source molecules:
Parquet conversion, molecular labels, all five scaffold folds, task dataset
loading, mixture generation and cleanup, checkpoint lookup, prediction CLI
arguments and summary paths. This CPU check intercepted model training and
prediction computation. Existing pretrained encoders were reused with network
downloads disabled. The eight data and pipeline regression checks passed.

## Release execution check

The final pipeline completed actual CPU training, checkpoint-reload evaluation,
standalone prediction and summary generation for all five tasks on aligned
scaffold subsets (eight molecules per split, one epoch, fold 1). The complete
check took 240 seconds. Completed-fold reuse also passed for all five tasks.
This check used the full model architectures, pretrained encoders and smoke
decoding settings. All 24 downstream configurations loaded their packaged
data successfully; the other downstream task types passed forward and
backward computation checks. Pretraining, test evaluation and checkpoint
resume also passed on a small aligned subset.
