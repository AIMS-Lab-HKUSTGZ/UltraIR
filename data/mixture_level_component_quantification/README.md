# Mixture-Level Component Quantification Data

## Source

Obtain the experimental FTIRMix2022 release from [Zenodo record
5498197](https://doi.org/10.5281/zenodo.5498197). `fold-demo` contains a small
packaged example.

## Expected source tree

After extraction, pass the directory named `FTIR_and_Machine_Learning` to the
specialized processor. It expects the source layout used by the release:

```text
FTIR_and_Machine_Learning/
  Experimental data/4-AN/<run>/Labels.csv
  Experimental data/4-AN/<run>/<sample>.csv
  Synthetic data/4 components, AN ADN PN EDTA/Molecules/*.csv
```

Experimental `Labels.csv` must contain `Sample`, `F AN`, `F ADN`, `F PN`, and
`F glycerol`. Synthetic component files must include Acrylonitrile,
Adiponitrile, Propionitrile, EDTA, and Water.

## Specialized preparation

Use the task-specific processor for FTIRMix. It reads the two-column
experimental files and the
two-row component files, verifies their wavenumber grids, reconstructs the
configured four-component targets, and creates five folds using the supplied
seed:

```bash
python -m data.mixture_level_component_quantification.prepare \
  --source-root /path/to/FTIR_and_Machine_Learning \
  --output-dir /path/to/prepared/mixture_level_component_quantification \
  --seed 42
```

The output includes `full_data/` arrays, `manifest.json`, and:

```text
fold-<1..5>/<train|valid|test>/
  experimental_four_component_spectra.npy   float [N_exp, L_exp]
  experimental_four_component_targets.npy   float [N_exp, 4]
  synthetic_four_component_spectra.npy      float [N_syn, L_syn]
  synthetic_four_component_targets.npy      float [N_syn, 4]
```

Experimental target columns are `an_pct`, `adn_pct`, `pn_pct`, and
`glycerol_pct`. Synthetic target columns are `an_pct`, `adn_pct`, `pn_pct`, and
`edta_pct`. The processor converts the experimental composition percentages to
the physical target scale used by the original task and generates deterministic
synthetic Sobol/random mixtures. The written target arrays are direct task
inputs and retain the component order above.

## Runtime processing

The specialized processor preserves source amplitudes while sorting and
interpolating the spectral axes. At runtime, UltraIR computes point-wise signal
statistics from the selected training fold, applies them to all three splits,
and resizes each spectrum to 1792 points. Target values are standardized from
the same training fold according to the selected YAML.

## Run and cross-validation

Run the packaged experimental or synthetic example:

```bash
python -m scripts.run \
  --config configs/mixture_level_component_quantification/experimental_four_component.yaml \
  --fold demo

python -m scripts.run \
  --config configs/mixture_level_component_quantification/synthetic_four_component.yaml \
  --fold demo
```

Run one prepared fold:

```bash
python -m scripts.run \
  --config configs/mixture_level_component_quantification/experimental_four_component.yaml \
  --data-root /path/to/prepared/mixture_level_component_quantification \
  --fold 1
```

Use `synthetic_four_component.yaml` for the synthetic arrays. Run all five folds
with `--kfold`:

```bash
python -m scripts.run \
  --config configs/mixture_level_component_quantification/synthetic_four_component.yaml \
  --data-root /path/to/prepared/mixture_level_component_quantification \
  --kfold
```
