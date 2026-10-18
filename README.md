# EvidenceConserve

Implementation of **Evidence-conserving active view acquisition with risk-calibrated stopping**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

Each object episode supplies candidate visibility descriptors, quality, motion cost, reachability and recorded numerical evidence. The selected view reveals its evidence. Calibration and evaluation use independent generated objects.

The bundled generator writes synthetic finite-model scenarios as JSONL. Fix the seed and keep the generated input for paired comparisons.

```bash
python -m evidenceconserve prepare --count 200 --seed 17 --output data/generated.jsonl
```

Prepare independent calibration objects and fit the stopping correction before running the method. `configs/development_objects.json` is empty for the fixed numerical generator; replace it with all fitting/tuning object IDs for imported data.

```bash
python -m evidenceconserve prepare --count 2000 --seed 37 --output data/calibration.jsonl
python -m evidenceconserve calibrate --data data/calibration.jsonl --development-objects configs/development_objects.json --grid-size 21 --delta 0.05 --minimum-coverage 0.1 --output checkpoints/calibration.json
```

## Execution and evaluation

```bash
python -m evidenceconserve run --data data/generated.jsonl --calibration checkpoints/calibration.json --output runs/main
```

## Comparisons and ablations

```bash
python -m evidenceconserve suite --data data/generated.jsonl --calibration checkpoints/calibration.json --output runs/comparison
```

Use `--methods` to select: `evidenceconserve`, `naive_sum`, `no_transport`, `uniform_views`, `no_calibration`.

## Outputs

`runs/main/evidenceconserve/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m evidenceconserve plot --inputs runs/comparison/evidenceconserve --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m evidenceconserve.statistics --records runs/main/evidenceconserve/records.jsonl --output results/intervals.json
```
