# EvidenceConserve

Implementation of **Evidence-conserving active view acquisition with risk-calibrated stopping**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Validated on Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1 on CPU. `python -m evidenceconserve.smoke --output runs/smoke` checks JSONL loading, all 11 policies, episode calibration save/load, deterministic replay, exact-duplicate zero gain, positive disjoint-surface gain, hand-calculated Brier/NLL/F1 and plotting. On 2026-11-01 all checks passed; four generated test episodes per policy were used. These generated cases check functionality and do not reproduce the paper results.

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

Use `--methods` to select `evidenceconserve`, `naive_sum`, `no_transport`, `uniform_views`, `entropy_nbv`, `random_views`, `overlap_only`, `no_overlap`, `no_reliability`, `no_cost`, or `no_calibration`. Every changed policy is calibrated independently; the no-calibration ablation reports its stops separately.

## Outputs

`runs/main/evidenceconserve/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m evidenceconserve plot --inputs runs/comparison/evidenceconserve --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m evidenceconserve.statistics --records runs/main/evidenceconserve/records.jsonl --output results/intervals.json
```

## Method and metric mapping

| Paper component | Entry | Command / convention |
|---|---|---|
| Overlap, explained evidence, reliability-weighted residual | `core.increment`, `core.run_case` | `run`; exact replay admits zero support, identity transport is explicit ablation |
| Frozen-policy object-episode stopping | `calibration.calibrate`, `select_correction` | `calibrate`; simultaneous fixed-grid Clopper-Pearson bound, calibration objects disjoint from development/test |
| Cost/entropy/fixed/random selectors; residual/overlap/reliability ablations | `core.METHODS` | `suite`; pre-capture predicted evidence only |
| Accuracy, macro F1 over all configured classes, NLL, Brier, risk-coverage, views per correct decision | `core.score`, `cli.summarize` | `run` / `suite`; forced-stop errors and calibrated-stop errors have separate denominators |
| Duplicate/new-surface gains, gain ratio, path cost, capture duration, latency and RSS | `core.score`, `cli.execute` | `run`; `capture_seconds` defaults to zero when no timing is supplied |
| Curves and bootstrap intervals | `cli.plot`, `statistics` | commands above |

For imported episodes, `views[j].from[str(previous_view)]` can declare directed `reachable` and `cost` metadata. Missing directed metadata uses the candidate's fixed reachability/cost. Visibility coordinates must align across views. Candidate `predicted_evidence` is supplied before capture; `evidence` is revealed only after the selected action. The default predictor `3 * current_probability` is a fixed numerical convention, not the paper's learned candidate model. Seed defaults to 17 for the random selector and is independent of labels.

## Visual data and implementation scope

To use a trained visual encoder, pass `--frontend-config` to both calibration and execution. The JSON contains an argv `command`, optional `cwd`, and all `training_objects`. `python -m evidenceconserve.torchscript_frontend --checkpoint PATH --image-size 224` accepts a captured image request on stdin and returns nonnegative class evidence; optional `--imagenet-normalize` must match encoder training. The calibration artifact records frontend configuration and prohibits development/calibration/test object overlap.

CO3D, OmniObject3D and MVImgNet images, trained encoders, overlap/transport estimators, learned candidate predictors and physical view graphs are not bundled. Their training and dataset-specific conversion remain external to this numerical implementation. GenNBV, FisherRF, MC-dropout, Set Transformer and oracle sequence baselines are not implemented here; `entropy_nbv` is the entropy selector for the same supplied candidate-evidence model, rather than a recreation of those learned baselines. The original paper's raw runs and checkpoints were not found in this repository. This revision implements and verifies the numerical policy mechanism; it does not establish the paper's reported performance, energy, physical capture time, oracle regret or duplicate-induced stop-time change.
