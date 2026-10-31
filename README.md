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

CO3D, OmniObject3D and MVImgNet images, trained image encoders, pose/transport estimators and physical view graphs are not bundled. The cached-feature candidate/quality/overlap predictors, exact finite-graph oracle and duplicate-stop diagnostic are implemented below. GenNBV, FisherRF, MC-dropout and Set Transformer model training/inference remain external, with a common complete-trace calibration/evaluation adapter. `entropy_nbv` is the entropy selector for the supplied candidate-evidence model. Original raw experiment runs/checkpoints were not found; this revision does not establish manuscript performance, physical energy or hardware capture timing.

## Cached-feature predictor lifecycle

`predictors.py` trains frozen candidate evidence/quality and geometry-overlap heads. The NPZ cache contains `candidate_features[N,F]`, nonnegative `evidence[N,K]`, optional `quality[N]`, and `candidate_group`/`candidate_split`; the geometry rows use `overlap_features[M,G]`, `overlap[M]` in [0,1], and `overlap_group`/`overlap_split`. Splits are `train`, `val` (or `policy_validation`), `calibration`, `test`, and source objects cannot cross any task/split. Features must be available before capture; target evidence and class labels are not candidate inputs.

```bash
python -m evidenceconserve.predictors train --input data/predictor-cache.npz --output checkpoints/predictors --epochs 150 --patience 15
python -m evidenceconserve calibrate --data data/calibration.jsonl --predictors checkpoints/predictors/predictors.joblib --development-objects configs/development_objects.json --output checkpoints/learned-calibration.json
python -m evidenceconserve run --data data/test.jsonl --predictors checkpoints/predictors/predictors.joblib --calibration checkpoints/learned-calibration.json --output runs/learned
```

Each view supplies `candidate_features[F]` and `overlap_features[str(previous_view_index)]` vectors. Predicted evidence/quality score uncaptured actions; learned overlap and optional `correspondence_quality[str(j)]` weight the explained evidence update. Input and target scalers fit training rows only. Widths {32,64}, regularization {1e-4,1e-3}, seeds {13,37,73}, a 150-epoch cap and patience 15 use policy-validation MSE for checkpoint/configuration selection. These small-MLP choices implement the unspecified predictor details rather than recover original weights. Ensemble spread is descriptive uncertainty, not a calibrated risk bound. The calibration artifact records predictor registration and excludes its train/validation objects from calibration and test.

## Oracle and duplicate intervention

```bash
python -m evidenceconserve run --data data/test.jsonl --calibration checkpoints/calibration.json --oracle-reference --duplicate-intervention --output runs/diagnostics
python -m evidenceconserve run --data data/test.jsonl --methods oracle --output runs/oracle
```

The privileged `interventions.oracle` reads archived evidence/labels only for evaluation, enumerates reachable simple paths under the same graph and capture budget, and minimizes charged cost among correct decisions satisfying the raw target-risk and minimum-view rule. Its stop is not a calibrated certificate. `oracle_regret` is policy cost minus oracle cost only when both satisfy that declared low-risk/correct criterion; other cases remain null. Error regret is reported separately. If no such path exists, the fallback maximizes true-class support. Exact search is exponential and defaults to at most ten candidates; set the episode's `oracle_max_views` explicitly for a larger search.

The duplicate intervention forces one identical copy after the first capture, retaining the capture budget, original policy and frozen correction. `duplicate_stop_time_change` counts the difference in captures, `duplicate_increment` measures support admitted on the inserted copy, and cost/prediction changes are logged. This perturbed sequence is a diagnostic and does not inherit the original calibration guarantee. New-surface gain uses `reference_visibility` or an explicitly declared `visibility_is_reference=true`; it is null when no reference overlap is supplied. The bundled finite generator declares reference geometry and uses a development target alpha=.2; set alpha=.03 in imported paper-protocol episodes.

## Externally trained baseline trace adapter

Export the actual model's complete held-out object episodes, not renamed output from a numerical selector. Each JSONL row supplies the ordinary episode metadata/graph and a `trace` list. Each state contains `views` (the acquired prefix), `probability[K]`, cumulative `cost`, and the frozen policy's `eligible` boolean. Optional measured `gains` and `latency_ms` enable their metrics; absent diagnostics remain missing. The adapter checks prefix reachability, common graph cost, budget and disjoint source objects.

Register a JSON policy file such as `{"baseline":"gennbv","checkpoint":"your-frozen-checkpoint-id","seed":13,"budget":6}`; baseline names are `gennbv`, `fisherrf`, `mc_dropout`, `set_transformer`, `active_implicit`. Record all encoder/policy fitting and configuration-selection objects in the development list.

```bash
python -m evidenceconserve.external calibrate --data data/gennbv-calibration.jsonl --policy-config configs/gennbv.json --development-objects data/gennbv-development.json --output checkpoints/gennbv-calibration.json
python -m evidenceconserve.external evaluate --data data/gennbv-test.jsonl --policy-config configs/gennbv.json --calibration checkpoints/gennbv-calibration.json --output runs/gennbv
python -m evidenceconserve plot --inputs runs/gennbv runs/learned/evidenceconserve --output figures/comparison
```

Actual GenNBV/FisherRF/MC-dropout/Set-Transformer architecture training, their original inference/export scripts and checkpoints are still required externally, so their algorithm reproduction remains incomplete. The adapter supplies an executable matched evaluation route without substituting a different algorithm.

The extended CPU smoke also passed learned-head fitting/save/load, registration JSON, independent object calibration and replay, exact oracle hand cost/regret and inserted-copy zero increment. External adapters were checked with explicitly generated functionality fixtures; these are not runs of the named learned baselines.
