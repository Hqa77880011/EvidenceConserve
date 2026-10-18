# EvidenceConserve

Implementation of **Evidence-conserving active view acquisition with risk-calibrated stopping**.

Accumulates only quality-weighted new-surface and unexplained positive evidence, ranks reachable views by predicted risk reduction per motion cost and separates calibrated from forced stopping.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/EvidenceConserve.git
cd EvidenceConserve
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Each object episode provides candidate view visibility descriptors, quality, cost and reachability. Recorded `evidence` is read only after that view is selected. Alternatively a frozen local image model supplies it through the capture adapter below. Optional `predicted_evidence` comes from a frozen candidate model, never test annotations. Calibration corrections are fitted by the executable independent-object workflow below; generated cases no longer contain an arbitrary correction.

```bash
python -m evidenceconserve prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m evidenceconserve prepare --count 200 --seed 37 --output data/calibration.jsonl
python -m evidenceconserve calibrate --data data/calibration.jsonl --development-objects configs/development_objects.json --output checkpoints/calibration.json --delta 0.05 --minimum-coverage 0.1
python -m evidenceconserve run --data data/workload.jsonl --calibration checkpoints/calibration.json --output runs/main
python -m evidenceconserve suite --data data/workload.jsonl --calibration checkpoints/calibration.json --output runs/comparison
```

Select methods with `--methods`. evidenceconserve, naive_sum, no_transport, uniform_views and no_calibration.

`calibrate` freezes acquisition settings and replays each independent calibration object to the acquisition budget. It stores every decision prefix, then chooses a nonnegative upper-risk correction on a prespecified grid. Each candidate is evaluated at its first permitted stop, using one-sided Clopper–Pearson bounds with error probability `delta / candidate_count`. The actual candidate count and all bounds are saved. The selected correction maximizes empirical coverage subject to the registered error upper bound and minimum coverage. An infeasible calibration disables calibrated stopping; those episodes end as forced stops. The bound applies only to calibrated stops under independent exchangeable object episodes: forced-stop errors, uncalibrated stopping and deployment shift have no such guarantee and are reported separately. This is a conservative binomial selective-error calibration implementation, rather than an unimplemented claim of an arbitrary conformal correction.

Each calibration object must have a unique `object_id` (or `cluster`/`family` fallback). `--development-objects` is a required JSON list of **all** objects used for visual/evidence/overlap/candidate fitting, validation, or policy/configuration selection. Create that manifest from your registered development splits; use `[]` only for the fixed numerical generator when no objects were used for fitting or tuning. Calibration rejects overlap with that manifest or frontend training IDs. Evaluation rejects both development/test and calibration/test overlap. Artifacts record these object IDs, method, class count, acquisition budget, alpha, minimum views, value threshold and frontend configuration. Repeat calibration after changing weights, geometry, candidate predictions, policy or stopping settings. `--grid-size` sets the registered grid before labels are examined. Only `--methods no_calibration` runs without an artifact.

## Capture from a real visual frontend

Install `python -m pip install -e ".[vision]"` for the supplied TorchScript image adapter. A locally trained frozen model must take RGB `[1,3,H,W]` in `[0,1]` (or ImageNet-normalized with the adapter flag), and return nonnegative class evidence, or a dictionary with `evidence` and optional scalar `quality`. Export the evidential head with its backbone; raw logits must not be passed as nonnegative evidence. This adapter loads an existing checkpoint and downloads nothing.

Save this configuration as `configs/frontend.json`, using your existing trained checkpoint and actual independent training-object IDs:

```json
{
  "command": ["python", "-m", "evidenceconserve.torchscript_frontend", "--checkpoint", "checkpoints/evidence.ts", "--image-size", "224"],
  "cwd": "..",
  "training_objects": ["training-object-1", "training-object-2"]
}
```

Imported episode rows use the same numerical schema as `prepare`, with `object_id` and each view's `image` path added. Image paths resolve relative to the JSONL file. With this adapter, `evidence` can be omitted from view records; candidate `visibility`, `quality`, positive `cost` and `reachable` remain required. Only the selected image is opened. `predicted_evidence` must be prepared independently if using a learned candidate model; the explicit fallback is a posterior-based numerical heuristic. All views share one registered surface-coordinate descriptor. Scalar costs describe the supplied reachable-action protocol, not unmeasured robot travel.

```bash
python -m evidenceconserve calibrate --data data/real_calibration.jsonl --development-objects data/development_objects.json --frontend-config configs/frontend.json --output checkpoints/real_calibration.json
python -m evidenceconserve run --data data/real_test.jsonl --frontend-config configs/frontend.json --calibration checkpoints/real_calibration.json --output runs/real
```

Another local visual/geometry implementation can use the same `command` JSON-stdin/stdout protocol. Input has `protocol="evidenceconserve.capture.v1"`, episode/object IDs, selected view metadata and already acquired observations; it contains no evaluation label or unacquired image evidence. Return `{ "evidence": [0.1, 2.0, 0.2], "quality": 0.9 }`, optionally refined `visibility` and `transports`. `transports` maps each previous view index (a string) to its nonnegative K-by-K `W_tj`; it belongs to the current target view. Without those matrices, identity is the declared transport convention. External failures and malformed/nonfinite/negative evidence raise errors. Observations are cached for later residual updates and saved with decision traces in results.

## Evaluation and visualization

Accuracy, NLL/Brier, acquisition count, motion cost, forced stopping, unsafe calibrated stop, exact-duplicate gain and new-surface gain.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m evidenceconserve plot --inputs runs/comparison/evidenceconserve --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

Inputs are the shared visual frontend's evidence and geometry descriptors. The core works with exact visibility masks or calibrated proxies. Visibility uses IoU and explained support uses target-specific overlap-weighted transport. The default simulator supplies controlled object episodes; the local capture adapter connects actual image evidence. Model training, overlap supervision and robot motion must be supplied and frozen independently; this package does not manufacture learned checkpoints or physical-platform measurements.


## Uncertainty and paired comparisons

```bash
python -m evidenceconserve.statistics --records runs/main/evidenceconserve/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.

## Fixed reproduction setting

Frozen numerical evidence model: independent calibration/test objects, three classes, eight candidate views, six-view budget; fixed 21-point CP grid.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
