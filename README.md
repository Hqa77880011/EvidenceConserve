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

Each object episode provides candidate view visibility descriptors, observed evidence revealed on capture, quality, cost and reachability. Optional predicted_evidence comes from a frozen candidate model, never test annotations. A calibration correction must be estimated using independent complete episodes replayed under the frozen policy.

```bash
python -m evidenceconserve prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m evidenceconserve run --data data/workload.jsonl --output runs/main
python -m evidenceconserve suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. evidenceconserve, naive_sum, no_transport, uniform_views and no_calibration.

## Evaluation and visualization

Accuracy, NLL/Brier, acquisition count, motion cost, forced stopping, unsafe calibrated stop, exact-duplicate gain and new-surface gain.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m evidenceconserve plot --inputs runs/comparison/evidenceconserve --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

Inputs are the shared visual frontend's evidence and geometry descriptors. The core works with exact visibility masks or calibrated proxies. Visibility uses IoU, explained support uses overlap-weighted transport, and the default simulator supplies controlled object episodes. Frame-level evidence training and robot motion are external to this numerical policy interface.


## Uncertainty and paired comparisons

```bash
python -m evidenceconserve.statistics --records runs/main/evidenceconserve/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.
