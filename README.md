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

The native RGB workflow below trains an encoder, pose-conditioned transport, candidate predictor and learned baselines. The numerical generator remains a fast finite-model control. CO3D, OmniObject3D and MVImgNet images, physical view graphs and original experiment checkpoints/runs are not bundled. Small generated checks establish executable training/evaluation, not manuscript performance or hardware capture timing.

## Native RGB training and acquisition

```bash
python -m pip install -e '.[vision]'
python -m evidenceconserve.native train --data data/episodes.jsonl --output checkpoints/native
```

Each episode retains the ordinary graph, budget and stopping fields and adds `object_id`, `split` (`train`, `policy_validation`, `calibration`, `test`), `start_view` and `voxel_shape`. Each view supplies RGB `image`, `image_shape: [height,width]`, `intrinsics` (3x3), `pose: [world_x,world_y,world_z,roll,pitch,yaw]` with radians, shared-coordinate `visibility`/quality, and `voxel_path` (a declared coarse occupancy grid). Directed reachability/cost uses the existing `from` contract. Images and occupancy from a candidate are opened only after capture; known camera rays/poses and acquired history drive proposal/ranking. Training object IDs cannot cross splits. RGB normalization fits train pixels only; validation alone selects model checkpoints; calibration/test images are never used for fitting.

Create `configs/native.json` containing `{"kind":"native","checkpoint":"../checkpoints/native/native.pt"}`; checkpoint paths resolve relative to that config file. Supply disjoint calibration/test subsets of the episodes and a development list containing the train/validation object IDs (the native checkpoint also registers them):

```bash
python -m evidenceconserve calibrate --data data/calibration.jsonl --frontend-config configs/native.json --development-objects data/development.json --methods evidenceconserve mc_dropout gennbv fisherrf set_transformer feature_concatenation --output checkpoints/native-calibration.json
python -m evidenceconserve suite --data data/test.jsonl --frontend-config configs/native.json --calibration checkpoints/native-calibration.json --methods evidenceconserve mc_dropout gennbv fisherrf set_transformer feature_concatenation --output runs/native
```

The complete native `evidenceconserve` method uses ResNet18/Dirichlet evidence, learned pose transport and the existing residual/overlap/reliability update, with the same calibrated stopping protocol. Model defaults are three-channel 224px inputs, 64 feature/hidden dimensions, dropout .2, AdamW 3e-4, up to 150 epochs and patience 15; `--config` accepts JSON overrides (see `native_training.DEFAULTS`). Transport is a positive relative-pose-conditioned KxK matrix, forced to identity at equal poses, fitted on overlapping train view pairs against frozen encoder evidence with a small identity penalty. Candidate dynamics predicts evidence and embedding from a known candidate pose, relative pose, acquired mean features/posterior and novelty. The explicit architectures, losses and widths fill details not specified in the manuscript; they are implementation choices rather than recovered original models.

| Native comparison | Executable algorithm and declared adaptation |
|---|---|
| `set_transformer` | Original upstream ISAB/ISAB/PMA attention modules, trained with the evidential classification loss on view prefixes; permutation invariant |
| `feature_concatenation` | Ordered padded feature concatenation plus trained evidential head |
| `mc_dropout` | Stochastic candidate-dynamics dropout; BALD mutual information `H(mean p)-mean H(p)` divided by charged cost, without candidate RGB |
| `gennbv` | Released GenNBV occupancy-Conv3D and periodic pose-history encoder; actor/critic and clipped PPO with GAE learn coverage/path/terminal reward on the common legal finite graph. This is the manuscript's GenNBV-style graph adaptation, not Isaac Gym drone training |
| `fisherrf` | Differentiable emission/absorption radiance MLP fits captured RGB online; squared RGB Jacobians yield diagonal Fisher and the original `sum(H_candidate/(H_observed+lambda))` criterion. CPU MLP replaces the upstream custom 3D Gaussian rasterizer; the criterion is implemented, original 3DGS-backend reproduction remains separate |

All comparisons share the visual evidence encoder, graph, capture budget and method-specific frozen stopping calibration. Candidate ranking and predicted risk reduction are distinct: learned acquisition priorities select views, while the same evidence-risk model governs eligibility. The start view is common and exogenous. GenNBV simulation rewards may use archived train/validation coverage; deployment consumes acquired occupancy only. FisherRF resets its field prior for every episode and trains only on acquired RGB. Licenses, upstream sources and exact adaptation details are retained in [third_party/NOTICE.md](src/evidenceconserve/third_party/NOTICE.md).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL python verify_native.py --output runs/native-functional
```

On 2026-11-01 the generated five-object check trained every component (including actual PPO updates), independently calibrated and replayed all six native methods, checked nonzero gradients and trainable-weight updates, rejected split overlap, verified no calibration/test fitting images and no uncaptured inference images, saved/loaded all six learned modules with atol 1e-7, checked exact equal-pose transport, Set permutation invariance and the analytic constant-density Fisher diagonal `6*(1-exp(-1))² = 2.3974583`. Online RF gradients were nonzero. Generated cases and one-epoch settings are functionality checks, not performance comparisons.

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

This trace route remains available for original external checkpoints and simulators. Native implementations above execute their stated algorithms and graph/renderer adaptations directly; an external trace alone does not establish native reproduction. Original AIR-OSVP CUDA execution is validated below. The upstream Isaac Gym/3DGS backends remain distinct from the declared graph/CPU-radiance adaptations.

The original one-shot source is publicly available at [AIR-OSVP](https://github.com/psc0628/AIR-OSVP), revision `6847c0a57ca3b4e9d9a82d4cbf12f08074f71a89`; it must not be confused with the different ActiveImplicitRecon paper. Its [installation](https://github.com/psc0628/AIR-OSVP/blob/main/poco_pointr/README.md) requires original POCO/PoinTr and compiled Pointnet2/Chamfer. Those dependencies now build and execute in the private Linux CUDA environment below, with public original weights. The native pipeline and unchanged-batch graph charging are exercised on one public capture. Full original RGB/physical-graph evaluation episodes are still absent. Original FisherRF needs its [modified CUDA rasterizer/knn/Hessian extensions](https://github.com/JiangWenPL/FisherRF#environment-setup); this revision implements its Fisher algorithm with the explicitly declared CPU radiance backend.

## Original AIR-OSVP execution entry

`air_osvp.py` now supplies a direct original-source pipeline, in addition to the generic trace evaluator. It fetches pinned AIR-OSVP, PoinTr and POCO source trees and their submodules, retains upstream licenses and applies exactly the author-documented source overlay in a private workspace. Author model/script source is fetched at runtime rather than redistributed here (the pinned AIR repository has no root license). It executes the original POCO `train_and_infer_one.py`, then the original COSC `infer_once.py`; it does not accept an arbitrary predicted-view file as native inference.

```bash
python -m evidenceconserve.air_osvp fetch --output .deps/air-sources
python -m evidenceconserve.air_osvp prepare --sources .deps/air-sources --output .deps/air-runtime
python -m evidenceconserve.air_osvp preflight --workspace .deps/air-runtime
python -m evidenceconserve.air_osvp infer --data data/air-episodes.jsonl --workspace .deps/air-runtime --checkpoint data/original-cosc-best_ckpt.pth --poco-checkpoint data/original-poco-checkpoint.pth --encoder-checkpoint checkpoints/native/native.pt --poco-epochs 10 --output runs/air-calibration
python -m evidenceconserve.external calibrate --data runs/air-calibration/traces.jsonl --policy-config runs/air-calibration/policy.json --development-objects data/air-development.json --output checkpoints/air-calibration.json
```

Install/build the original dependency stack using its retained README or the isolated recipe below. Public original POCO weights are linked by the AIR installation to [POCO's ShapeNet release](https://github.com/valeoai/POCO/releases/download/v0.0.0/ShapeNet_3k.zip); COSC data/weights are linked to [the author's Kaggle dataset](https://www.kaggle.com/datasets/sicongpan/air-osvp-dataset). `preflight` reaches compiled imports before checking checkpoints. The original [Chamfer setup](https://github.com/yuxumin/PoinTr/blob/master/extensions/chamfer_dist/setup.py) and [pointnet2 setup](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/setup.py) execute actual CUDAExtension/BuildExtension sources in the private Linux prefix; an alternate CPU algorithm is not substituted.

Each episode retains the common graph/budget/start-view metadata and adds `views[j].air_view_id` (unique original index 0–31) and calibrated `air_to_world[4,4]`. Camera centres must match the original `view_space_32.txt` transformed into the common world; original view directions point towards the object. The initial view's `air_inputs` provides `observed_view_indices: [start_view]` and paths, relative to the episode JSONL, to `surface` [N,3] observed points, `scene` [M,4] xyz plus binary observed/free-space occupancy supervision, and `bbx` [Q,3] prior-box query points. Points are in the original AIR object frame. These are exactly the author's POCO input contracts read by `datasets/cosc.py`; future scans, full object truth and class labels must not produce the input supervision. The POCO ShapeNet prior is fine-tuned on the initial observation (author default ten epochs; one epoch may be used for a functionality check).

COSC returns the original 32-way sigmoid thresholded at .5. The wrapper keeps that set unchanged, removes the already captured start, and enumerates the minimum-cost legal ordering using the same directed graph charges and maximum captures. Off-graph, unreachable or over-budget batches are explicitly rejected; no extra numeric selector prunes the original output. Exact routing supports up to eight additional views, covering the paper's six-capture setting. Only selected RGB images enter the same frozen evidence encoder; trace eligibility uses the shared candidate risk-reduction model and the same minimum-view rule. `traces.jsonl` and `policy.json` then enter the existing independent episode calibration/evaluation commands. Include all encoder/COSC/POCO fitting and configuration-selection objects in the development registration.

```bash
python verify_air_glue.py --native-fixture runs/native-functional --output runs/air-glue
```

Generated routing/trace checks verify the unchanged batch's hand route [0,2,1] and cost 3, budget/off-graph/duplicate rejection, label isolation, selected-only shared-encoder capture and trace readback. These cover graph/classifier glue; the separate original CUDA execution check below covers actual POCO/COSC algorithms.

The extended CPU smoke also passed learned-head fitting/save/load, registration JSON, independent object calibration and replay, exact oracle hand cost/regret and inserted-copy zero increment. External adapters were checked with explicitly generated functionality fixtures; these are not runs of the named learned baselines.

## Original CUDA baseline environment and one-capture check

The standalone helpers in `native/` use a separate Linux Python 3.10 / PyTorch 2.0.1+cu118 / GCC 11 / CUDA 11.8 prefix. They do not change the main project's environment. CUDA development headers for cuSPARSE, cuBLAS and cuSOLVER are required in addition to nvcc/cudart. The isolated [MetricDamage CUDA bootstrap](https://github.com/Hqa77880011/MetricDamage/blob/main/native/cuda/bootstrap.py) creates this prefix without system installs; AIR's remaining dependencies are listed in `native/requirements-air.txt`.

```bash
# In Linux/WSL; PREFIX is the isolated CUDA prefix, WORK the prepared AIR workspace.
"$PREFIX/bin/python" -m pip install -r native/requirements-air.txt
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git "$CACHE/Pointnet2_PyTorch"
git -C "$CACHE/Pointnet2_PyTorch" checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
"$PREFIX/bin/python" native/build_air_cuda.py --prefix "$PREFIX" --workspace "$WORK" --pointnet2 "$CACHE/Pointnet2_PyTorch"
"$PREFIX/bin/python" native/fetch_air_assets.py --output "$CACHE/air-assets"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PREFIX/bin/python" native/verify_air_cuda.py --workspace "$WORK" --assets "$CACHE/air-assets" --output "$CACHE/air-check"
```

Use the `air_osvp fetch` / `prepare` commands above to create `WORK` before building. `build_air_cuda.py` compiles the original Pointnet2 and PoinTr Chamfer sources for the actual GPU, checks original CUDA FPS on four hand points and checks Chamfer distance/backward against a hand-sized distance calculation. Compatibility changes select the GPU architecture, restrict PoinTr's import-time registry to COSC, change POCO transforms to PyG 2.x `Data.keys()`, and put voxel-grid batch indices on the point tensor's device. COSC/POCO networks, attention/convolution math, native kernels, losses and the original optimizer step are preserved.

`fetch_air_assets.py` retrieves the author's actual COSC checkpoint, POCO's released ShapeNet checkpoint and only `Asian_Dragon/rotate_0/cloud_0.txt` from public AIR dataset version 1. `verify_air_cuda.py` executes the original `train_and_infer_one.py` for one adaptation update and the original `infer_once.py` for COSC. It records actual gradients/weight changes, original native FPS calls, the full 32 probabilities and unchanged .5-threshold batch. Failed stages remain errors; completed `--stage poco` / `--stage cosc` results can be collected with `--stage report` without another adaptation update.

The declared `public-single-capture-v1` check derives occupied endpoints and free-ray supervision only from that one observed cloud and original view-0 camera centre. A fixed 20 cm prior box supplies a 48³ query grid; 2,048 manifold and 1,024 nonmanifold training points bound the single update. These are explicit functionality settings, with no future clouds, object class or ground-truth shape inputs. Public original RGB/physical-graph episodes and the manuscript's exact scene construction are still needed for the original common-task benchmark. The existing unchanged-batch routing rejects a native set that exceeds the shared graph budget; it never prunes that set to manufacture a successful comparison.

On 2026-11-01 the original CUDA FPS/Chamfer and complete POCO→COSC stages passed on the RTX 4060 Laptop GPU. The 3,583-point single capture produced one POCO update with gradient norm 7.69335, 238 gradient tensors and L1 parameter change 12,579.4336; its 1,161 completed points entered the original 14,971,680-parameter COSC, including actual native FPS calls at 512 and 256 centres. Peak torch allocation was 3,066,641,920 bytes for POCO and 225,563,648 for COSC. The original .5-threshold set was `[1,10,15,18,20,26]`, unchanged. Actual `charge_batch` rejects the seven total captures at budget six and accepts budget seven on a declared complete 32-view public graph: route `[0,20,15,26,1,10,18]`, cost 8.511185698. This graph uses unit capture plus Euclidean travel cost and is explicitly not the missing original physical graph. Raw probabilities, completion, update/call counts and rejection reason are retained; no performance figure or full training is claimed.
