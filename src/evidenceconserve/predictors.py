"""Train frozen candidate/geometry predictors from object-disjoint cached features.

Cache keys: candidate_features (N,F), evidence (N,C), optional quality (N,),
candidate_group/candidate_split (N,); overlap_features (M,G), overlap (M,),
overlap_group/overlap_split (M,). Targets are offline supervision. Features
must be available before capture; class labels are never prediction inputs.
The small MLPs and ensemble spread are implementation choices, not recovered
original experiment models or calibrated uncertainty bounds.
"""
import argparse
import copy
import json
import uuid
from itertools import product
from pathlib import Path

import numpy as np


def _splits(groups, splits):
    aliases = {"policy_validation": "val", "validation": "val"}
    splits = np.array([aliases.get(str(s), str(s)) for s in splits])
    if not set(splits) <= {"train", "val", "calibration", "test"}:
        raise ValueError("Splits must be train/val/calibration/test")
    assignment = {}
    for group, split in zip(groups, splits):
        group = str(group)
        if group in assignment and assignment[group] != split:
            raise ValueError(f"Object {group} occurs in multiple splits")
        assignment[group] = split
    if not np.any(splits == "train") or not np.any(splits == "val"):
        raise ValueError("Predictor fitting requires nonempty train and val splits")
    return splits, assignment


def _project(values, task, classes):
    values = np.asarray(values).copy()
    if task == "candidate":
        values[:, :classes] = np.maximum(values[:, :classes], 0.)
        if values.shape[1] > classes:
            values[:, classes:] = np.clip(values[:, classes:], 0., 1.)
    else:
        values = np.clip(values, 0., 1.)
    return values


def _fit(features, targets, splits, task, classes, hidden_sizes, alphas, seeds,
         epochs, patience):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    train, val = splits == "train", splits == "val"
    x_scaler = StandardScaler().fit(features[train])
    y_scaler = StandardScaler().fit(targets[train])
    tx, vx = x_scaler.transform(features[train]), x_scaler.transform(features[val])
    ty = y_scaler.transform(targets[train])
    best = None
    candidates = []
    for width, alpha in product(hidden_sizes, alphas):
        models, best_epochs = [], []
        for seed in seeds:
            model = MLPRegressor(hidden_layer_sizes=(int(width),), alpha=float(alpha),
                                 solver="adam", learning_rate_init=.001,
                                 batch_size=min(64, len(tx)), shuffle=True,
                                 random_state=int(seed), max_iter=1)
            best_loss, best_model, stale = float("inf"), None, 0
            for epoch in range(1, epochs + 1):
                # partial_fit performs one epoch; validation does not update weights.
                model.partial_fit(tx, ty.ravel() if ty.shape[1] == 1 else ty)
                raw = np.asarray(model.predict(vx)).reshape(len(vx), -1)
                predicted = _project(y_scaler.inverse_transform(raw), task, classes)
                score = float(np.mean((predicted - targets[val]) ** 2))
                if not np.isfinite(score) or not np.isfinite(model.loss_):
                    raise ValueError("Predictor training produced a nonfinite loss")
                if score < best_loss - 1e-10:
                    best_loss, best_model, best_epoch, stale = score, copy.deepcopy(model), epoch, 0
                else:
                    stale += 1
                if stale >= patience:
                    break
            models.append(best_model)
            best_epochs.append(best_epoch)
        predictions = np.array([
            _project(y_scaler.inverse_transform(np.asarray(m.predict(vx)).reshape(len(vx), -1)), task, classes)
            for m in models])
        score = float(np.mean((predictions.mean(0) - targets[val]) ** 2))
        candidates.append(dict(width=int(width), alpha=float(alpha), validation_mse=score))
        if best is None or score < best[0]:
            best = (score, models, best_epochs, int(width), float(alpha))
    return dict(models=best[1], best_epochs=best[2], input_scaler=x_scaler,
                target_scaler=y_scaler, validation_mse=best[0], width=best[3],
                alpha=best[4], task=task, classes=classes, input_dimensions=features.shape[1],
                output_dimensions=targets.shape[1], selection=candidates)


def fit_predictors(cache, output, *, seeds=(13, 37, 73), epochs=150, patience=15,
                   hidden_sizes=(32, 64), alphas=(1e-4, 1e-3)):
    """Fit on training objects, select on policy validation, then freeze both heads.

    Calibration/test rows may be present but never enter scaling, optimization,
    checkpoint selection or hyperparameter selection. At most 20 configurations
    per task are allowed; each configuration uses the declared independent seeds.
    """
    import joblib
    import sklearn

    if epochs < 1 or patience < 1 or not seeds or not hidden_sizes or not alphas:
        raise ValueError("Require positive epochs/patience and nonempty search/seeds")
    if len(hidden_sizes) * len(alphas) > 20:
        raise ValueError("At most 20 hyperparameter configurations per task")
    if isinstance(cache, (str, Path)):
        with np.load(cache, allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
    else:
        arrays = {k: np.asarray(v) for k, v in cache.items()}
    classes = arrays["evidence"].shape[1]
    if classes < 2:
        raise ValueError("Evidence targets need at least two classes")
    candidate_y = np.asarray(arrays["evidence"], float)
    if np.any(candidate_y < 0):
        raise ValueError("Evidence supervision must be nonnegative")
    if "quality" in arrays:
        quality = np.asarray(arrays["quality"], float).reshape(-1, 1)
        if np.any((quality < 0) | (quality > 1)):
            raise ValueError("Quality supervision must lie in [0,1]")
        candidate_y = np.column_stack((candidate_y, quality))
    overlap_y = np.asarray(arrays["overlap"], float).reshape(-1, 1)
    if np.any((overlap_y < 0) | (overlap_y > 1)):
        raise ValueError("Overlap supervision must lie in [0,1]")
    tasks, object_splits = {}, {}
    for task, targets in (("candidate", candidate_y), ("overlap", overlap_y)):
        x = np.asarray(arrays[task + "_features"], float)
        groups, split = arrays[task + "_group"], arrays[task + "_split"]
        if x.ndim != 2 or not x.shape[1] or len({len(x), len(targets), len(groups), len(split)}) != 1:
            raise ValueError(f"Invalid aligned {task} cache arrays")
        if not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError("Cache features and targets must be finite")
        splits, assignment = _splits(groups, split)
        for group, value in assignment.items():
            if group in object_splits and object_splits[group] != value:
                raise ValueError(f"Object {group} crosses candidate/overlap split assignments")
            object_splits[group] = value
        tasks[task] = (x, targets, splits)
    artifact = dict(version=1, sklearn_version=sklearn.__version__, classes=classes,
                    has_quality="quality" in arrays, seeds=list(seeds), epoch_cap=epochs,
                    patience=patience, object_counts={s: sum(v == s for v in object_splits.values())
                                                     for s in ("train", "val", "calibration", "test")})
    for task, (x, targets, splits) in tasks.items():
        artifact[task] = _fit(x, targets, splits, task, classes, hidden_sizes,
                              alphas, seeds, epochs, patience)
    train_objects = sorted(k for k, v in object_splits.items() if v == "train")
    val_objects = sorted(k for k, v in object_splits.items() if v == "val")
    artifact["development_objects"] = sorted(train_objects + val_objects)
    artifact["registration"] = dict(
        artifact_id=str(uuid.uuid4()), train_objects=train_objects,
        validation_objects=val_objects, seeds=list(seeds), classes=classes,
        epoch_cap=epochs, patience=patience,
        selected={task: {k: artifact[task][k] for k in
                         ("width", "alpha", "best_epochs", "input_dimensions",
                          "output_dimensions", "validation_mse")}
                  for task in tasks})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / "predictors.joblib")
    summary = {k: v for k, v in artifact.items() if k not in tasks}
    summary["tasks"] = {task: {k: v for k, v in artifact[task].items()
                               if k not in ("models", "input_scaler", "target_scaler")}
                        for task in tasks}
    (output / "predictors.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return artifact


def load_predictors(path):
    """Load a trusted predictor artifact written by fit_predictors."""
    import joblib
    return joblib.load(path)


def predict_features(features, artifact, task="candidate"):
    """Return ensemble means/stds; spread is descriptive, not a risk certificate."""
    fitted = artifact[task]
    features = np.asarray(features, float)
    if features.ndim != 2 or features.shape[1] != fitted["input_dimensions"] or not np.isfinite(features).all():
        raise ValueError(f"Expected finite (N,{fitted['input_dimensions']}) {task} features")
    x = fitted["input_scaler"].transform(features)
    values = np.array([_project(fitted["target_scaler"].inverse_transform(
        np.asarray(m.predict(x)).reshape(len(x), -1)), task, artifact["classes"])
        for m in fitted["models"]])
    return values.mean(0), values.std(0)


def predict_case(case, artifact):
    """Fill pre-capture predictions from features without reading class/evidence labels."""
    result = copy.deepcopy(case)
    views = result["views"]
    if result["classes"] != artifact["classes"]:
        raise ValueError("Case class count differs from the trained evidence head")
    evidence, spread = predict_features([v["candidate_features"] for v in views], artifact)
    for i, view in enumerate(views):
        view["predicted_evidence"] = evidence[i, :artifact["classes"]].tolist()
        view["predicted_evidence_std"] = spread[i, :artifact["classes"]].tolist()
        if artifact["has_quality"]:
            view["predicted_quality"] = float(evidence[i, artifact["classes"]])
        pairs = view.get("overlap_features", {})
        if pairs:
            ids = list(pairs)
            prediction, uncertainty = predict_features([pairs[j] for j in ids], artifact, "overlap")
            view["predicted_overlaps"] = {str(j): float(p[0]) for j, p in zip(ids, prediction)}
            view["predicted_overlap_std"] = {str(j): float(p[0]) for j, p in zip(ids, uncertainty)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--input", required=True, help="NPZ cache described in module docstring")
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--patience", type=int, default=15)
    apply = sub.add_parser("apply")
    apply.add_argument("--artifact", required=True)
    apply.add_argument("--input", required=True, help="JSON array of feature-bearing episodes")
    apply.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        fit_predictors(args.input, args.output, epochs=args.epochs, patience=args.patience)
    else:
        artifact = load_predictors(args.artifact)
        cases = json.loads(Path(args.input).read_text(encoding="utf-8"))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([predict_case(c, artifact) for c in cases], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
