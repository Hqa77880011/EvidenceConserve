"""Calibrate and score traces exported by trained acquisition baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import select_correction
from .cli import read_jsonl, summarize, write_jsonl
from .core import score
from .frontend import object_id


def validate(rows):
    """Validate externally recorded post-capture beliefs and reachable prefixes."""
    if not rows:
        raise ValueError("External baseline trace corpus is empty")
    for row in rows:
        if not 0 <= row["label"] < row["classes"]:
            raise ValueError("External evaluation labels must index the configured classes")
        previous = []
        previous_cost = 0.0
        if not row["trace"]:
            raise ValueError("Each external object needs a complete nonempty trace")
        for state in row["trace"]:
            p = np.asarray(state["probability"], dtype=float)
            indices = state["views"]
            if (p.shape != (row["classes"],) or not np.isfinite(p).all() or (p < 0).any()
                    or not np.isclose(p.sum(), 1, atol=1e-6)):
                raise ValueError("External probabilities must be normalized finite class vectors")
            if (indices[:len(previous)] != previous or len(set(indices)) != len(indices)
                    or any(j < 0 or j >= len(row["views"]) for j in indices)):
                raise ValueError("External acquired views must extend a valid simple-path prefix")
            if len(indices) > row["budget"]:
                raise ValueError("External baseline exceeded the common acquisition budget")
            charged = 0.0
            for position, index in enumerate(indices):
                view = row["views"][index]
                edge = view.get("from", {}).get(str(indices[position - 1])) if position else None
                if not view["reachable"] or edge is not None and not edge.get("reachable", True):
                    raise ValueError("External baseline used an unreachable graph action")
                charged += edge.get("cost", view["cost"]) if edge is not None else view["cost"]
            if not np.isclose(charged, state["cost"], atol=1e-6):
                raise ValueError("External baseline cost differs from the common graph charges")
            if not np.isfinite(state["cost"]) or state["cost"] < previous_cost:
                raise ValueError("Recorded cumulative path cost must be finite and nondecreasing")
            state["prediction"] = int(p.argmax())
            state["raw_risk"] = float(1 - p.max())
            state["eligible"] = bool(state["eligible"] and len(indices) >= row["minimum_views"])
            previous, previous_cost = indices, state["cost"]
    if any(row["classes"] != rows[0]["classes"] or row["alpha"] != rows[0]["alpha"] for row in rows):
        raise ValueError("Calibrate one class/task-risk configuration at a time")
    return rows


def calibrate(rows, policy_config, development_objects, output, delta=.05, grid_size=21):
    """Fit the same simultaneous episode-level bound to a frozen external model."""
    rows = validate(rows)
    objects = [object_id(row) for row in rows]
    development = set(map(str, development_objects))
    if len(set(objects)) != len(objects) or development & set(objects):
        raise ValueError("Calibration objects must be independent and disjoint from all development objects")
    artifact = dict(protocol="evidenceconserve.external.v1", policy=policy_config,
                    classes=rows[0]["classes"], alpha=rows[0]["alpha"],
                    calibration_objects=objects, development_objects=sorted(development),
                    fit=select_correction([row["trace"] for row in rows], [row["label"] for row in rows],
                                          rows[0]["alpha"], delta, .1, np.linspace(0, 1, grid_size)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(str(output))
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def evaluate(rows, artifact, policy_config, output):
    """Evaluate calibrated stops without discarding forced-budget episodes."""
    rows = validate(rows)
    if artifact.get("protocol") != "evidenceconserve.external.v1" or policy_config != artifact["policy"]:
        raise ValueError("The external model/checkpoint/policy differs from calibration")
    excluded = set(artifact["calibration_objects"] + artifact["development_objects"])
    records = []
    for row in rows:
        if object_id(row) in excluded:
            raise ValueError("External calibration/development and test objects overlap")
        if row["classes"] != artifact["classes"] or row["alpha"] != artifact["alpha"]:
            raise ValueError("External evaluation task settings differ from calibration")
        selected = None
        if artifact["fit"]["enabled"]:
            correction = artifact["fit"]["selected"]["correction"]
            selected = next((state for state in row["trace"] if state["eligible"] and state["raw_risk"] + correction <= row["alpha"]), None)
        state = selected if selected is not None else row["trace"][-1]
        result = dict(probability=state["probability"], label=row["label"], views=state["views"],
                      cost=state["cost"], gains=state.get("gains", []),
                      forced_stop=selected is None, calibrated_stop=selected is not None,
                      stop_kind="forced" if selected is None else "calibrated")
        measured = score(row, result)
        measured["latency_ms"] = row.get("latency_ms")
        records.append(dict(id=row["id"], family=object_id(row), cluster=object_id(row), method=policy_config["baseline"],
                            result=result, metrics=measured))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "records.jsonl", records)
    (output / "metrics.json").write_text(json.dumps(summarize(records), indent=2, allow_nan=False), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "evaluate"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy-config", required=True, help="JSON registration with baseline/checkpoint/configuration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-objects", help="JSON array of every fitting/configuration-selection object")
    parser.add_argument("--calibration")
    args = parser.parse_args()
    policy = json.loads(Path(args.policy_config).read_text(encoding="utf-8"))
    if policy.get("baseline") not in ("gennbv", "fisherrf", "mc_dropout", "set_transformer", "active_implicit") or not policy.get("checkpoint"):
        parser.error("Register the actual baseline name and frozen checkpoint identity")
    rows = read_jsonl(args.data)
    if args.command == "calibrate":
        if not args.development_objects:
            parser.error("--development-objects is required")
        calibrate(rows, policy, json.loads(Path(args.development_objects).read_text(encoding="utf-8")), args.output)
    else:
        if not args.calibration:
            parser.error("--calibration is required")
        evaluate(rows, json.loads(Path(args.calibration).read_text(encoding="utf-8")), policy, args.output)


if __name__ == "__main__":
    main()
