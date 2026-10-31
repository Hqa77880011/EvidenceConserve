import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil

from . import core
from .calibration import apply_calibration, calibrate
from .frontend import load_frontend, object_id
from sklearn.metrics import f1_score


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        for view in row.get("views", []):
            if "image" in view:
                view["image"] = str((Path(path).resolve().parent / view["image"]).resolve())
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def summarize(rows):
    report = {}
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row["metrics"].items()
            if value is None or isinstance(value, (int, float, bool))
        }
    )
    for key in keys:
        values = np.array(
            [float(row["metrics"][key]) for row in rows if row["metrics"].get(key) is not None],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if not len(values):
            report[key] = {"count": 0, "mean": None}
            continue
        report[key] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "p50": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }
    elapsed = sum(row["metrics"].get("latency_ms") or 0 for row in rows) / 1000
    legal = sum(row["metrics"].get("legal_effects", 0) for row in rows)
    report["legal_throughput_per_second"] = legal / elapsed if elapsed else None
    if rows:
        probabilities = np.asarray([row["result"]["probability"] for row in rows])
        labels = np.asarray([row["result"].get("label", -1) for row in rows])
        prediction = probabilities.argmax(1)
        report["macro_f1"] = float(f1_score(labels, prediction, labels=np.arange(probabilities.shape[1]), average="macro", zero_division=0)) if (labels >= 0).all() else None
        correct = np.asarray([bool(row["metrics"]["accuracy"]) for row in rows])
        report["views_per_correct_decision"] = sum(row["metrics"]["view_count"] for row in rows) / int(correct.sum()) if correct.any() else None
        order = np.argsort(1 - probabilities.max(1), kind="stable")
        report["risk_coverage"] = [dict(coverage=(i + 1) / len(rows), risk=float((~correct[order[:i + 1]]).mean())) for i in range(len(rows))]
        duplicate = report.get("duplicate_gain", {}).get("mean")
        new = report.get("new_surface_gain", {}).get("mean")
        report["duplicate_new_gain_ratio"] = duplicate / new if duplicate is not None and new is not None and new > 0 else None
    return report


def execute(cases, method, output, frontend=None, oracle_references=None, intervention=False):
    from .interventions import duplicate_intervention, oracle, oracle_metrics
    rows = []
    process = psutil.Process()
    for case in cases:
        start_cpu = time.process_time()
        start = time.perf_counter()
        result = oracle(case) if method == "oracle" else core.run_case(case, method, frontend=frontend)
        elapsed = time.perf_counter() - start
        cpu = time.process_time() - start_cpu
        metrics = core.score(case, result)
        if oracle_references is not None:
            metrics.update(oracle_metrics(case, result, oracle_references[str(case["id"])]))
        if intervention and method != "oracle":
            metrics.update(duplicate_intervention(case, method, result, frontend))
        metrics.update(
            latency_ms=elapsed * 1000,
            cpu_seconds=cpu,
            rss_mib=process.memory_info().rss / 2**20,
            output_bytes=len(json.dumps(result).encode()),
        )
        rows.append(
            {
                "id": case["id"],
                "family": case.get("family", case["id"]),
                "method": method,
                "cluster": object_id(case),
                "result": result,
                "metrics": metrics,
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "records.jsonl", rows)
    (output / "metrics.json").write_text(
        json.dumps(summarize(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    return rows


def plot(paths, output):
    records = [row for path in paths for row in read_jsonl(Path(path) / "records.jsonl")]
    methods = sorted({row["method"] for row in records})
    keys = sorted({key for row in records for key in row["metrics"]})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "metric", "n", "mean", "std", "p50", "p95", "p99"])
        for method in methods:
            summary = summarize([r for r in records if r["method"] == method])
            for key, value in summary.items():
                if isinstance(value, dict):
                    writer.writerow(
                        [
                            method,
                            key,
                            value["count"],
                            value["mean"],
                            value.get("std"),
                            value.get("p50"),
                            value.get("p95"),
                            value.get("p99"),
                        ]
                    )
    for key in keys:
        groups = [
            [
                r["metrics"][key]
                for r in records
                if r["method"] == method and r["metrics"].get(key) is not None
            ]
            for method in methods
        ]
        fig, ax = plt.subplots(figsize=(max(6, len(methods)), 4))
        means = [np.mean(group) if group else np.nan for group in groups]
        ax.bar(methods, means, color="#527c91")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output / (key + ".png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--count", type=int, default=200)
    prepare.add_argument("--seed", type=int, default=17)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--data", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--methods", nargs="+", choices=core.METHODS, default=core.METHODS[:-1])
    calibration.add_argument("--delta", type=float, default=0.05)
    calibration.add_argument("--minimum-coverage", type=float, default=0.1)
    calibration.add_argument("--grid-size", type=int, default=41)
    calibration.add_argument("--frontend-config")
    calibration.add_argument("--predictors", help="Frozen cached-feature predictor artifact")
    calibration.add_argument("--development-objects", required=True,
                             help="JSON list of every object used for fitting or configuration selection; [] only for a data-independent fixed policy")
    for name in ("run", "suite"):
        item = sub.add_parser(name)
        item.add_argument("--data", required=True)
        item.add_argument("--output", required=True)
        item.add_argument("--calibration")
        item.add_argument("--frontend-config")
        item.add_argument("--predictors", help="Must match the frozen predictors used for calibration")
        item.add_argument("--oracle-reference", action="store_true", help="Exact offline graph search, exponential in candidate count")
        item.add_argument("--duplicate-intervention", action="store_true", help="Paired exact replay after first capture, fixed budget and correction")
        item.add_argument(
            "--methods",
            nargs="+",
            choices=core.METHODS + ["oracle"],
            default=[core.METHODS[0]] if name == "run" else core.METHODS,
        )
    item = sub.add_parser("plot")
    item.add_argument("--inputs", nargs="+", required=True)
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.count < 1:
            parser.error("--count must be positive")
        rows = list(core.generate(random.Random(args.seed), args.count))
        for row in rows:
            row["id"] = str(args.seed) + ":" + row["id"]
            row["family"] = str(args.seed) + ":" + row["family"]
        write_jsonl(args.output, rows)
    elif args.command == "calibrate":
        if args.grid_size < 2:
            parser.error("--grid-size must be at least two")
        cases = read_jsonl(args.data)
        development = json.loads(Path(args.development_objects).read_text(encoding="utf-8"))
        predictor = None
        if args.predictors:
            from .predictors import load_predictors, predict_case
            predictor = load_predictors(args.predictors)
            cases = [predict_case(case, predictor) for case in cases]
            development = list(set(map(str, development)) | set(predictor["development_objects"]))
        artifact = calibrate(cases, args.methods, args.delta,
                             args.minimum_coverage, np.linspace(0, 1, args.grid_size),
                             load_frontend(args.frontend_config),
                             development)
        artifact["predictor_registration"] = None if predictor is None else predictor["registration"]
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(str(target))
        target.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    elif args.command in ("run", "suite"):
        cases = read_jsonl(args.data)
        predictor = None
        if args.predictors:
            from .predictors import load_predictors, predict_case
            predictor = load_predictors(args.predictors)
            if {object_id(case) for case in cases} & set(predictor["development_objects"]):
                raise ValueError("Predictor fitting/validation and test objects overlap")
            cases = [predict_case(case, predictor) for case in cases]
        frontend = load_frontend(args.frontend_config)
        if frontend and {object_id(case) for case in cases} & frontend.training_objects:
            raise ValueError("Frontend training and test objects overlap")
        artifact = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
        registration = None if predictor is None else predictor["registration"]
        if artifact is not None and artifact.get("predictor_registration") != registration:
            raise ValueError("Frozen predictors differ from the calibrated policy")
        if any(method not in ("no_calibration", "oracle") for method in args.methods) and artifact is None:
            parser.error("Calibrated methods require --calibration from independent object episodes")
        references = None
        if args.oracle_reference:
            from .interventions import oracle
            references = {str(case["id"]): oracle(case) for case in cases}
        for method in args.methods:
            selected = cases if method in ("no_calibration", "oracle") else apply_calibration(cases, method, artifact, frontend)
            execute(selected, method, Path(args.output) / method, frontend, references, args.duplicate_intervention)
    else:
        plot(args.inputs, args.output)


if __name__ == "__main__":
    main()
