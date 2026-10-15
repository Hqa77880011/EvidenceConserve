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


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    elapsed = sum(row["metrics"]["latency_ms"] for row in rows) / 1000
    legal = sum(row["metrics"].get("legal_effects", 0) for row in rows)
    report["legal_throughput_per_second"] = legal / elapsed if elapsed else None
    return report


def execute(cases, method, output):
    rows = []
    process = psutil.Process()
    for case in cases:
        start_cpu = time.process_time()
        start = time.perf_counter()
        result = core.run_case(case, method)
        elapsed = time.perf_counter() - start
        cpu = time.process_time() - start_cpu
        metrics = core.score(case, result)
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
                "cluster": case.get("cluster", case["id"]),
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
    for name in ("run", "suite"):
        item = sub.add_parser(name)
        item.add_argument("--data", required=True)
        item.add_argument("--output", required=True)
        item.add_argument(
            "--methods",
            nargs="+",
            choices=core.METHODS,
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
        write_jsonl(args.output, rows)
    elif args.command in ("run", "suite"):
        cases = read_jsonl(args.data)
        for method in args.methods:
            execute(cases, method, Path(args.output) / method)
    else:
        plot(args.inputs, args.output)


if __name__ == "__main__":
    main()
