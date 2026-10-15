import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(values):
    valid = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read(args.records)
    reference = {row["id"]: row for row in read(args.reference)} if args.reference else None
    if reference is not None and set(reference) != {row["id"] for row in rows}:
        raise ValueError("Paired comparison requires identical case IDs")
    clusters = {}
    for row in rows:
        clusters.setdefault(row.get("cluster", row["id"]), []).append(row)
    groups = list(clusters.values())
    rng = np.random.default_rng(args.seed)
    report = {}
    keys = sorted({key for row in rows for key in row["metrics"]})
    for key in keys:
        values = []
        for group in groups:
            if reference is None:
                value = mean([row["metrics"].get(key) for row in group])
            else:
                differences = [
                    row["metrics"][key] - reference[row["id"]]["metrics"][key]
                    for row in group
                    if row["metrics"].get(key) is not None
                    and reference[row["id"]]["metrics"].get(key) is not None
                ]
                value = mean(differences)
            if value is not None:
                values.append(value)
        if not values:
            continue
        values = np.asarray(values)
        boot = np.array(
            [
                values[rng.integers(len(values), size=len(values))].mean()
                for _ in range(args.resamples)
            ]
        )
        item = {
            "mean": float(values.mean()),
            "clusters": len(values),
            "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
        if reference is None:
            binary = [
                row["metrics"].get(key) for row in rows if row["metrics"].get(key) is not None
            ]
            if binary and set(binary) <= {0, 1, False, True} and not sum(binary):
                item["zero_event_cluster_probability_upper95"] = float(
                    1 - 0.05 ** (1 / len(groups))
                )
        else:
            pairs = [
                (row["metrics"].get(key), reference[row["id"]]["metrics"].get(key)) for row in rows
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if (
                all(len(group) == 1 for group in groups)
                and pairs
                and {v for pair in pairs for v in pair} <= {0, 1, False, True}
            ):
                b = sum(a == 1 and c == 0 for a, c in pairs)
                c = sum(a == 0 and d == 1 for a, d in pairs)
                item["mcnemar_exact_p"] = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
        report[key] = item
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
