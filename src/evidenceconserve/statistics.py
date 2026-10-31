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


def aggregate(rows):
    """Recompute nonlinear metrics after resampling complete object episodes."""
    classes = len(rows[0]["result"]["probability"])
    confusion = np.zeros((classes, classes), dtype=int)
    correct = 0
    has_labels = all("label" in row["result"] for row in rows)
    for row in rows:
        if "label" not in row["result"]:
            correct += bool(row["metrics"]["accuracy"])
            continue
        label = row["result"]["label"]
        prediction = int(np.argmax(row["result"]["probability"]))
        confusion[label, prediction] += 1
        correct += prediction == label
    numerator = 2 * np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(numerator, denominator, out=np.zeros(classes), where=denominator > 0)
    duplicate = mean([row["metrics"].get("duplicate_gain") for row in rows])
    novel = mean([row["metrics"].get("new_surface_gain") for row in rows])
    return dict(macro_f1=float(f1.mean()) if has_labels else None,
                views_per_correct_decision=sum(len(row["result"]["views"]) for row in rows) / correct if correct else None,
                duplicate_new_gain_ratio=duplicate / novel if duplicate is not None and novel is not None and novel > 0 else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = read(args.records)
    if not rows or args.resamples < 1:
        parser.error("Require nonempty records and a positive --resamples")
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
    point = aggregate(rows)
    point_reference = aggregate([reference[row["id"]] for row in rows]) if reference is not None else None
    samples = {key: [] for key in point}
    for _ in range(args.resamples):
        selected = [row for index in rng.integers(len(groups), size=len(groups)) for row in groups[index]]
        measured = aggregate(selected)
        compared = aggregate([reference[row["id"]] for row in selected]) if reference is not None else None
        for key, value in measured.items():
            if value is not None and (compared is None or compared[key] is not None):
                samples[key].append(value if compared is None else value - compared[key])
    for key, values in samples.items():
        if values and point[key] is not None and (point_reference is None or point_reference[key] is not None):
            report[key] = dict(mean=point[key] if point_reference is None else point[key] - point_reference[key],
                               clusters=len(groups), bootstrap95=np.quantile(values, [.025, .975]).tolist(),
                               valid_resamples=len(values), aggregation="recomputed on complete object resamples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
