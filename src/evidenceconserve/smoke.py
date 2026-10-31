"""Small deterministic checks of conservation, calibration and file-based replay."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .calibration import apply_calibration, calibrate, select_correction
from .cli import execute, plot, read_jsonl, summarize, write_jsonl
from .core import METHODS, generate, increment, run_case, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    case = next(generate(random.Random(17), 1))
    record = case["views"][0]
    duplicate, omega = increment(record["evidence"], record["visibility"], [0], case, "evidenceconserve", 1)
    np.testing.assert_allclose(duplicate, 0, atol=1e-12)
    assert omega == 1
    novel, omega = increment([1, 2, 3], 1 - np.asarray(record["visibility"]), [0], case, "evidenceconserve", 4)
    np.testing.assert_allclose(novel, [1, 2, 3])
    assert omega == 0
    traces = [[dict(eligible=True, raw_risk=0.01, prediction=0)] for _ in range(100)]
    fit = select_correction(traces, [0] * 100, .1, .05, .5, [0, .5])
    assert fit["enabled"] and fit["selected"]["errors"] == 0
    calibration = list(generate(random.Random(37), 24))
    for row in calibration:
        row["id"], row["family"] = "cal:" + row["id"], "cal:" + row["family"]
    artifact = calibrate(calibration, METHODS[:-1], .05, .1, [0, .5, 1], development_objects=[])
    (output / "calibration.json").write_text(json.dumps(artifact), encoding="utf-8")
    restored = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    cases = list(generate(random.Random(73), 4))
    write_jsonl(output / "test.jsonl", cases)
    cases = read_jsonl(output / "test.jsonl")
    for method in METHODS:
        selected = cases if method == "no_calibration" else apply_calibration(cases, method, restored)
        rows = execute(selected, method, output / method)
        np.testing.assert_allclose(rows[0]["result"]["probability"], run_case(selected[0], method)["probability"], atol=1e-12)
    known = dict(probability=[.75, .25], label=0, views=[0, 1], cost=2,
                 forced_stop=False, calibrated_stop=True, stop_kind="calibrated", gains=[])
    metric = score(dict(label=0, views=[{}, {}]), known)
    np.testing.assert_allclose(metric["brier"], .125)
    np.testing.assert_allclose(metric["nll"], -np.log(.75))
    report = summarize([dict(result=known, metrics={**metric, "latency_ms": 1})])
    assert report["macro_f1"] == .5 and report["views_per_correct_decision"] == 2
    plot([output / "evidenceconserve"], output / "figures")
    print(json.dumps(dict(checks="passed", methods=len(METHODS), episodes=4,
                          exact_duplicate_gain=float(duplicate.sum()), new_surface_gain=float(novel.sum()))))


if __name__ == "__main__":
    main()
