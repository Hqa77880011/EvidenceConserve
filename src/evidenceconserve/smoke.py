"""Small deterministic checks of conservation, calibration and file-based replay."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from .calibration import apply_calibration, calibrate, select_correction
from .cli import execute, main as cli_main, plot, read_jsonl, summarize, write_jsonl
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
    from .predictors import fit_predictors, load_predictors, predict_case
    from .interventions import duplicate_intervention, oracle, oracle_metrics
    rng = np.random.default_rng(13)
    features = rng.random((12, 2))
    groups = np.array(["fit-" + str(i) for i in range(12)])
    splits = np.array(["train"] * 8 + ["val"] * 4)
    cache = dict(candidate_features=features, evidence=np.c_[features, features[:, 0]],
                 candidate_group=groups, candidate_split=splits, quality=np.full(12, .9),
                 overlap_features=features, overlap=features[:, 0],
                 overlap_group=groups, overlap_split=splits)
    fitted = fit_predictors(cache, output / "predictors", seeds=(13,), epochs=2,
                            patience=2, hidden_sizes=(8,), alphas=(1e-4,))
    json.dumps(fitted["registration"], allow_nan=False)
    loaded = load_predictors(output / "predictors/predictors.joblib")
    learned_calibration = list(generate(random.Random(37), 4))
    learned_test = list(generate(random.Random(73), 2))
    for prefix, episodes in (("learncal:", learned_calibration), ("learntest:", learned_test)):
        for episode in episodes:
            episode.update(id=prefix + episode["id"], family=prefix + episode["family"], budget=3)
            episode["views"] = episode["views"][:3]
            for j, view in enumerate(episode["views"]):
                view["candidate_features"] = [j / 3, view["cost"] / 2]
                view["overlap_features"] = {str(k): [float(j == k), .9] for k in range(3) if k != j}
    np.testing.assert_allclose(predict_case(learned_test[0], fitted)["views"][0]["predicted_evidence"],
                               predict_case(learned_test[0], loaded)["views"][0]["predicted_evidence"], atol=1e-12)
    write_jsonl(output / "learncal.jsonl", learned_calibration)
    write_jsonl(output / "learntest.jsonl", learned_test)
    (output / "development.json").write_text("[]", encoding="utf-8")
    sys.argv = ["evidenceconserve", "calibrate", "--data", str(output / "learncal.jsonl"),
                "--output", str(output / "learned-calibration.json"), "--predictors", str(output / "predictors/predictors.joblib"),
                "--development-objects", str(output / "development.json"), "--methods", "evidenceconserve", "--grid-size", "3"]
    cli_main()
    sys.argv = ["evidenceconserve", "run", "--data", str(output / "learntest.jsonl"),
                "--output", str(output / "learned-run"), "--predictors", str(output / "predictors/predictors.joblib"),
                "--calibration", str(output / "learned-calibration.json"), "--oracle-reference", "--duplicate-intervention"]
    cli_main()
    learned_rows = read_jsonl(output / "learned-run/evidenceconserve/records.jsonl")
    assert all(row["metrics"]["duplicate_inserted"] == 1 for row in learned_rows)
    np.testing.assert_allclose([row["metrics"]["duplicate_increment"] for row in learned_rows], 0, atol=1e-12)
    small = dict(classes=2, label=0, budget=2, minimum_views=1, alpha=.2, value_threshold=.001,
                 views=[dict(evidence=[9, 0], visibility=[1, 0], quality=1, cost=2, reachable=True),
                        dict(evidence=[3, 0], visibility=[0, 1], quality=1, cost=1, reachable=True)])
    reference = oracle(small)
    assert reference["oracle_feasible"] and reference["views"] == [1] and reference["cost"] == 1
    regret = oracle_metrics(small, dict(probability=[10 / 11, 1 / 11], views=[0], cost=2), reference)
    np.testing.assert_allclose(regret["oracle_regret"], 1)
    execute([dict(small, id="oracle-hand")], "oracle", output / "oracle-hand")
    print(json.dumps(dict(checks="passed", methods=len(METHODS), episodes=4,
                          exact_duplicate_gain=float(duplicate.sum()), new_surface_gain=float(novel.sum()),
                          learned_predictors="train/save/load/calibrate/run passed", oracle_and_interventions="passed")))


if __name__ == "__main__":
    main()
