"""Generated graph/trace glue checks; original AIR CUDA execution is separate."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from evidenceconserve.air_osvp import charge_batch, trace
from evidenceconserve.cli import read_jsonl, write_jsonl
from evidenceconserve.external import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-fixture", required=True, help="An already verified verify_native.py output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    case = dict(start_view=0, budget=3, views=[
        dict(air_view_id=0, reachable=True, cost=1.),
        dict(air_view_id=1, reachable=True, cost=1., **{"from": {"0": {"cost": 4.}, "2": {"cost": 1.}}}),
        dict(air_view_id=2, reachable=True, cost=1., **{"from": {"0": {"cost": 1.}, "1": {"cost": .5}}})])
    plan = charge_batch(case, [1, 2])
    assert plan["route"] == [0, 2, 1] and plan["cost"] == 3.
    changed = copy.deepcopy(case); changed["label"] = 99
    assert charge_batch(changed, [1, 2]) == plan
    for invalid_case, ids in [(dict(case, budget=2), [1, 2]), (case, [31]), (case, [1, 1])]:
        try:
            charge_batch(invalid_case, ids)
        except ValueError:
            pass
        else:
            raise AssertionError("Unchargeable original batch was silently altered")
    from evidenceconserve.native import NativeBackend
    fixture = Path(args.native_fixture)
    backend = NativeBackend(fixture / "trained/native.pt")
    episode = read_jsonl(fixture / "episodes.jsonl")[-1]
    result = trace(episode, [0, 2], backend)
    assert backend.diagnostics["captured_images"] == [0, 2]
    evidence = sum(episode["views"][j]["quality"] * backend.cache[j][0].numpy() for j in [0, 2])
    np.testing.assert_allclose(result[-1]["probability"], (evidence + 1) / (evidence.sum() + 2), atol=1e-7)
    row = dict(episode, trace=result)
    validate([row]); write_jsonl(output / "traces.jsonl", [row])
    assert read_jsonl(output / "traces.jsonl")[0]["trace"] == row["trace"]
    report = dict(kind="generated_air_graph_trace_glue_only", hand_route=plan, original_cuda_inference="not_run",
                  captured_rgb=[0, 2], shared_encoder_trace_and_readback="passed")
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
