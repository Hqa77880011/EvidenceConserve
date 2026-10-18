"""Frozen numerical evidence model: independent calibration/test objects, three classes, eight candidate views, six-view budget; fixed 21-point CP grid."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "evidenceconserve", "prepare", "--count", "2000", "--seed", "17", "--output", str(output / "calibration.jsonl")],
        ["-m", "evidenceconserve", "prepare", "--count", "400", "--seed", "29", "--output", str(output / "test.jsonl")],
        ["-m", "evidenceconserve", "calibrate", "--data", str(output / "calibration.jsonl"), "--development-objects", "configs/development_objects.json", "--grid-size", "21", "--delta", "0.05", "--minimum-coverage", "0.1", "--output", str(output / "calibration.json")],
        ["-m", "evidenceconserve", "suite", "--data", str(output / "test.jsonl"), "--calibration", str(output / "calibration.json"), "--output", str(output / "comparison")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='runs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")

    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
