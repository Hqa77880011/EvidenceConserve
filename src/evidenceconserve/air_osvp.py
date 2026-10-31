"""Run original AIR-OSVP POCO/COSC sources and charge their one-shot batch.

No alternate completion network, pointnet/CUDA kernel, or predicted-view file
is substituted. Inference executes the original author scripts sequentially.
"""

import argparse
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .cli import read_jsonl, write_jsonl
from .frontend import object_id


SOURCES = {
    "AIR-OSVP": ("psc0628/AIR-OSVP", "6847c0a57ca3b4e9d9a82d4cbf12f08074f71a89"),
    "PoinTr": ("yuxumin/PoinTr", "4603257ed3db9e7dad349b712e1b2fe0da207015"),
    "POCO": ("valeoai/POCO", "7f85ace3f353cedb63d77e9aafaf6342af4710a3"),
}


def fetch(output):
    """Clone pinned original source trees and their declared submodules privately."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for name, (repository, revision) in SOURCES.items():
        target = output / name
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "https://github.com/" + repository + ".git", revision], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
        subprocess.run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"], check=True)
    (output / "sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")


def prepare(sources, output):
    """Apply the author's documented POCO/PoinTr overlays, retaining source licenses."""
    sources, output = Path(sources).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    air_revision = subprocess.run(["git", "-C", str(sources / "AIR-OSVP"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if air_revision != SOURCES["AIR-OSVP"][1]:
        raise ValueError("AIR author overlay revision differs from the registered pin")
    for name, destination in [("POCO", "POCO"), ("PoinTr", "pointrVP")]:
        subprocess.run(["git", "-C", str(sources / name), "diff", "--exit-code"], check=True, capture_output=True)
        revision = subprocess.run(["git", "-C", str(sources / name), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if revision != SOURCES[name][1]:
            raise ValueError("Native source revision differs from the registered pin: " + name)
        shutil.copytree(sources / name, output / destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    overlay = sources / "AIR-OSVP/poco_pointr"
    for name in ("POCO", "pointrVP"):
        shutil.copytree(overlay / name, output / name, dirs_exist_ok=True)
    for name in ("data", "run_time"):
        (output / name).mkdir(exist_ok=True)
    (output / "view_space_32.txt").write_bytes((sources / "AIR-OSVP/View_planning_simulator/view_space_32.txt").read_bytes())
    # Original dependency licenses remain within POCO/PoinTr. AIR has no root
    # license file in this pinned release; no author source is redistributed here.
    (output / "sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")


def preflight(workspace):
    """Reach the real compiled-operator imports before requiring any checkpoint."""
    workspace = Path(workspace).resolve()
    if not (workspace / "pointrVP/models/COSC.py").is_file():
        raise FileNotFoundError("Prepare the actual author COSC source under workspace/pointrVP")
    sys.path.insert(0, str(workspace / "pointrVP"))
    # These names and CUDAExtension contracts are directly imported by the
    # original COSC.py; an unavailable extension is an explicit runtime error.
    importlib.import_module("pointnet2_ops.pointnet2_utils")
    importlib.import_module("extensions.chamfer_dist")
    importlib.import_module("models.COSC")
    if not (workspace / "POCO/train_and_infer_one.py").is_file():
        raise FileNotFoundError("Prepare the original POCO fine-tuning/completion script")
    sys.path.insert(0, str(workspace / "POCO"))
    for name in ("torch_geometric", "lightconvpoint", "open3d"):
        importlib.import_module(name)
    return {"native_imports": "passed", "sources": SOURCES}


def charge_batch(case, predicted_ids):
    """Route the actual binary batch by exact common-graph cost, within its budget.

    COSC supplies an unordered 32-way thresholded set. This routing convention
    changes neither that set nor its members; an unchargeable batch is rejected.
    """
    start = int(case["start_view"])
    mapping = {int(view["air_view_id"]): i for i, view in enumerate(case["views"])}
    if len(mapping) != len(case["views"]) or any(not 0 <= j < 32 for j in mapping):
        raise ValueError("Each graph candidate needs one distinct original AIR view ID in [0,31]")
    if len(set(predicted_ids)) != len(predicted_ids) or any(int(j) not in mapping for j in predicted_ids):
        raise ValueError("Original AIR batch includes duplicate or off-graph views; cannot charge this episode")
    selected = sorted({mapping[int(j)] for j in predicted_ids} - {start})
    if len(selected) + 1 > case["budget"]:
        raise ValueError("Original one-shot batch exceeds the common capture budget; no pruning is substituted")
    if len(selected) > 8:
        raise ValueError("Exact batch routing supports at most eight additional captures")
    options = []
    for order in itertools.permutations(selected):
        route, cost, legal = [start, *order], 0., True
        for position, j in enumerate(route):
            view = case["views"][j]
            edge = view.get("from", {}).get(str(route[position - 1])) if position else None
            if not view["reachable"] or edge is not None and not edge.get("reachable", True):
                legal = False; break
            value = float(edge.get("cost", view["cost"]) if edge is not None else view["cost"])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Common graph cost must be finite and positive")
            cost += value
        if legal:
            options.append((cost, route))
    if not options:
        raise ValueError("No common-graph route reaches the unchanged original AIR batch")
    cost, route = min(options, key=lambda item: (item[0], item[1]))
    return {"route": route, "cost": cost, "original_view_ids": list(map(int, predicted_ids))}


def validate_grid(case, workspace):
    grid = np.loadtxt(Path(workspace) / "view_space_32.txt", skiprows=1)
    transform = np.asarray(case["air_to_world"], dtype=float)
    if grid.shape != (32, 3) or transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("Require original 32-view grid and a calibrated AIR-to-world 4x4 transform")
    for view in case["views"]:
        expected = (transform @ np.r_[grid[int(view["air_view_id"])], 1])[:3]
        if not np.allclose(expected, view["pose"][:3], rtol=0, atol=case.get("air_pose_tolerance", .001)):
            raise ValueError("AIR view-ID camera centre differs from the registered common graph")


def run_author(script, arguments, workspace, log):
    """Run one unmodified author script, with one torch/BLAS CPU thread."""
    launcher = "import runpy,sys,torch;torch.set_num_threads(1);sys.path.insert(0,sys.argv[1]);p=sys.argv[2];sys.argv=sys.argv[2:];runpy.run_path(p,run_name='__main__')"
    script = Path(script).resolve()
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MKL_THREADING_LAYER="SEQUENTIAL")
    with Path(log).open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-c", launcher, str(script.parent), str(script), *map(str, arguments)],
                       cwd=workspace, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def trace(case, route, backend):
    """Capture only the original batch route with the shared frozen RGB encoder."""
    backend.reset(case, "naive_sum")
    from .core import overlap
    evidence = np.zeros(case["classes"]); observations = {}; rows = []; cost = 0.; gains = []
    for position, j in enumerate(route):
        record = backend.capture(case, j, observations)
        evidence += record["quality"] * np.asarray(record["evidence"])
        omega = max((overlap(record["visibility"], old["visibility"]) for old in observations.values()), default=0.)
        reference = max((overlap(case["views"][j].get("reference_visibility", record["visibility"]),
                                 case["views"][k].get("reference_visibility", old["visibility"]))
                         for k, old in observations.items()), default=0.)
        gains.append(dict(gain=float(record["quality"] * np.sum(record["evidence"])),
                          duplicate=bool(case["views"][j].get("duplicate", False)), overlap=omega,
                          reference_overlap=reference,
                          reference_overlap_available=bool(case.get("visibility_is_reference", False) or "reference_visibility" in case["views"][j])))
        edge = case["views"][j].get("from", {}).get(str(route[position - 1])) if position else None
        cost += edge.get("cost", case["views"][j]["cost"]) if edge is not None else case["views"][j]["cost"]
        observations[j] = record
        probability = (evidence + 1) / (evidence.sum() + case["classes"])
        best = 0.
        for k in route[position + 1:]:
            proposed = backend.propose(case, k, route[:position + 1], probability)
            candidate = evidence + case["views"][k].get("predicted_quality", case["views"][k]["quality"]) * proposed
            following = (candidate + 1) / (candidate.sum() + case["classes"])
            previous_edge = case["views"][k].get("from", {}).get(str(j))
            if previous_edge is not None and not previous_edge.get("reachable", True):
                continue
            expense = previous_edge.get("cost", case["views"][k]["cost"]) if previous_edge is not None else case["views"][k]["cost"]
            best = max(best, float(following.max() - probability.max()) / expense)
        rows.append(dict(views=route[:position + 1], probability=probability.tolist(), cost=cost,
                         gains=list(gains),
                         eligible=position + 1 >= case["minimum_views"] and best <= case["value_threshold"]))
    return rows


def infer(data, workspace, checkpoint, poco_checkpoint, encoder_checkpoint, output, poco_epochs=10):
    """Original POCO completion -> original COSC inference -> graph route -> RGB trace."""
    preflight(workspace)  # Compiler/extension failure precedes checkpoint checks.
    from .native import NativeBackend
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Supply the public original COSC checkpoint or one trained by the author script")
    if checkpoint != workspace / "pointrVP/trained_network/best_ckpt.pth":
        shutil.copy2(checkpoint, workspace / "pointrVP/trained_network/best_ckpt.pth")
    if poco_epochs < 1:
        raise ValueError("Original POCO's fine-tuning loop requires at least one epoch")
    if Path(poco_checkpoint).resolve() != workspace / "POCO/checkpoint.pth":
        shutil.copy2(Path(poco_checkpoint).resolve(), workspace / "POCO/checkpoint.pth")
    backend = NativeBackend(encoder_checkpoint)
    cases = read_jsonl(data); rows = []
    from .calibration import policy as registered_policy
    import yaml
    native_config = yaml.safe_load((workspace / "pointrVP/trained_network/config.yaml").read_text(encoding="utf-8"))
    if not cases or any(registered_policy(case) != registered_policy(cases[0]) for case in cases):
        raise ValueError("AIR export needs one frozen common task/budget/stopping configuration")
    if native_config["threshold_gamma"] != .5:
        raise ValueError("This entry registers the released COSC threshold .5")
    for index, case in enumerate(cases):
        started = time.perf_counter()
        if object_id(case) in backend.training_objects:
            raise ValueError("Shared encoder fitting/validation object used for AIR evaluation")
        validate_grid(case, workspace)
        name = f"episode{index:06d}"
        start = int(case["start_view"])
        supplied = case["views"][start]["air_inputs"]
        if supplied.get("observed_view_indices") != [start]:
            raise ValueError("One-shot completion inputs must derive only from the common initial capture")
        for suffix, columns in [("surface", 3), ("scene", 4), ("bbx", 3)]:
            path = (Path(data).resolve().parent / supplied[suffix]).resolve()
            value = np.loadtxt(path, ndmin=2)
            if value.shape[1] != columns or not np.isfinite(value).all():
                raise ValueError("Original POCO input needs finite " + suffix + f"[N,{columns}]")
            if suffix == "scene" and not np.isin(value[:, 3], [0, 1]).all():
                raise ValueError("Original scene occupancy labels must be binary observed/free-space supervision")
            shutil.copy2(path, workspace / "data" / f"{name}_{suffix}.txt")
        state = np.zeros(32, dtype=int); state[int(case["views"][start]["air_view_id"])] = 1
        np.savetxt(workspace / "data" / f"{name}_vs.txt", state, fmt="%d")
        run_author(workspace / "POCO/train_and_infer_one.py", ["--config", "./POCO/configs/config_cosc.yaml", "--epoch", poco_epochs, "--obj_name", name], workspace, output / f"{name}-poco.log")
        run_author(workspace / "pointrVP/infer_once.py", ["--experiment_path", "./pointrVP/trained_network", "--infer_file", f"./data/{name}_pc.txt"], workspace, output / f"{name}-cosc.log")
        selected = np.loadtxt(workspace / "data" / f"{name}_ids.txt", dtype=int, ndmin=1).ravel().tolist()
        plan = charge_batch(case, selected)
        row = dict(case, trace=trace(case, plan["route"], backend), air_batch=plan,
                   latency_ms=(time.perf_counter() - started) * 1000)
        rows.append(row)
    write_jsonl(output / "traces.jsonl", rows)
    policy = dict(baseline="active_implicit", checkpoint=str(checkpoint), sources=SOURCES,
                  poco_checkpoint=str(Path(poco_checkpoint).resolve()), native_config=native_config,
                  task=registered_policy(cases[0]),
                  poco_epochs=poco_epochs, batch_threshold=.5, routing="minimum charged common-graph permutation; unchanged batch",
                  shared_encoder=backend.config)
    (output / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch"); get.add_argument("--output", required=True)
    stage = commands.add_parser("prepare"); stage.add_argument("--sources", required=True); stage.add_argument("--output", required=True)
    check = commands.add_parser("preflight"); check.add_argument("--workspace", required=True)
    run = commands.add_parser("infer")
    for key in ("data", "workspace", "checkpoint", "poco-checkpoint", "encoder-checkpoint", "output"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--poco-epochs", type=int, default=10)
    args = vars(parser.parse_args()); command = args.pop("command")
    result = {"fetch": fetch, "prepare": prepare, "preflight": preflight, "infer": infer}[command](**args)
    if command == "preflight":
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
