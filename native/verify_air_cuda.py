"""Exercise original POCO one-update completion and pretrained COSC on one public capture."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml


def report(workspace, assets, output):
    """Collect completed stages without repeating the adaptation update."""
    metadata = dict(protocol="public-single-capture-v1", source_points=len(np.loadtxt(assets/"cloud_0.txt")),
                    observed_view_ids=[0], future_captures_read=0, prior_box_meters=[-.1, .1],
                    query_points=len(np.loadtxt(workspace/"data/public_bbx.txt")), full_training=False,
                    poco=json.loads((output/"poco-validation.json").read_text()),
                    cosc=json.loads((output/"cosc-validation.json").read_text()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
    from evidenceconserve.air_osvp import charge_batch
    grid = np.loadtxt(workspace/"view_space_32.txt", skiprows=1)
    views = [dict(air_view_id=i, reachable=True, cost=1.,
                  **{"from": {str(j): dict(reachable=True, cost=1.+float(np.linalg.norm(position-other)))
                              for j, other in enumerate(grid) if j != i}})
             for i, position in enumerate(grid)]
    charged = {}
    for budget in (6, 7):
        try:
            route = charge_batch(dict(start_view=0, budget=budget, views=views), metadata["cosc"]["original_view_ids"])
        except ValueError as error:
            if "exceeds the common capture budget" not in str(error):
                raise
            charged[str(budget)] = dict(status="rejected", reason=str(error))
        else:
            charged[str(budget)] = dict(status="accepted", **route)
    metadata["graph_charge"] = dict(protocol="Declared complete 32-view graph, capture 1 + Euclidean travel cost",
                                     original_physical_graph=False, results=charged)
    (output/"validation.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "poco", "cosc", "report"], default="all")
    args = parser.parse_args()
    workspace, assets, output = args.workspace.resolve(), args.assets.resolve(), args.output.resolve()
    torch.set_num_threads(1)
    torch.manual_seed(47)
    np.random.seed(47)
    os.chdir(workspace)
    if args.stage == "report":
        report(workspace, assets, output)
        return
    if args.stage == "all":
        output.mkdir(parents=True, exist_ok=False)
        points = np.loadtxt(assets/"cloud_0.txt")
        camera = np.loadtxt(workspace/"view_space_32.txt", skiprows=1)[0]
        rng = np.random.default_rng(47)
        endpoints = points[rng.integers(len(points), size=1024)]
        free = camera+(endpoints-camera)*rng.uniform(.1, .8, (1024, 1))
        scene = np.r_[np.c_[endpoints, np.ones(1024)], np.c_[free, np.zeros(1024)]]
        # Declared 20 cm prior box, independent of unseen surfaces/future captures.
        grid = np.stack(np.meshgrid(*[np.linspace(-.1, .1, 48)]*3, indexing="ij"), -1).reshape(-1, 3)
        data = workspace/"data"
        data.mkdir(exist_ok=True)
        np.savetxt(data/"public_surface.txt", points)
        np.savetxt(data/"public_scene.txt", scene)
        np.savetxt(data/"public_bbx.txt", grid)
        state = np.zeros(32); state[0] = 1
        np.savetxt(data/"public_vs.txt", state)
        config = yaml.safe_load((workspace/"POCO/configs/config_cosc.yaml").read_text())
        config.update(manifold_points=2048, non_manifold_points=1024, threads=0)
        (output/"poco-config.yaml").write_text(yaml.safe_dump(config))
        shutil.copy2(assets/"poco.pth", workspace/"POCO/checkpoint.pth")
        shutil.copy2(assets/"cosc.pth", workspace/"pointrVP/trained_network/best_ckpt.pth")
        for stage in ("poco", "cosc"):
            print("Running original", stage, flush=True)
            with (output/(stage+".log")).open("w") as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                         "--workspace", str(workspace), "--assets", str(assets),
                                         "--output", str(output), "--stage", stage],
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((output/(stage+".log")).read_text(errors="replace")[-6000:])
                raise subprocess.CalledProcessError(result.returncode, stage)
        report(workspace, assets, output)
        return

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.stage == "poco":
        sys.path.insert(0, str(workspace/"POCO"))
        updates = []
        original_step = torch.optim.Adam.step

        def measured_step(optimizer, *values, **keywords):
            parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            before = [p.detach().clone() for p in parameters]
            norm = float(sum(p.grad.float().square().sum() for p in parameters).sqrt())
            result = original_step(optimizer, *values, **keywords)
            change = float(sum((p.detach()-old).abs().sum() for p, old in zip(parameters, before)))
            updates.append(dict(gradient_norm=norm, parameter_change_l1=change,
                                gradient_tensors=len(parameters), parameters=sum(p.numel() for p in parameters)))
            return result

        torch.optim.Adam.step = measured_step
        sys.argv = [str(workspace/"POCO/train_and_infer_one.py"), "--config", str(output/"poco-config.yaml"),
                    "--obj_name", "public", "--epoch", "1"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        completion = np.loadtxt(workspace/"data/public_pc.txt", ndmin=2)
        if len(updates) != 1 or updates[0]["gradient_norm"] <= 0 or updates[0]["parameter_change_l1"] <= 0:
            raise RuntimeError("The original POCO optimizer did not perform one real update")
        if len(completion) < 512 or completion.shape[1] != 3 or not np.isfinite(completion).all():
            raise RuntimeError("Original POCO completion cannot enter the original 512-centre COSC network")
        metadata = dict(original_script="POCO/train_and_infer_one.py", updates=updates,
                        completed_points=len(completion), source_cloud="Asian_Dragon/rotate_0/cloud_0.txt")
        shutil.copy2(workspace/"data/public_pc.txt", output/"completed_pc.txt")
    else:
        sys.path.insert(0, str(workspace/"pointrVP"))
        from models.COSC import COSC
        from pointnet2_ops import pointnet2_utils
        forwards, fps_calls = [], []
        original_forward, original_fps = COSC.forward, pointnet2_utils.furthest_point_sample

        def measured_forward(model, *values, **keywords):
            result = original_forward(model, *values, **keywords)
            forwards.append(dict(probabilities=result.detach().cpu().numpy().ravel().tolist(),
                                 parameters=sum(p.numel() for p in model.parameters())))
            return result

        def measured_fps(*values, **keywords):
            result = original_fps(*values, **keywords)
            fps_calls.append(list(result.shape))
            return result

        COSC.forward, pointnet2_utils.furthest_point_sample = measured_forward, measured_fps
        sys.argv = [str(workspace/"pointrVP/infer_once.py"), "--infer_file", str(workspace/"data/public_pc.txt")]
        result = runpy.run_path(sys.argv[0], run_name="__main__")
        probabilities = np.asarray(forwards[0]["probabilities"])
        predicted = np.flatnonzero(probabilities >= .5).tolist()
        if len(forwards) != 1 or not fps_calls or probabilities.shape != (32,) or not np.isfinite(probabilities).all():
            raise RuntimeError("The original pretrained COSC/native FPS path was not exercised")
        if predicted != result["pred_vs"]:
            raise RuntimeError("The author script's batch differs from its original sigmoid/.5 rule")
        metadata = dict(original_script="pointrVP/infer_once.py", **forwards[0], native_fps_calls=fps_calls,
                        original_threshold=.5, original_view_ids=predicted, batch_pruned=False)
    torch.cuda.synchronize()
    metadata.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    elapsed_seconds=time.perf_counter()-started,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/(args.stage+"-validation.json")).write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
