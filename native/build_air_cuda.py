"""Build original AIR CUDA dependencies in the supplied isolated Linux prefix."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

POINTNET_REF = "b5ceb6d9ca0467ea34beb81023f96ee82228f626"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pointnet2", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    workspace, pointnet, prefix = args.workspace.resolve(), args.pointnet2.resolve(), args.prefix.resolve()
    revision = subprocess.check_output(["git", "-C", str(pointnet), "rev-parse", "HEAD"], text=True).strip()
    if revision != POINTNET_REF or not torch.cuda.is_available():
        raise ValueError("Use the pinned Pointnet2 source and a real CUDA device")
    architecture = ".".join(map(str, torch.cuda.get_device_capability()))
    environment = dict(os.environ, CUDA_HOME=str(prefix), TORCH_CUDA_ARCH_LIST=architecture,
                       CC=str(prefix/"bin/x86_64-conda-linux-gnu-cc"),
                       CXX=str(prefix/"bin/x86_64-conda-linux-gnu-c++"),
                       MAX_JOBS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PATH=str(prefix/"bin")+os.pathsep+os.environ["PATH"])
    logs = workspace/"build-logs"
    logs.mkdir(exist_ok=True)
    transforms = workspace/"POCO/lightconvpoint/utils/transforms.py"
    transforms.write_text(transforms.read_text().replace("data.keys:", "data.keys():"))
    sampling = workspace/"POCO/lightconvpoint/spatial/sampling/sampling_quantized.py"
    sampling.write_text(sampling.read_text().replace(
        "torch.zeros(data.pos.shape[0])", "torch.zeros(data.pos.shape[0], device=data.pos.device, dtype=torch.long)"))

    def command(arguments, directory, name):
        print(name, flush=True)
        with (logs/name).open("w") as output:
            result = subprocess.run(arguments, cwd=directory, env=environment, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print((logs/name).read_text(errors="replace")[-8000:], flush=True)
            raise subprocess.CalledProcessError(result.returncode, arguments)

    setup = pointnet/"pointnet2_ops_lib/setup.py"
    setup.write_text(re.sub(r'(?m)^os.environ\["TORCH_CUDA_ARCH_LIST"\] = .+$',
                           '# Target architecture is supplied by the build environment.', setup.read_text()))
    command([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."],
            setup.parent, "pointnet2.log")
    command([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."],
            workspace/"pointrVP/extensions/chamfer_dist", "chamfer.log")
    # Restrict import-time registry population to the one actual baseline.
    # Other PoinTr models import gridding/EMD/KNN extensions that COSC never uses.
    # No COSC, POCO, transformer, pointnet or Chamfer algorithm source is edited.
    (workspace/"pointrVP/models/__init__.py").write_text(
        "from .build import build_model_from_cfg\nimport models.COSC\n")
    sys.path.insert(0, str(workspace/"pointrVP"))
    from pointnet2_ops import pointnet2_utils
    from extensions.chamfer_dist import ChamferDistanceL2
    from models.COSC import COSC
    points = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]], device="cuda")
    sampled = pointnet2_utils.furthest_point_sample(points, 2)
    assert sampled.tolist() == [[0, 3]], sampled
    left = points.detach().clone().requires_grad_(True)
    right = points+torch.tensor([10., 0., 0.], device="cuda")
    distance = ChamferDistanceL2()(left, right)
    expected = ((left[:, :, None]-right[:, None, :]).square().sum(-1).min(2).values.mean()
                +(left[:, :, None]-right[:, None, :]).square().sum(-1).min(1).values.mean())
    torch.testing.assert_close(distance, expected)
    distance.backward()
    assert left.grad.norm() > 0
    metadata = dict(pointnet_revision=revision, model_class=COSC.__name__, torch=torch.__version__, cuda=torch.version.cuda,
                    architecture=architecture, original_cuda_fps_indices=sampled.tolist(),
                    chamfer_distance=float(distance.detach()), chamfer_gradient_norm=float(left.grad.norm()),
                    source_changes=["Pointnet2 compiler architecture selection", "COSC-only model registry imports",
                                    "PyG 2.x Data.keys() API in POCO transforms",
                                    "Voxel-grid batch tensor on the input device"])
    (logs/"validation.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
