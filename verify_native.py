"""Small native image/transport/PPO/RF checks; generated data are not paper results."""

import argparse
import copy
import json
import io
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent / "src"))
from evidenceconserve import core
from evidenceconserve.calibration import apply_calibration, calibrate
from evidenceconserve.cli import execute, read_jsonl, write_jsonl
from evidenceconserve.native import NativeBackend
from evidenceconserve.native_models import fisher_diagonal, fisher_score, render_rays
from evidenceconserve.native_training import make_models, train, read_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rng = np.random.default_rng(13)
    cases = []
    for object_index, split in enumerate(("train", "train", "policy_validation", "calibration", "test")):
        views = []
        for j in range(4):
            image = rng.random((64, 64, 3), dtype=np.float32) * .15
            image[10:54, 12:52, object_index % 2] += .65
            image[14 + j:17 + j, :, :] += .1
            image_path = output / f"object-{object_index}-view-{j}.png"
            Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).save(image_path)
            occupancy = ((np.arange(64) + 4 * j) % 64 < 40).reshape(4, 4, 4).astype(np.float32)
            voxel_path = output / f"object-{object_index}-view-{j}.npy"
            np.save(voxel_path, occupancy)
            angle = j * np.pi / 3
            center = np.array([2 * np.cos(angle), 2 * np.sin(angle), .3])
            forward = -center / np.linalg.norm(center)
            right = np.cross(forward, [0, 0, 1]); right /= np.linalg.norm(right)
            rotation = np.column_stack([right, np.cross(forward, right), forward])
            pose = np.r_[center, Rotation.from_matrix(rotation).as_euler("xyz")].tolist()
            views.append(dict(image=str(image_path), image_shape=[64, 64], intrinsics=[[50, 0, 32], [0, 50, 32], [0, 0, 1]],
                              pose=pose, voxel_path=str(voxel_path), visibility=occupancy.ravel().tolist(),
                              quality=.9, cost=1 + .1 * j, reachable=True, capture_seconds=.01))
        cases.append(dict(id=f"native-{object_index}", object_id=f"native-{object_index}", split=split,
                          classes=2, label=object_index % 2, views=views, voxel_shape=[4, 4, 4],
                          budget=3, minimum_views=2, alpha=.2, value_threshold=.001, start_view=0,
                          visibility_is_reference=True))
    write_jsonl(output / "episodes.jsonl", cases)
    leaking = copy.deepcopy(cases)
    leaking[-1]["object_id"] = leaking[0]["object_id"]
    write_jsonl(output / "split-overlap.jsonl", leaking)
    try:
        read_cases(output / "split-overlap.jsonl")
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("Native object split overlap was accepted")
    cfg = dict(seed=13, epochs=1, patience=1, batch_size=2, image_size=64, feature_dim=16, width=16,
               max_views=3, ppo_updates=1, ppo_epochs=1, rf_online_steps=1, rf_rays_side=1,
               rf_samples=4, mc_samples=4, learning_rate=.001)
    training_reads = []
    image_open = Image.open
    with patch("PIL.Image.open", side_effect=lambda path, *a, **kw: (training_reads.append(str(path)) or image_open(path, *a, **kw))):
        artifact, history = train(output / "episodes.jsonl", output / "trained", cfg)
    assert set(training_reads) == {view["image"] for case in cases[:3] for view in case["views"]}
    torch.manual_seed(13)
    initial = make_models(artifact["config"])
    changed = {}
    for name in ("encoder", "transport", "set_transformer", "feature_concatenation", "dynamics", "gennbv"):
        delta = sum(float((artifact["states"][name][key] - value).abs().sum())
                    for key, value in initial[name].named_parameters())
        assert delta > 0, name
        assert max(row["maximum_gradient_norm"] for row in history[name]) > 0, name
        changed[name] = delta
    del initial
    backend = NativeBackend(output / "trained/native.pt")
    with torch.no_grad():
        features = torch.randn(1, 3, 16)
        a = backend.models["set_transformer"](features)
        b = backend.models["set_transformer"](features[:, [2, 0, 1]])
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
        pose = torch.zeros(1, 6)
        torch.testing.assert_close(backend.models["transport"](pose, pose)[0], torch.eye(2), rtol=0, atol=0)
    methods = ["evidenceconserve", *core.NATIVE_METHODS]
    calibration = calibrate([cases[3]], methods, .05, .1, [0, .5, 1], backend, [])
    (output / "calibration.json").write_text(json.dumps(calibration, allow_nan=False), encoding="utf-8")
    calibration = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    open_image = Image.open
    for method in methods:
        selected = apply_calibration([cases[4]], method, calibration, backend)
        paths = []
        with patch("PIL.Image.open", side_effect=lambda path, *a, **kw: (paths.append(str(path)) or open_image(path, *a, **kw))):
            rows = execute(selected, method, output / method, backend)
        result = rows[0]["result"]
        assert result["native_diagnostics"]["captured_images"] == result["views"]
        expected = {cases[4]["views"][j]["image"] for j in result["views"]}
        assert set(paths) == expected
        assert rows == read_jsonl(output / method / "records.jsonl")
        changed_labels = copy.deepcopy(selected[0]); changed_labels["label"] = 1 - changed_labels["label"]
        other = core.run_case(changed_labels, method, backend)
        assert other["views"] == result["views"]
        np.testing.assert_allclose(other["probability"], result["probability"], atol=1e-7)
        if method == "fisherrf":
            assert min(result["native_diagnostics"]["online_rf_gradient_norms"]) > 0
            torch.save(backend.models["radiance"].state_dict(), output / "rf-online.pt")
            field = backend.models["radiance"]
            with torch.no_grad():
                before_rf = render_rays(field, torch.zeros(1, 3), torch.ones(1, 3))
            with torch.no_grad():
                next(field.parameters()).add_(.25)
            field.load_state_dict(torch.load(output / "rf-online.pt", weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(before_rf, render_rays(field, torch.zeros(1, 3), torch.ones(1, 3)), rtol=0, atol=1e-7)
    saved = NativeBackend(output / "trained/native.pt")
    test_image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(saved.models["encoder"](test_image)[0], backend.models["encoder"](test_image)[0], rtol=0, atol=1e-7)
    sample_features = torch.rand(1, 3, 16)
    model_inputs = dict(encoder=(test_image,), transport=(torch.ones(1, 6), torch.zeros(1, 6)),
                        set_transformer=(sample_features,), feature_concatenation=(sample_features,),
                        dynamics=(torch.rand(1, 6 + 9 + 2 + 16 + 1),),
                        gennbv=(torch.ones(1, 1, 4, 4, 4), torch.ones(1, 2, 6), torch.rand(1, 3, 6)))
    for name, inputs in model_inputs.items():
        model = saved.models[name].eval()
        with torch.no_grad():
            before = model(*inputs)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        with torch.no_grad():
            next(model.parameters()).add_(.25)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            after = model(*inputs)
        if isinstance(before, tuple):
            for first, second in zip(before, after):
                torch.testing.assert_close(first, second, rtol=0, atol=1e-7)
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=1e-7)

    class ConstantDensity(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.color = torch.nn.Parameter(torch.tensor(.2))
        def forward(self, points, directions):
            return torch.ones_like(points[..., :1]), self.color.expand_as(points)

    diagonal = fisher_diagonal(ConstantDensity(), torch.zeros(2, 3), torch.ones(2, 3), near=.1, far=1.1, samples=4)
    np.testing.assert_allclose(diagonal.numpy(), [6 * (1 - np.exp(-1)) ** 2], atol=1e-6)
    assert float(fisher_score(torch.tensor([2., 4.]), torch.tensor([1., 3.]), 1.)) == 2
    report = dict(kind="generated_native_functional_check", learned_parameter_l1_changes=changed,
                  methods=methods, unseen_image_reads=0, transport_identity=True,
                  set_permutation_invariance=True, fisher_hand_diagonal=diagonal.tolist(), checkpoint_atol=1e-7)
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
