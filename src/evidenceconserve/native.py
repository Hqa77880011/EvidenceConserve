"""Native learned baselines with capture-time RGB and legal-graph acquisition."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F

from .native_models import fisher_diagonal, fisher_score, render_rays
from .native_training import (dynamics_input, image_tensor, make_models, pose_tensor, train, voxels)


def camera_rays(view, side):
    """Generate known candidate rays without opening its uncaptured image."""
    height, width = view["image_shape"]
    intrinsic = np.asarray(view["intrinsics"], dtype=float)
    rows = np.linspace(.1 * (height - 1), .9 * (height - 1), side)
    columns = np.linspace(.1 * (width - 1), .9 * (width - 1), side)
    x, y = np.meshgrid(columns, rows)
    pixel = np.stack([x.ravel(), y.ravel(), np.ones(x.size)], -1)
    pose = pose_tensor(view).numpy()
    rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    direction = (rotation @ np.linalg.solve(intrinsic, pixel.T)).T
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origin = np.repeat(pose[None, :3], len(direction), 0)
    return torch.as_tensor(origin, dtype=torch.float32), torch.as_tensor(direction, dtype=torch.float32), (x, y)


class NativeBackend:
    """Frozen visual/policy bundle; FisherRF alone fits observed RGB online.

    propose/rank inspect geometry and acquired history only. Image and voxel
    files are read exclusively by capture after core chooses a legal action.
    """

    def __init__(self, checkpoint):
        torch.set_num_threads(1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol") != "evidenceconserve.native.v1":
            raise ValueError("Unsupported native checkpoint")
        self.cfg = payload["config"]
        self.models = make_models(self.cfg)
        for name, model in self.models.items():
            model.load_state_dict(payload["states"][name])
            model.eval()
        self.rf_initial = payload["states"]["radiance"]
        self.mean, self.std = payload["mean"], payload["std"]
        registration = payload["registration"]
        self.config = dict(kind="native", registration=registration)
        self.training_objects = set(registration["train_objects"] + registration["validation_objects"])

    def reset(self, case, method):
        self.case, self.method = case, method
        self.cache, self.proposals, self.captured_voxels, self.ray_records = {}, {}, {}, {}
        self.diagnostics = dict(captured_images=[], online_rf_losses=[], online_rf_gradient_norms=[])
        if case["classes"] != self.cfg["classes"] or case["budget"] > self.cfg["max_views"]:
            raise ValueError("Native checkpoint class/view-budget registration differs from the episode")
        self.models["radiance"].load_state_dict(self.rf_initial)
        self.rf_optimizer = torch.optim.Adam(self.models["radiance"].parameters(), lr=self.cfg["learning_rate"])

    def propose(self, case, index, acquired, probability):
        values = dynamics_input(case, index, acquired, [self.cache[j][1] for j in acquired],
                                [self.cache[j][0] for j in acquired], self.cfg)
        with torch.no_grad():
            evidence, feature = self.models["dynamics"](values[None])
        self.proposals[index] = (evidence[0], feature[0], values)
        return evidence[0].numpy()

    def aggregate(self, acquired, candidate=None):
        features = [self.cache[j][1] for j in acquired]
        if candidate is not None:
            features.append(self.proposals[candidate][1])
        if not features:
            return np.zeros(self.cfg["classes"])
        with torch.no_grad():
            return self.models[self.method](torch.stack(features)[None])[0].numpy()

    def matrices(self, target, sources):
        with torch.no_grad():
            return {str(j): self.models["transport"](pose_tensor(self.case["views"][target])[None],
                                                      pose_tensor(self.case["views"][j])[None])[0].tolist() for j in sources}

    def rank(self, index, acquired, evidence, default_value, cost):
        if self.method == "mc_dropout":
            network = self.models["dynamics"]
            network.train()
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(self.cfg["seed"] + 101 * len(acquired) + index)
                values = self.proposals[index][2][None].expand(self.cfg["mc_samples"], -1)
                predicted, _ = network(values)
                total = predicted + torch.as_tensor(evidence)[None] + 1
                probability = total / total.sum(-1, keepdim=True)
                mean = probability.mean(0)
                information = -(mean * mean.clamp_min(1e-12).log()).sum() + (probability * probability.clamp_min(1e-12).log()).sum(-1).mean()
            network.eval()
            return max(0., float(information)) / (cost + 1e-8)
        if self.method == "gennbv":
            with torch.no_grad():
                logits, _ = self.models["gennbv"](*self._policy_state(acquired))
                return float(logits[0, index])
        if self.method == "fisherrf":
            with torch.enable_grad():
                field = self.models["radiance"]
                origins, directions, _ = camera_rays(self.case["views"][index], self.cfg["rf_rays_side"])
                candidate = fisher_diagonal(field, origins, directions, **self._render_options())
                prior = torch.zeros_like(candidate)
                for previous in acquired:
                    a, b, _ = self.ray_records[previous]
                    prior += fisher_diagonal(field, a, b, **self._render_options())
                return float(fisher_score(candidate, prior, self.cfg["rf_regularization"])) / (cost + 1e-8)
        return default_value

    def _policy_state(self, acquired):
        shape = tuple(self.case["voxel_shape"])
        voxel = torch.zeros(shape)
        for j in acquired:
            voxel = torch.maximum(voxel, self.captured_voxels[j])
        history = torch.stack([pose_tensor(self.case["views"][j]) for j in acquired]) if acquired else torch.zeros(1, 6)
        poses = torch.stack([pose_tensor(view) for view in self.case["views"]])
        return voxel[None, None], history[None], poses[None]

    def _render_options(self):
        return dict(near=self.cfg["rf_near"], far=self.cfg["rf_far"], samples=self.cfg["rf_samples"])

    def capture(self, case, index, history):
        view = case["views"][index]
        image = (image_tensor(view["image"], self.cfg["image_size"]) - self.mean[:, None, None]) / self.std[:, None, None]
        with torch.no_grad():
            evidence, feature = self.models["encoder"](image[None])
            matrices = self.matrices(index, history)
        self.cache[index] = (evidence[0], feature[0])
        self.diagnostics["captured_images"].append(index)
        if self.method == "gennbv":
            self.captured_voxels[index] = voxels(view)
        if self.method == "fisherrf":
            origins, directions, (x, y) = camera_rays(view, self.cfg["rf_rays_side"])
            with Image.open(view["image"]) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
            target = torch.from_numpy(rgb[np.rint(y).astype(int).ravel(), np.rint(x).astype(int).ravel()])
            self.ray_records[index] = origins, directions, target
            with torch.enable_grad():
                for _ in range(self.cfg["rf_online_steps"]):
                    self.rf_optimizer.zero_grad(set_to_none=True)
                    losses = [F.mse_loss(render_rays(self.models["radiance"], a, b, **self._render_options()), color)
                              for a, b, color in self.ray_records.values()]
                    loss = torch.stack(losses).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite observed RGB radiance loss")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.models["radiance"].parameters(), 10.)
                    self.rf_optimizer.step()
                    self.diagnostics["online_rf_losses"].append(float(loss.detach()))
                    self.diagnostics["online_rf_gradient_norms"].append(float(norm))
        return dict(evidence=evidence[0].tolist(), visibility=view["visibility"], quality=view["quality"], transports=matrices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("train")
    item.add_argument("--data", required=True, help="Object episodes with registered split and image/pose/voxel metadata")
    item.add_argument("--config", help="JSON overrides for documented native model settings")
    item.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    train(args.data, args.output, cfg)


if __name__ == "__main__":
    main()
