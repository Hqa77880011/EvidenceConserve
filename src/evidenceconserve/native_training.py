"""Object-disjoint native training for images, transport, aggregation and PPO."""

import copy
import json
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .cli import read_jsonl
from .core import overlap
from .frontend import object_id
from .native_models import (CandidateDynamics, ConcatEvidence, EvidenceEncoder,
                            GenNBVGraphPolicy, PoseTransport, SetEvidence,
                            TinyRadianceField, evidential_loss, ppo_loss, relative_pose)


DEFAULTS = dict(seed=13, epochs=150, patience=15, batch_size=8, learning_rate=3e-4,
                image_size=224, feature_dim=64, width=64, dropout=.2, max_views=6,
                transport_min_overlap=.5, ppo_updates=150, ppo_epochs=4,
                ppo_gamma=.99, ppo_lambda=.95, path_weight=.01, terminal_bonus=.05,
                rf_online_steps=10, rf_rays_side=3, rf_samples=8, rf_near=.1, rf_far=4.,
                rf_regularization=.001, mc_samples=16)


def read_cases(path):
    """Resolve image/voxel paths and reject object overlap among registered splits."""
    cases = read_jsonl(path)
    aliases = {"validation": "policy_validation", "val": "policy_validation"}
    assignment = {}
    for case in cases:
        case["split"] = aliases.get(case["split"], case["split"])
        if case["split"] not in ("train", "policy_validation", "calibration", "test"):
            raise ValueError("Use train/policy_validation/calibration/test object splits")
        identity = object_id(case)
        if identity in assignment and assignment[identity] != case["split"]:
            raise ValueError("An object occurs in multiple native training/evaluation splits")
        assignment[identity] = case["split"]
        for view in case["views"]:
            if "voxel_path" in view:
                view["voxel_path"] = str((Path(path).resolve().parent / view["voxel_path"]).resolve())
    return cases


def image_tensor(path, size):
    """Load RGB only when requested, keeping the declared resize convention."""
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


def pose_tensor(view):
    """Camera-to-world xyz and xyz Euler angles in radians."""
    value = torch.as_tensor(view["pose"], dtype=torch.float32)
    if value.shape != (6,) or not torch.isfinite(value).all():
        raise ValueError("Native view pose must contain six finite xyz/xyz-Euler values")
    return value


def make_models(cfg):
    """Instantiate the same registered architecture for training and checkpoint loading."""
    k, d, width = cfg["classes"], cfg["feature_dim"], cfg["width"]
    return dict(encoder=EvidenceEncoder(k, d, cfg["dropout"]), transport=PoseTransport(k, width),
                set_transformer=SetEvidence(d, k, width),
                feature_concatenation=ConcatEvidence(d, k, cfg["max_views"], width),
                dynamics=CandidateDynamics(k, d, width, cfg["dropout"]),
                gennbv=GenNBVGraphPolicy(width=min(width, 64)), radiance=TinyRadianceField(min(width, 24)))


def fit_supervised(model, training, validation, loss_batch, cfg):
    """Fit only training rows; select the checkpoint only by held-out object loss."""
    if not training or not validation:
        raise ValueError("Native training needs nonempty train and policy-validation samples")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
    rng = np.random.default_rng(cfg["seed"])
    best, stale, selected = float("inf"), 0, None
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss, maximum_gradient = 0., 0.
        indices = rng.permutation(len(training))
        for start in range(0, len(indices), cfg["batch_size"]):
            batch = [training[i] for i in indices[start:start + cfg["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Native model produced non-finite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("Native model produced non-finite gradients")
            maximum_gradient = max(maximum_gradient, float(gradient))
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            valid = np.mean([float(loss_batch(model, validation[start:start + cfg["batch_size"]]))
                             for start in range(0, len(validation), cfg["batch_size"])])
        if not np.isfinite(valid):
            raise FloatingPointError("Native model produced non-finite validation loss")
        history.append(dict(epoch=epoch + 1, train_loss=train_loss / len(training),
                            validation_loss=float(valid), maximum_gradient_norm=maximum_gradient))
        if valid < best:
            best, stale, selected = valid, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(selected)
    model.eval()
    return history


def dynamics_input(case, candidate, acquired, features, evidences, cfg):
    """Use only candidate geometry and already captured image/evidence features."""
    target = pose_tensor(case["views"][candidate])
    previous = pose_tensor(case["views"][acquired[-1]]) if acquired else target
    evidence = torch.stack(evidences).sum(0) if evidences else torch.zeros(cfg["classes"])
    probability = (evidence + 1) / (evidence.sum() + cfg["classes"])
    mean_feature = torch.stack(features).mean(0) if features else torch.zeros(cfg["feature_dim"])
    novelty = 1 - max((overlap(case["views"][candidate]["visibility"], case["views"][j]["visibility"])
                       for j in acquired), default=0.)
    return torch.cat([target, relative_pose(target, previous), probability, mean_feature, torch.tensor([novelty])])


def voxels(view):
    """Read only a captured view's occupancy/depth-fusion grid."""
    value = np.load(view["voxel_path"], allow_pickle=False) if "voxel_path" in view else np.asarray(view["observed_voxels"])
    if value.ndim != 3 or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError("GenNBV needs a finite observed occupancy grid [D,H,W] in [0,1]")
    return torch.as_tensor(value, dtype=torch.float32)


def policy_state(case, acquired):
    """Released GenNBV occupancy/periodic-motion streams from acquired history."""
    shape = tuple(case["voxel_shape"])
    voxel = torch.zeros(shape)
    poses = []
    for j in acquired:
        value = voxels(case["views"][j])
        if value.shape != voxel.shape:
            raise ValueError("Captured voxel grids must align with voxel_shape")
        voxel = torch.maximum(voxel, value)
        poses.append(pose_tensor(case["views"][j]))
    motion = torch.stack(poses) if poses else torch.zeros(1, 6)
    candidates = torch.stack([pose_tensor(view) for view in case["views"]])
    return voxel[None, None], motion[None], candidates[None]


def available(case, acquired):
    allowed = []
    for j, view in enumerate(case["views"]):
        edge = view.get("from", {}).get(str(acquired[-1])) if acquired else None
        if j not in acquired and view["reachable"] and (edge is None or edge.get("reachable", True)):
            allowed.append(j)
    return allowed


def coverage_rollout(model, case, cfg, stochastic):
    """Coverage-reward rollout; only training/validation objects expose reference coverage."""
    model.eval()
    acquired, transitions, rewards = [], [], []
    covered = torch.zeros(tuple(case["voxel_shape"]))
    reference = torch.stack([voxels(view) for view in case["views"]]).amax(0) > .5
    denominator = max(int(reference.sum()), 1)
    for step in range(min(case["budget"], cfg["max_views"])):
        allowed = available(case, acquired)
        if not allowed:
            break
        state = policy_state(case, acquired)
        with torch.no_grad():
            logits, value = model(*state)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[:, allowed] = True
            logits = logits.masked_fill(~mask, -1e9)
            policy = torch.distributions.Categorical(logits=logits)
            action = int(policy.sample()) if stochastic else int(logits.argmax(-1))
            if step == 0:
                action = int(case.get("start_view", allowed[0]))
            old_log = policy.log_prob(torch.tensor([action])).item()
        updated = torch.maximum(covered, voxels(case["views"][action]))
        gain = float(((updated > .5) & ~(covered > .5) & reference).sum()) / denominator
        edge = case["views"][action].get("from", {}).get(str(acquired[-1])) if acquired else None
        cost = edge.get("cost", case["views"][action]["cost"]) if edge else case["views"][action]["cost"]
        reward = gain - cfg["path_weight"] * cost
        if step + 1 == min(case["budget"], cfg["max_views"]):
            reward += cfg["terminal_bonus"] * float(((updated > .5) & reference).sum()) / denominator
        transitions.append((state, mask, action, old_log, float(value)))
        rewards.append(reward)
        acquired.append(action)
        covered = updated
    return transitions, rewards, float(((covered > .5) & reference).sum()) / denominator


def fit_policy(model, training, validation, cfg):
    """GenNBV-style clipped PPO with GAE, trained on reconstruction coverage."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best, stale, selected, history = -float("inf"), 0, None, []
    for update in range(cfg["ppo_updates"]):
        model.train()
        batch = []
        for case in training:
            transitions, rewards, _ = coverage_rollout(model, case, cfg, True)
            gae, following_value = 0., 0.
            for i in range(len(transitions) - 1, -1, -1):
                value = transitions[i][-1]
                delta = rewards[i] + cfg["ppo_gamma"] * following_value - value
                gae = delta + cfg["ppo_gamma"] * cfg["ppo_lambda"] * gae
                if i > 0:
                    # The matched initial view is exogenous, not sampled by PPO.
                    batch.append((transitions[i], gae, gae + value))
                following_value = value
        if len(batch) < 2:
            raise ValueError("PPO needs at least two legal rollout transitions")
        advantages = torch.tensor([row[1] for row in batch])
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        maximum_gradient = 0.
        for _ in range(cfg["ppo_epochs"]):
            model.train()
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm3d):
                    layer.eval()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (transition, _, returns) in enumerate(batch):
                state, mask, action, old_log, _ = transition
                logits, values = model(*state)
                losses.append(ppo_loss(logits.masked_fill(~mask, -1e9), values, torch.tensor([action]),
                                       torch.tensor([old_log]), advantages[i:i + 1], torch.tensor([returns])))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            maximum_gradient = max(maximum_gradient, float(norm))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            coverage = np.mean([coverage_rollout(model, case, cfg, False)[2] for case in validation])
        history.append(dict(update=update + 1, loss=float(loss.detach()), validation_coverage=float(coverage), maximum_gradient_norm=maximum_gradient))
        if coverage > best:
            best, stale, selected = coverage, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(selected)
    model.eval()
    return history


def train(path, output, configuration=None):
    """Train all native components; calibration/test objects never enter fitting."""
    cfg = dict(DEFAULTS, **(configuration or {}))
    if cfg["image_size"] < 64:
        raise ValueError("Use image_size >=64 for ResNet training batches of one")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    cases = read_cases(path)
    training = [case for case in cases if case["split"] == "train"]
    validation = [case for case in cases if case["split"] == "policy_validation"]
    if not training or not validation:
        raise ValueError("Declare disjoint train and policy_validation objects")
    cfg["classes"] = training[0]["classes"]
    dev = training + validation
    pixels = [image_tensor(view["image"], cfg["image_size"]) for case in training for view in case["views"]]
    pixel_stack = torch.stack(pixels)
    mean = pixel_stack.mean((0, 2, 3))
    std = pixel_stack.std((0, 2, 3), unbiased=False).clamp_min(.01)
    models = make_models(cfg)
    report = {}
    images = {split: [(image_tensor(view["image"], cfg["image_size"]), case["label"])
                       for case in group for view in case["views"]]
              for split, group in (("train", training), ("val", validation))}

    def encoder_loss(model, batch):
        image = torch.stack([row[0] for row in batch])
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]
        return evidential_loss(model(image)[0], torch.tensor([row[1] for row in batch]))

    report["encoder"] = fit_supervised(models["encoder"], images["train"], images["val"], encoder_loss, cfg)
    cache = {}
    with torch.no_grad():
        for case in dev:
            encoded = []
            for view in case["views"]:
                image = (image_tensor(view["image"], cfg["image_size"]) - mean[:, None, None]) / std[:, None, None]
                evidence, feature = models["encoder"](image[None])
                encoded.append((evidence[0], feature[0]))
            cache[object_id(case)] = encoded
    samples = {kind: {"train": [], "val": []} for kind in ("transport", "aggregation", "dynamics")}
    for case in dev:
        split = "train" if case["split"] == "train" else "val"
        encoded = cache[object_id(case)]
        for j, target in enumerate(case["views"]):
            for k, source in enumerate(case["views"]):
                weight = overlap(target["visibility"], source["visibility"])
                if j != k and weight >= cfg["transport_min_overlap"]:
                    samples["transport"][split].append((pose_tensor(target), pose_tensor(source), encoded[k][0], encoded[j][0], weight))
        for count in range(1, min(len(encoded), cfg["max_views"]) + 1):
            features = torch.stack([row[1] for row in encoded[:count]])
            samples["aggregation"][split].append((features, case["label"]))
        for count in range(min(len(encoded), cfg["max_views"])):
            acquired = list(range(count))
            for j in range(count, len(encoded)):
                values = dynamics_input(case, j, acquired, [encoded[k][1] for k in acquired], [encoded[k][0] for k in acquired], cfg)
                samples["dynamics"][split].append((values, *encoded[j]))

    def transport_loss(model, batch):
        target, source, evidence, expected, weight = (torch.stack([torch.as_tensor(row[i]) for row in batch]) for i in range(5))
        matrix = model(target, source)
        predicted = (matrix @ evidence[..., None])[..., 0]
        identity = torch.eye(cfg["classes"])[None]
        return ((predicted - expected).square().mean(-1) * weight).mean() + .001 * (matrix - identity).square().mean()

    report["transport"] = fit_supervised(models["transport"], samples["transport"]["train"], samples["transport"]["val"], transport_loss, cfg)

    def aggregation_loss(model, batch):
        return torch.stack([evidential_loss(model(feature[None]), torch.tensor([label])) for feature, label in batch]).mean()

    for name in ("set_transformer", "feature_concatenation"):
        report[name] = fit_supervised(models[name], samples["aggregation"]["train"], samples["aggregation"]["val"], aggregation_loss, cfg)

    def dynamics_loss(model, batch):
        value, evidence, feature = (torch.stack([row[i] for row in batch]) for i in range(3))
        predicted_evidence, predicted_feature = model(value)
        return F.mse_loss(predicted_evidence, evidence) + F.mse_loss(predicted_feature, feature)

    report["dynamics"] = fit_supervised(models["dynamics"], samples["dynamics"]["train"], samples["dynamics"]["val"], dynamics_loss, cfg)
    report["gennbv"] = fit_policy(models["gennbv"], training, validation, cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    registration = dict(id=str(uuid.uuid4()), train_objects=sorted({object_id(case) for case in training}),
                        validation_objects=sorted({object_id(case) for case in validation}), configuration=cfg)
    artifact = dict(protocol="evidenceconserve.native.v1", config=cfg, mean=mean, std=std,
                    registration=registration, states={name: model.state_dict() for name, model in models.items()})
    torch.save(artifact, output / "native.pt")
    (output / "training.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    return artifact, report
