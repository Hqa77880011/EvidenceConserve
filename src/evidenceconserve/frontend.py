"""Capture-time evidence from a frozen, client-local visual frontend."""

import json
from pathlib import Path
import subprocess

import numpy as np


def object_id(case):
    return str(case.get("object_id", case.get("cluster", case.get("family", case["id"]))))


def evidence_record(record, view, classes):
    """Check the external evidence boundary before accumulating support."""
    evidence = np.asarray(record["evidence"], dtype=float)
    visibility = np.asarray(record.get("visibility", view["visibility"]), dtype=float)
    quality = float(record.get("quality", view["quality"]))
    if evidence.shape != (classes,) or not np.isfinite(evidence).all() or (evidence < 0).any():
        raise ValueError("Frontend evidence must be a finite nonnegative class vector")
    if (visibility.ndim != 1 or not visibility.size or
            visibility.shape != np.shape(view["visibility"]) or
            not np.isfinite(visibility).all() or (visibility < 0).any() or (visibility > 1).any()):
        raise ValueError("Captured and candidate visibility descriptors must align")
    if not 0 <= quality <= 1:
        raise ValueError("Frontend quality must lie in [0,1]")
    transports = record.get("transports", view.get("transports", {}))
    if not isinstance(transports, dict):
        raise ValueError("transports must map previous view indices to matrices")
    for matrix in list(transports.values()) + ([record["transport"]] if record.get("transport") is not None else []):
        value = np.asarray(matrix, dtype=float)
        if value.shape != (classes, classes) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError("Evidence transports must be finite nonnegative K by K matrices")
    return dict(
        evidence=evidence.tolist(), visibility=visibility.tolist(), quality=quality,
        transports=transports, transport=record.get("transport", view.get("transport")),
    )


class CommandFrontend:
    """Invoke a frozen local image/geometry model only after selecting a view."""

    def __init__(self, config):
        self.config = dict(config)
        command = config.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("Frontend command must be a nonempty argv list")
        self.training_objects = set(map(str, config.get("training_objects", [])))

    def capture(self, case, index, history):
        view = case["views"][index]
        request = {
            "protocol": "evidenceconserve.capture.v1", "episode_id": str(case["id"]),
            "object_id": object_id(case), "classes": case["classes"], "view_index": index,
            "view": {key: view[key] for key in (
                "id", "image", "pose", "visibility", "quality", "cost", "transports"
            ) if key in view},
            "history": [{"view_index": j, **record} for j, record in sorted(history.items())],
        }
        response = subprocess.run(
            self.config["command"], input=json.dumps(request), text=True,
            encoding="utf-8", capture_output=True, check=True,
            timeout=float(self.config.get("timeout_seconds", 120)),
            cwd=self.config.get("cwd"), shell=False,
        )
        record = json.loads(response.stdout)
        if not isinstance(record, dict):
            raise ValueError("Frontend stdout must contain one JSON object")
        return evidence_record(record, view, case["classes"])


def load_frontend(path):
    if path is None:
        return None
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("kind") == "native":
        from .native import NativeBackend
        return NativeBackend((config_path.parent / config["checkpoint"]).resolve())
    if "cwd" in config:
        config["cwd"] = str((config_path.parent / config["cwd"]).resolve())
    return CommandFrontend(config)
