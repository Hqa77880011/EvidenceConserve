"""Object-episode calibration of a frozen acquisition and stopping policy."""

import numpy as np
from scipy.stats import beta

from .frontend import object_id


def policy(case):
    return {key: case[key] for key in (
        "classes", "budget", "alpha", "minimum_views", "value_threshold"
    )}


def select_correction(traces, labels, alpha, delta, minimum_coverage, grid):
    """Choose coverage on a fixed grid using simultaneous binomial bounds."""
    grid = np.asarray(grid, dtype=float)
    if not len(traces) or not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("Calibration needs independent episodes and alpha/delta in (0,1)")
    if not 0 <= minimum_coverage <= 1 or grid.ndim != 1 or not len(grid):
        raise ValueError("Invalid minimum coverage or correction grid")
    if not np.isfinite(grid).all() or (grid < 0).any():
        raise ValueError("Correction grid must be finite and nonnegative")
    rows = []
    for correction in grid:
        stopped = [
            next((state for state in trace if state["eligible"] and
                  state["raw_risk"] + correction <= alpha), None)
            for trace in traces
        ]
        selected = [(state, label) for state, label in zip(stopped, labels) if state is not None]
        count = len(selected)
        errors = sum(state["prediction"] != label for state, label in selected)
        upper = float(beta.ppf(1 - delta / len(grid), errors + 1, count - errors)) if count > errors else 1.0
        rows.append(dict(correction=float(correction), stopped=count, errors=errors,
                         coverage=count / len(traces), risk_upper=upper))
    feasible = [row for row in rows if row["coverage"] >= minimum_coverage and
                row["stopped"] > 0 and row["risk_upper"] <= alpha]
    chosen = max(feasible, key=lambda row: (row["coverage"], -row["correction"])) if feasible else None
    return dict(selected=chosen, candidates=rows, candidate_count=len(grid), enabled=chosen is not None)


def calibrate(cases, methods, delta, minimum_coverage, grid, frontend=None, development_objects=None):
    from .core import run_case

    if not cases:
        raise ValueError("Calibration corpus is empty")
    if development_objects is None:
        raise ValueError("Declare all fitting and configuration-selection objects before calibration")
    if not isinstance(development_objects, list) or not all(isinstance(value, (str, int)) for value in development_objects):
        raise ValueError("Development-object provenance must be a JSON list of object IDs")
    development = set(map(str, development_objects))
    if frontend:
        development |= frontend.training_objects
    objects = [object_id(case) for case in cases]
    if len(set(objects)) != len(objects):
        raise ValueError("Use one complete episode per independent calibration object")
    identities = [str(case["id"]) for case in cases]
    if len(set(identities)) != len(identities):
        raise ValueError("Calibration episode IDs must be unique")
    registered = policy(cases[0])
    if any(policy(case) != registered for case in cases):
        raise ValueError("Calibrate one registered policy configuration at a time")
    if set(objects) & development:
        raise ValueError("Fitting/configuration-selection and calibration objects overlap")
    fitted = {}
    for method in methods:
        if method == "no_calibration":
            continue
        traces = [run_case(case, method, frontend=frontend, stop=False)["decisions"] for case in cases]
        fitted[method] = select_correction(
            traces, [case["label"] for case in cases], registered["alpha"],
            delta, minimum_coverage, grid,
        )
    return dict(
        protocol="evidenceconserve.calibration.v1", policy=registered,
        episode_ids=identities, object_ids=objects, delta=delta,
        development_objects=sorted(development),
        minimum_coverage=minimum_coverage, methods=fitted,
        frontend=None if frontend is None else frontend.config,
        interpretation="simultaneous fixed-grid upper bound on calibrated-stop classification error",
    )


def apply_calibration(cases, method, artifact, frontend=None):
    if artifact.get("protocol") != "evidenceconserve.calibration.v1":
        raise ValueError("Unsupported calibration artifact")
    actual = None if frontend is None else frontend.config
    if actual != artifact["frontend"]:
        raise ValueError("The frozen frontend differs from calibration")
    fit = artifact["methods"].get(method)
    if fit is None:
        raise ValueError("No independent calibration for method " + method)
    for case in cases:
        if (str(case["id"]) in artifact["episode_ids"] or
                object_id(case) in set(artifact["object_ids"]) | set(artifact["development_objects"])):
            raise ValueError("Development/calibration and test episode/object IDs overlap")
        if policy(case) != artifact["policy"]:
            raise ValueError("Test policy settings differ from calibration")
    value = fit["selected"]["correction"] if fit["enabled"] else 1.0
    return [dict(case, calibration_correction=value, calibrated_stopping=fit["enabled"]) for case in cases]
