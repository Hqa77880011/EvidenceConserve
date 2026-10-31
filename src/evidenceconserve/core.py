import numpy as np

from .frontend import evidence_record


METHODS = ["evidenceconserve", "naive_sum", "no_transport", "uniform_views",
           "entropy_nbv", "random_views", "overlap_only", "no_overlap",
           "no_reliability", "no_cost", "no_calibration"]


def overlap(left, right):
    a, b = np.asarray(left) > 0.5, np.asarray(right) > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def increment(evidence, visibility, acquired, case, method, target=None, observations=None):
    if not acquired:
        return np.asarray(evidence), 0.0
    observed = observations if observations is not None else {j: case["views"][j] for j in acquired}
    overlaps = np.array([overlap(visibility, observed[j]["visibility"]) for j in acquired])
    omega = overlaps.max()
    if method in ("naive_sum", "no_overlap"):
        return np.asarray(evidence), omega
    if overlaps.sum() == 0:
        explained = np.zeros_like(evidence, dtype=float)
    else:
        transformed = []
        for j in acquired:
            e = np.asarray(observed[j]["evidence"])
            current = case["views"][target] if target is not None else {}
            matrix = current.get("transports", {}).get(str(j))
            if matrix is None:
                matrix = observed[j].get("transport")
            transport = np.asarray(np.eye(len(e)) if matrix is None else matrix)
            transformed.append(e if method == "no_transport" else transport @ e)
        explained = np.average(np.asarray(transformed), axis=0, weights=overlaps)
    evidence = np.asarray(evidence)
    if method == "overlap_only":
        return (1 - omega) * evidence, omega
    return (1 - omega) * evidence + omega * np.maximum(evidence - explained, 0), omega


def generate(rng, count):
    for index in range(count):
        label = rng.randrange(3)
        views = []
        for i in range(8):
            visibility = [int((j + i) % 8 < 4) for j in range(8)]
            evidence = np.array([rng.random() * 0.2 for _ in range(3)])
            evidence[label] += 1 + i * 0.1
            views.append(
                {
                    "evidence": evidence.tolist(),
                    "visibility": visibility,
                    "quality": 0.9,
                    "cost": rng.uniform(0.5, 2.0),
                    "reachable": True,
                    "duplicate": False,
                }
            )
        views[1] = {**views[0], "duplicate": True}
        yield {
            "id": str(index),
            "family": f"object-{index}",
            "views": views,
            "label": label,
            "classes": 3,
            "budget": 6,
            "alpha": 0.2,
            "minimum_views": 2,
            "value_threshold": 0.001,
        }


def run_case(case, method, frontend=None, stop=True):
    if method not in METHODS:
        raise ValueError("Unknown acquisition method")
    if not 0 < case["alpha"] < 1 or case["budget"] < 1 or case["classes"] < 2:
        raise ValueError("Require alpha in (0,1), a positive budget and at least two classes")
    dimensions = {np.shape(view["visibility"]) for view in case["views"]}
    if len(dimensions) > 1:
        raise ValueError("All views must use the same surface visibility coordinates")
    for view in case["views"]:
        if not np.isfinite(view["cost"]) or view["cost"] <= 0 or not 0 <= view["quality"] <= 1:
            raise ValueError("Views need positive costs and quality in [0,1]")
        for edge in view.get("from", {}).values():
            cost = edge.get("cost", view["cost"])
            if not np.isfinite(cost) or cost <= 0:
                raise ValueError("Directed motion costs must be finite and positive")
    evidence = np.zeros(case["classes"])
    acquired, gain_log = [], []
    observations, decisions = {}, []
    total_cost = 0.0
    forced = True
    for step in range(case["budget"]):
        probability = (evidence + 1) / (evidence.sum() + case["classes"])
        risk = (
            1
            - probability.max()
            + (0 if method == "no_calibration" else case.get("calibration_correction", 1.0))
        )
        options = []
        for j, view in enumerate(case["views"]):
            edge = view.get("from", {}).get(str(acquired[-1])) if acquired else None
            if j in acquired or not view["reachable"] or (edge is not None and not edge.get("reachable", True)):
                continue
            predicted_evidence = view.get("predicted_evidence", (3 * probability).tolist())
            predicted_record = evidence_record(dict(evidence=predicted_evidence), view, case["classes"])
            gain, _ = increment(predicted_record["evidence"], view["visibility"], acquired, case, method, j, observations)
            quality = 1.0 if method == "no_reliability" else view["quality"]
            following = evidence + quality * gain
            predicted = (following + 1) / (following.sum() + case["classes"])
            cost = edge.get("cost", view["cost"]) if edge is not None else view["cost"]
            reduction = predicted.max() - probability.max()
            if method == "entropy_nbv":
                reduction = -(probability * np.log(probability)).sum() + (predicted * np.log(predicted)).sum()
            value = reduction / (1.0 if method == "no_cost" else cost + 1e-8)
            options.append((float(value), -j, j))
        best = max(options, default=(0.0, 0, None))
        eligible = step >= case["minimum_views"] and best[0] <= case["value_threshold"]
        decisions.append(dict(views=list(acquired), raw_risk=float(1 - probability.max()),
                              prediction=int(probability.argmax()), best_value=best[0],
                              eligible=bool(eligible)))
        if (
            stop and (method == "no_calibration" or case.get("calibrated_stopping", False))
            and eligible
            and risk <= case["alpha"]
        ):
            forced = False
            break
        if best[2] is None:
            break
        j = min(j for _, _, j in options) if method == "uniform_views" else best[2]
        if method == "random_views":
            # Fixed per-episode seed is declared before seeing labels or captures.
            j = np.random.default_rng(case.get("seed", 17) + step).choice([option[2] for option in options]).item()
        view = case["views"][j]
        record = frontend.capture(case, j, observations) if frontend else evidence_record(view, view, case["classes"])
        record = evidence_record(record, view, case["classes"])
        effective_case = dict(case, views=list(case["views"]))
        effective_case["views"][j] = dict(view, transports=record["transports"])
        gain, omega = increment(record["evidence"], record["visibility"], acquired, effective_case, method, j, observations)
        admitted = (1.0 if method == "no_reliability" else record["quality"]) * gain
        evidence += admitted
        gain_log.append(
            {
                "gain": float(admitted.sum()),
                "duplicate": view.get("duplicate", False),
                "overlap": omega,
            }
        )
        edge = view.get("from", {}).get(str(acquired[-1])) if acquired else None
        total_cost += edge.get("cost", view["cost"]) if edge is not None else view["cost"]
        acquired.append(j)
        observations[j] = record
    probability = (evidence + 1) / (evidence.sum() + case["classes"])
    calibrated = not forced and method != "no_calibration" and case.get("calibrated_stopping", False)
    return {
        "probability": probability.tolist(),
        "label": case["label"],
        "views": acquired,
        "cost": total_cost,
        "forced_stop": forced,
        "calibrated_stop": bool(calibrated),
        "stop_kind": "forced" if forced else "calibrated" if calibrated else "uncalibrated",
        "calibration_correction": case.get("calibration_correction") if calibrated else None,
        "gains": gain_log,
        "decisions": decisions,
        "observations": {str(j): record for j, record in observations.items()},
    }


def score(case, result):
    p = np.asarray(result["probability"])
    label = case["label"]
    correct = p.argmax() == label
    duplicate = [x["gain"] for x in result["gains"] if x["duplicate"]]
    new = [x["gain"] for x in result["gains"] if x["overlap"] < 0.1]
    return {
        "accuracy": int(correct),
        "view_count": len(result["views"]),
        "motion_cost": result["cost"],
        "capture_seconds": float(sum(case["views"][j].get("capture_seconds", 0) for j in result["views"])),
        "forced_stop": int(result["forced_stop"]),
        "calibrated_stop": int(result["calibrated_stop"]),
        "unsafe_calibrated_stop": int(not correct) if result["calibrated_stop"] else None,
        "unsafe_uncalibrated_stop": int(not correct) if result["stop_kind"] == "uncalibrated" else None,
        "forced_stop_error": int(not correct) if result["forced_stop"] else None,
        "duplicate_gain": float(np.mean(duplicate)) if duplicate else None,
        "new_surface_gain": float(np.mean(new)) if new else None,
        "brier": float(np.sum((p - np.eye(len(p))[label]) ** 2)),
        "nll": float(-np.log(max(p[label], 1e-12))),
        "legal_effects": int(correct),
    }
