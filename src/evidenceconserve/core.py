import numpy as np


METHODS = ["evidenceconserve", "naive_sum", "no_transport", "uniform_views", "no_calibration"]


def overlap(left, right):
    a, b = np.asarray(left) > 0.5, np.asarray(right) > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def increment(evidence, visibility, acquired, case, method):
    if not acquired or method == "naive_sum":
        return np.asarray(evidence), 0.0
    overlaps = np.array([overlap(visibility, case["views"][j]["visibility"]) for j in acquired])
    omega = overlaps.max()
    if overlaps.sum() == 0:
        explained = np.zeros_like(evidence, dtype=float)
    else:
        transformed = []
        for j in acquired:
            e = np.asarray(case["views"][j]["evidence"])
            transport = np.asarray(case["views"][j].get("transport", np.eye(len(e))))
            transformed.append(e if method == "no_transport" else transport @ e)
        explained = np.average(np.asarray(transformed), axis=0, weights=overlaps)
    evidence = np.asarray(evidence)
    return (1 - omega) * evidence + omega * np.maximum(evidence - explained, 0), omega


def correction(risks, losses, delta):
    values = np.sort(np.asarray(losses) - np.asarray(risks))
    index = int(np.ceil((len(values) + 1) * (1 - delta))) - 1
    return float(values[index]) if 0 <= index < len(values) else 1.0


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
            "calibration_correction": 0.05,
        }


def run_case(case, method):
    evidence = np.zeros(case["classes"])
    acquired, gain_log = [], []
    total_cost = 0.0
    forced = True
    for step in range(case["budget"]):
        probability = (evidence + 1) / (evidence.sum() + case["classes"])
        risk = (
            1
            - probability.max()
            + (0 if method == "no_calibration" else case["calibration_correction"])
        )
        options = []
        for j, view in enumerate(case["views"]):
            if j in acquired or not view["reachable"]:
                continue
            predicted_evidence = view.get("predicted_evidence", (3 * probability).tolist())
            gain, _ = increment(predicted_evidence, view["visibility"], acquired, case, method)
            following = evidence + view["quality"] * gain
            predicted = (following + 1) / (following.sum() + case["classes"])
            value = (probability.max() - predicted.max()) * -1 / (view["cost"] + 1e-8)
            options.append((float(value), -j, j))
        best = max(options, default=(0.0, 0, None))
        if (
            step >= case["minimum_views"]
            and risk <= case["alpha"]
            and best[0] <= case["value_threshold"]
        ):
            forced = False
            break
        if best[2] is None:
            break
        j = min(j for _, _, j in options) if method == "uniform_views" else best[2]
        view = case["views"][j]
        gain, omega = increment(view["evidence"], view["visibility"], acquired, case, method)
        admitted = view["quality"] * gain
        evidence += admitted
        gain_log.append(
            {
                "gain": float(admitted.sum()),
                "duplicate": view.get("duplicate", False),
                "overlap": omega,
            }
        )
        total_cost += view["cost"]
        acquired.append(j)
    probability = (evidence + 1) / (evidence.sum() + case["classes"])
    return {
        "probability": probability.tolist(),
        "views": acquired,
        "cost": total_cost,
        "forced_stop": forced,
        "gains": gain_log,
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
        "forced_stop": int(result["forced_stop"]),
        "unsafe_calibrated_stop": int(not correct) if not result["forced_stop"] else None,
        "duplicate_gain": float(np.mean(duplicate)) if duplicate else None,
        "new_surface_gain": float(np.mean(new)) if new else None,
        "brier": float(np.sum((p - np.eye(len(p))[label]) ** 2)),
        "nll": float(-np.log(max(p[label], 1e-12))),
        "legal_effects": int(correct),
    }
