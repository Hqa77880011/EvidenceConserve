"""Exact finite-view oracle and paired duplicate-insertion diagnostics."""

from copy import deepcopy

import numpy as np

from .core import increment, run_case
from .frontend import evidence_record


def oracle(case):
    """Enumerate reachable simple paths and minimize cost of a correct low-risk stop.

    This privileged offline reference reads archived evidence and labels. It is
    never used by the deployable acquisition policy or its calibration.
    """
    views = case["views"]
    if len(views) > case.get("oracle_max_views", 10):
        raise ValueError("Exact oracle is exponential; explicitly set oracle_max_views for a larger graph")
    if any("evidence" not in view for view in views):
        raise ValueError("The offline oracle requires archived evidence for every candidate")
    best = None
    best_key = None

    def visit(acquired, observations, evidence, cost, gains):
        nonlocal best, best_key
        p = (evidence + 1) / (evidence.sum() + case["classes"])
        correct = int(p.argmax()) == case["label"]
        feasible = bool(correct and len(acquired) >= case["minimum_views"] and 1 - p.max() <= case["alpha"])
        key = (0, cost, len(acquired), -p[case["label"]]) if feasible else (1, not correct, -p[case["label"]], cost)
        if acquired and (best_key is None or key < best_key):
            best_key = key
            best = dict(probability=p.tolist(), label=case["label"], views=list(acquired), cost=cost,
                        forced_stop=not feasible, calibrated_stop=False,
                        stop_kind="oracle", gains=list(gains), oracle_feasible=feasible)
        if len(acquired) >= case["budget"] or (best is not None and best["oracle_feasible"] and cost >= best["cost"]):
            return
        for j, view in enumerate(views):
            edge = view.get("from", {}).get(str(acquired[-1])) if acquired else None
            if j in acquired or not view["reachable"] or edge is not None and not edge.get("reachable", True):
                continue
            record = evidence_record(view, view, case["classes"])
            gain, omega = increment(record["evidence"], record["visibility"], acquired, case,
                                    "evidenceconserve", j, observations)
            admitted = record["quality"] * gain
            following = dict(observations)
            following[j] = record
            step_cost = edge.get("cost", view["cost"]) if edge is not None else view["cost"]
            visit(acquired + [j], following, evidence + admitted, cost + step_cost,
                  gains + [dict(gain=float(admitted.sum()), duplicate=view.get("duplicate", False), overlap=omega)])

    visit([], {}, np.zeros(case["classes"]), 0.0, [])
    if best is None:
        best = dict(probability=(np.ones(case["classes"]) / case["classes"]).tolist(),
                    label=case["label"], views=[], cost=0.0, forced_stop=True,
                    calibrated_stop=False, stop_kind="oracle", gains=[], oracle_feasible=False)
    return best


def oracle_metrics(case, result, reference):
    """Report error regret and comparable low-risk path-cost regret separately."""
    probability = np.asarray(result["probability"])
    correct = int(probability.argmax()) == case["label"]
    reference_correct = int(np.argmax(reference["probability"])) == case["label"]
    comparable = (reference["oracle_feasible"] and correct and
                  len(result["views"]) >= case["minimum_views"] and
                  1 - probability.max() <= case["alpha"])
    return dict(oracle_error_regret=int(not correct) - int(not reference_correct),
                oracle_regret=result["cost"] - reference["cost"] if comparable else None,
                oracle_reference_cost=reference["cost"], oracle_reference_feasible=int(reference["oracle_feasible"]))


def duplicate_intervention(case, method, result=None, frontend=None):
    """Force one exact replay after the first capture while retaining the budget.

    The frozen correction is held fixed for the intervention. The perturbed
    sequence is a diagnostic and does not inherit a calibration guarantee.
    """
    original = result if result is not None else run_case(case, method, frontend=frontend)
    if not original["views"] or case["budget"] < 2:
        return dict(duplicate_inserted=0, duplicate_stop_time_change=None, duplicate_increment=None)
    changed = deepcopy(case)
    source = original["views"][0]
    duplicate = deepcopy(changed["views"][source])
    duplicate["duplicate"] = True
    duplicate["from"] = {str(source): dict(reachable=True, cost=duplicate["cost"])}
    duplicate.setdefault("transports", {})[str(source)] = np.eye(case["classes"]).tolist()
    duplicate.setdefault("predicted_overlaps", {})[str(source)] = 1.0
    index = len(changed["views"])
    changed["views"].append(duplicate)
    perturbed = run_case(changed, method, frontend=frontend, forced_actions={1: index})
    inserted = index in perturbed["views"]
    position = perturbed["views"].index(index) if inserted else None
    return dict(duplicate_inserted=int(inserted),
                duplicate_stop_time_change=len(perturbed["views"]) - len(original["views"]),
                duplicate_increment=perturbed["gains"][position]["gain"] if inserted else None,
                duplicate_cost_change=perturbed["cost"] - original["cost"],
                duplicate_prediction_changed=int(np.argmax(perturbed["probability"]) != np.argmax(original["probability"])))
