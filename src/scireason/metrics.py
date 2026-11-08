"""Strict structured evaluation; no LLM judge or free-form hallucination claims."""

import json
import math

import numpy as np


def grade(text, expected):
    try:
        obj = json.loads(text.strip())
        valid = (
            isinstance(obj, dict)
            and set(obj) == {"answer", "abstain"}
            and isinstance(obj["abstain"], bool)
            and (
                obj["answer"] is None
                or type(obj["answer"]) in (int, float)
                or obj["answer"] in ("supported", "unsupported")
            )
        )
        if not valid:
            raise ValueError("schema")
    except (ValueError, TypeError):
        return {
            "correct": False,
            "format_valid": False,
            "abstained": False,
            "absolute_error": None,
            "unsupported_assertion": False,
            "error": "format",
        }
    answer = obj["answer"]
    abstained = obj["abstain"] and answer is None
    err = None
    if expected is None:
        correct = abstained
    elif type(expected) in (int, float):
        if type(answer) in (int, float) and math.isfinite(answer):
            err = abs(answer - expected)
        correct = (
            err is not None
            and math.isclose(answer, expected, rel_tol=1e-3, abs_tol=1e-3)
            and not obj["abstain"]
        )
    else:
        correct = answer == expected and not obj["abstain"]
    return {
        "correct": bool(correct),
        "format_valid": True,
        "abstained": abstained,
        "absolute_error": err,
        "unsupported_assertion": expected == "unsupported" and answer == "supported" and not obj["abstain"],
        "error": "correct" if correct else ("abstention" if expected is None or obj["abstain"] else "answer"),
    }


def summarise(rows):
    n = len(rows)
    tp = sum(r["grade"]["abstained"] and r["answer"] is None for r in rows)
    predicted = sum(r["grade"]["abstained"] for r in rows)
    needed = sum(r["answer"] is None for r in rows)
    numeric = [r for r in rows if type(r["answer"]) in (int, float)]
    errors = [r["grade"]["absolute_error"] for r in numeric if r["grade"]["absolute_error"] is not None]
    evidence = [r for r in rows if r["answer"] == "unsupported"]
    return {
        "n": n,
        "accuracy": sum(r["grade"]["correct"] for r in rows) / n,
        "format_valid_rate": sum(r["grade"]["format_valid"] for r in rows) / n,
        "numeric_accuracy": sum(r["grade"]["correct"] for r in numeric) / len(numeric) if numeric else None,
        "numeric_mae_parseable_only": float(np.mean(errors)) if errors else None,
        "numeric_parseable_n": len(errors),
        "numeric_n": len(numeric),
        "abstention_precision": tp / predicted if predicted else None,
        "abstention_recall": tp / needed if needed else None,
        "unsupported_assertion_rate": sum(r["grade"]["unsupported_assertion"] for r in evidence)
        / len(evidence)
        if evidence
        else None,
        "by_family": {
            f: sum(r["grade"]["correct"] for r in rows if r["family"] == f)
            / sum(r["family"] == f for r in rows)
            for f in sorted({r["family"] for r in rows})
        },
    }


def paired_ci(a, b, seed=2026):
    assert [r["id"] for r in a] == [r["id"] for r in b]
    diff = np.array([int(y["grade"]["correct"]) - int(x["grade"]["correct"]) for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "method": "paired example bootstrap; does not capture template or training-seed uncertainty",
    }
