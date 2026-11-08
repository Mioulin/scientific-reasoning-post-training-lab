"""Original deterministic synthetic tasks; no human preferences or private sources."""

import hashlib
import json
import random
from pathlib import Path

VERSION = "scireason-synthetic-v1"
SYSTEM = (
    "Solve the scientific task. Return only a JSON object with exactly two keys: "
    '"answer" (a number, "supported", "unsupported", or null) and "abstain" (boolean). '
    "Use null and true only when the requested quantity cannot be determined. "
    "For evidence questions, answer supported or unsupported and abstain=false."
)
FAMILIES = ["rate", "mean", "difference", "missing", "causality", "evidence"]
OOD = ["bayes", "decay", "missing_variance", "extrapolation"]


def response(answer):
    return json.dumps({"answer": answer, "abstain": answer is None}, separators=(",", ":"))


def example(family, split, i, rng):
    # Distinct wording per split; numerical instances are independently sampled.
    a, b, c = [rng.randint(3, 199) for _ in range(3)]
    variant = {"train": 0, "validation": 1, "test": 2, "final": 3}[split]
    if family == "rate":
        prompts = [
            f"A neuron emitted {a} spikes in {b} seconds. What is its mean firing rate in Hz?",
            f"Calculate spikes per second from a recording lasting {b} s containing {a} spikes.",
            f"The spike count is {a}; observation duration is {b} seconds. Report count divided by duration.",
            f"Over an interval of {b} seconds, {a} neural events occurred. Give the average event frequency (Hz).",
        ]
        answer = a / b
    elif family == "mean":
        prompts = [
            f"Three measurements are {a}, {b}, {c}. Find their arithmetic mean.",
            f"Report the average of observations [{a}, {b}, {c}].",
            f"A sensor produced {a} then {b} then {c}. Compute sum/number of readings.",
            f"Equal-weight replicate values: {a}; {b}; {c}. What is the sample mean?",
        ]
        answer = (a + b + c) / 3
    elif family == "difference":
        prompts = [
            f"Condition A has mean {a}; condition B has mean {b}. Calculate A minus B.",
            f"Report the signed contrast between treatment ({a}) and control ({b}): treatment - control.",
            f"Baseline is {b} units and follow-up is {a} units. Give follow-up minus baseline.",
            f"A signal changes from {b} to {a}. What is its signed change in units?",
        ]
        answer = a - b
    elif family == "missing":
        prompts = [
            f"A neuron emitted {a} spikes. Recording duration is not reported. What is its firing rate in Hz?",
            f"Compute events per second from {a} events, with the elapsed time unknown.",
            f"A spike counter reads {a}. No timing information exists. Determine the numerical frequency.",
            f"There were {a} neural events in an unspecified interval. Give the average event frequency (Hz).",
        ]
        answer = None
    elif family == "causality":
        prompts = [
            f'An observational sample of {a + b} people shows an association between X and Y. No interventions or confounder controls. Is "X causes Y" supported?',
            f"Among {a + b} cases, X correlates with Y; assignment was not random and confounding was not addressed. Is causation established by these data?",
            f"A survey (n={a + b}) links X with Y without adjusting for common causes. Is the causal claim supported?",
            f"In {a + b} subjects X predicts Y. Only observational associations are given. Does this establish that changing X changes Y?",
        ]
        answer = "unsupported"
    elif family == "evidence":
        supported = i % 2 == 0
        threshold = a - 1 if supported else a + 1
        prompts = [
            f'The measured mean is {a}. Is the claim "the measured mean exceeds {threshold}" supported?',
            f"An observed average equals {a}. Does it support the statement that this average is greater than {threshold}?",
            f"Reported sample mean: {a}. Assess the assertion sample mean > {threshold}.",
            f'Only the sample average {a} is reported. Is "the sample average is above {threshold}" supported?',
        ]
        answer = "supported" if supported else "unsupported"
    elif family == "bayes":
        # Integer population counts, exact conditional probability.
        prompts = [
            f"Among test-positive people, {a} have disease and {b} do not. What fraction of test-positive people have disease?"
        ] * 4
        answer = a / (a + b)
    elif family == "decay":
        prompts = [
            f"A substance starts with {a} mg and halves every hour. How many mg remain after 2 hours?"
        ] * 4
        answer = a / 4
    elif family == "missing_variance":
        prompts = [
            f"A sample has size {a + 2} and mean {b}. No dispersion information is supplied. What is its numerical standard deviation?"
        ] * 4
        answer = None
    else:
        prompts = [
            f'All {a + 2} measured cells in one selected tissue respond to X. Is "every cell in every tissue responds to X" supported?'
        ] * 4
        answer = "unsupported"
    prompt = prompts[variant]
    chosen = response(round(answer, 6) if isinstance(answer, (int, float)) else answer)
    wrong = (
        answer + max(1, abs(answer) * 0.2)
        if isinstance(answer, (int, float))
        else (float(c) if answer is None else ("unsupported" if answer == "supported" else "supported"))
    )
    rejected = response(round(wrong, 6) if isinstance(wrong, (int, float)) else wrong)
    return {
        "id": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "family": family,
        "template": f"{family}:{variant}",
        "split": split,
        "distribution": "ood" if family in OOD else "id",
        "prompt": prompt,
        "answer": answer,
        "chosen": chosen,
        "rejected": rejected,
        "preference_reason": "Programmatic ground truth; rejected answer violates the task oracle.",
        "difficulty": "single-step" if family not in OOD else "held-out-family",
        "source": VERSION,
    }


def prepare(root="data"):
    root = Path(root)
    root.mkdir(exist_ok=True)
    manifest = {"version": VERSION, "seed": 1729, "license": "CC0-1.0", "splits": {}}
    seen = set()
    for split, n in [("train", 48), ("validation", 8), ("test", 12), ("final", 12)]:
        rng = random.Random(f"1729-{split}")
        families = FAMILIES if split in ("train", "validation") else FAMILIES + OOD
        rows = []
        for family in families:
            for i in range(n):
                row = example(family, split, i, rng)
                while row["id"] in seen:
                    row = example(family, split, i, rng)
                seen.add(row["id"])
                rows.append(row)
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (root / f"{split}.jsonl").write_text(payload)
        manifest["splits"][split] = {
            "n": len(rows),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "families": families,
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read(split, root="data"):
    return [json.loads(line) for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()]


def messages(row):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}]
