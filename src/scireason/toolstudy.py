"""Fresh family-held-out benchmark and bounded program-aided inference.

No eval/exec, external tools, files or labels are exposed to the policy.
"""

import ast
import hashlib
import json
import math
import operator
import random
import time
from decimal import Decimal
from pathlib import Path

import torch
from transformers import set_seed

from .data import SYSTEM, response
from .metrics import grade, paired_ci, summarise
from .runtime import config, dump, hardware, load, memory

VERSION = "scireason-program-v1"
ROOT = "data/toolstudy"
TRAIN = ["rate", "mean", "difference", "weighted", "bayes", "missing_rate", "causality", "evidence"]
OOD = ["decay", "dilution", "zscore", "missing_sd"]
TOOL_SYSTEM = (
    "Solve the scientific task. Return only JSON with exactly keys answer and abstain. "
    'For numerical questions, answer must be an arithmetic expression STRING, e.g. "(12+6)/3". '
    "A calculator will execute it. Allowed: numbers, parentheses, + - * / ** and sqrt(). "
    'For evidence questions use answer "supported" or "unsupported" and abstain false. '
    "For insufficient information use answer null and abstain true. No explanations."
)


def calculate(expression):
    """Parse a tiny expression language, not Python execution."""
    if not isinstance(expression, str) or len(expression) > 160:
        raise ValueError("expression_length")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 48:
        raise ValueError("expression_complexity")
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def visit(node):
        if isinstance(node, ast.Expression):
            value = visit(node.body)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            value = binary[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, power = visit(node.left), visit(node.right)
            if abs(power) > 12:
                raise ValueError("power_limit")
            value = base**power
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            value = math.sqrt(visit(node.args[0]))
        else:
            raise ValueError("unsupported_syntax")
        if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("numeric_range")
        return value

    return visit(tree)


def make_example(family, split, i, rng):
    a, b, c = [rng.randint(11, 197) for _ in range(3)]
    d = rng.randint(2, 9)
    variant = {"train": 0, "validation": 1, "test": 2, "final": 3}[split]
    # Distinct wrappers alone would not establish task independence. Family-specific
    # formulations and numerical-condition grouping are both recorded below.
    D = Decimal
    if family == "rate":
        texts = [
            f"A cell emits {a} spikes in {b} seconds. Find mean firing rate in Hz.",
            f"Count={a} events; duration={b} s. Compute the event frequency.",
            f"In a {b}-second recording, {a} neural spikes occurred. What is the rate per second?",
            f"An observation lasts {b} seconds and registers {a} events. Give events/second.",
        ]
        answer = float(D(a) / D(b))
        expr = f"{a}/{b}"
    elif family == "mean":
        texts = [
            f"Replicate values are {a}, {b}, {c}. Compute their arithmetic mean.",
            f"Average the three observations [{a}, {b}, {c}].",
            f"Three equally weighted readings are {a}, {b}, {c}. Report their mean.",
            f"A measurement series contains {a}; {b}; {c}. What is sum divided by count?",
        ]
        answer = float((D(a) + D(b) + D(c)) / 3)
        expr = f"({a}+{b}+{c})/3"
    elif family == "difference":
        texts = [
            f"After treatment the value is {a}, before it was {b}. Compute after minus before.",
            f"Find the signed difference: experiment {a} minus control {b}.",
            f"Baseline={b}, follow-up={a}. Report the change follow-up minus baseline.",
            f"Signal B is {b} units; signal A is {a}. Calculate A-B, retaining the sign.",
        ]
        answer = a - b
        expr = f"{a}-{b}"
    elif family == "weighted":
        texts = [
            f"A group of {d} observations has mean {a}; a group of 10 has mean {b}. Find the combined mean.",
            f"Pool two samples: n1={d}, mean1={a}, n2=10, mean2={b}. Compute overall mean.",
            f"{d} measurements average {a}, while ten others average {b}. What is the pooled average?",
            f"Combine a size-{d} sample averaging {a} with a size-10 sample averaging {b}. Give the aggregate mean.",
        ]
        answer = float((D(d) * D(a) + 10 * D(b)) / (d + 10))
        expr = f"({d}*{a}+10*{b})/({d}+10)"
    elif family == "bayes":
        texts = [
            f"Among positive tests, {a} are true positives and {b} false positives. Find P(disease|positive).",
            f"A positive-test subgroup contains {a} diseased and {b} healthy individuals. Give disease fraction.",
            f"Positive results comprise {b} unaffected and {a} affected people. What probability of being affected follows for this subgroup?",
            f"{a} correct detections and {b} false alarms form all positive tests. Calculate positive predictive value as a fraction, not percent.",
        ]
        answer = float(D(a) / (a + b))
        expr = f"{a}/({a}+{b})"
    elif family == "missing_rate":
        texts = [
            f"{a} spikes occurred, but duration is unavailable. Determine the numerical rate in Hz.",
            f"Only event count {a} is known; no observation time. Give events per second.",
            f"The counter recorded {a} events during an unspecified interval. Find their numerical frequency.",
            f"Total spikes: {a}. The timing metadata is missing. What is the mean firing rate?",
        ]
        answer = None
        expr = None
    elif family == "causality":
        # Balanced support; causality itself not a constant-answer shortcut.
        supported = i % 2 == 0
        texts = [
            f'In an observational study of {a + b} subjects, X and Y are associated, with no confounding control. Is the claim "{claim}" supported?'
            for claim in [("X and Y are associated in this sample" if supported else "X causes Y")] * 4
        ]
        texts[1] = texts[1].replace("observational study", "non-randomised survey")
        texts[2] = texts[2].replace("subjects", "participants").replace("associated,", "correlated,")
        texts[3] = (
            texts[3]
            .replace("In an observational study of", "Observations from")
            .replace("subjects,", "cases show that")
        )
        answer = "supported" if supported else "unsupported"
        expr = answer
    elif family == "evidence":
        supported = i % 2 == 0
        threshold = a - 2 if supported else a + 2
        texts = [
            f'The reported sample mean is {a}. Is the claim "sample mean exceeds {threshold}" supported?',
            f"Observed average={a}. Does that support average > {threshold}?",
            f"A sample average of {a} was measured. Assess the statement that this average is above {threshold}.",
            f'The data summary gives mean {a}. Is "mean greater than {threshold}" consistent with this summary?',
        ]
        answer = "supported" if supported else "unsupported"
        expr = answer
    elif family == "decay":
        texts = [
            f"A substance starts at {a} mg and halves each hour. Find the remaining mass after {d} hours."
        ] * 4
        answer = float(D(a) / (D(2) ** d))
        expr = f"{a}/2**{d}"
    elif family == "dilution":
        texts = [
            f"{a} mL of a {d} mg/mL solution is diluted to {a + b} mL with solute conserved. Find the final concentration in mg/mL."
        ] * 4
        answer = float(D(a) * d / (a + b))
        expr = f"{a}*{d}/({a}+{b})"
    elif family == "zscore":
        texts = [
            f"A value is {a}, reference mean {b}, reference standard deviation {d}. Calculate z=(value-mean)/standard deviation."
        ] * 4
        answer = float((D(a) - D(b)) / d)
        expr = f"({a}-{b})/{d}"
    else:
        texts = [
            f"The sample has {a} observations and mean {b}. No spread statistic is given. Determine its numerical standard deviation."
        ] * 4
        answer = None
        expr = None
    prompt = texts[variant]
    # Numerical-instance key includes canonical inputs, independent of wording.
    import re

    key = family + ":" + ",".join(sorted(re.findall(r"\d+", prompt)))
    wrong = (
        answer + (-1 if i % 2 else 1) * max(1, abs(answer) * 0.2)
        if type(answer) in (int, float)
        else float(c)
        if answer is None
        else "unsupported"
        if answer == "supported"
        else "supported"
    )
    return {
        "id": hashlib.sha256(prompt.encode()).hexdigest()[:20],
        "family": family,
        "split": split,
        "distribution": "ood" if family in OOD else "id",
        "instance_key": key,
        "prompt": prompt,
        "answer": answer,
        "oracle_expression": expr,
        "chosen": response(round(answer, 7) if type(answer) in (int, float) else answer),
        "rejected": response(round(wrong, 7) if type(wrong) in (int, float) else wrong),
        "source": VERSION,
    }


def prepare():
    root = Path(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {"version": VERSION, "license": "CC0-1.0", "seed": 20261108, "splits": {}}
    for split, n in [("train", 48), ("validation", 8), ("test", 12), ("final", 12)]:
        rng = random.Random(f"20261108-{split}")
        rows = []
        for family in TRAIN + (OOD if split in ("test", "final") else []):
            for i in range(n):
                row = make_example(family, split, i, rng)
                while row["instance_key"] in seen:
                    row = make_example(family, split, i, rng)
                seen.add(row["instance_key"])
                rows.append(row)
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (root / f"{split}.jsonl").write_text(text)
        manifest["splits"][split] = {"n": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()}
    dump(root / "manifest.json", manifest)
    dump("experiments/toolstudy/data_manifest.json", manifest)


def resolve(text, mode):
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) != {"answer", "abstain"}:
            raise ValueError("schema")
        used = False
        if (
            mode == "calculator"
            and isinstance(obj["answer"], str)
            and obj["answer"] not in ("supported", "unsupported")
        ):
            obj["answer"] = calculate(obj["answer"])
            used = True
        return json.dumps(obj), used, None
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError) as exc:
        return text, False, type(exc).__name__


def evaluate_policy(name, adapter=None):
    set_seed(42)
    model, tok, device = load(adapter)
    model.eval()
    gen = json.loads(Path("configs/generation.json").read_text())
    gen["max_new_tokens"] = 64
    for split in ["test", "final"]:
        rows = [json.loads(s) for s in Path(ROOT, f"{split}.jsonl").read_text().splitlines()]
        for mode in ["direct", "calculator"]:
            start = time.perf_counter()
            records = []
            path = Path(f"experiments/toolstudy/{name}_{split}_{mode}.jsonl")
            with path.open("w") as f:
                for i, row in enumerate(rows):
                    # Only system instruction and problem text cross the model boundary.
                    msgs = [
                        {"role": "system", "content": SYSTEM if mode == "direct" else TOOL_SYSTEM},
                        {"role": "user", "content": row["prompt"]},
                    ]
                    inp = tok(
                        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                        return_tensors="pt",
                    ).to(device)
                    with torch.inference_mode():
                        out = model.generate(**inp, **gen)
                    pred = tok.decode(out[0, inp.input_ids.shape[1] :], skip_special_tokens=True)
                    resolved, used, error = resolve(pred, mode)
                    rec = {
                        **row,
                        "prediction": pred,
                        "resolved": resolved,
                        "tool_used": used,
                        "tool_error": error,
                        "generated_tokens": int(out.shape[1] - inp.input_ids.shape[1]),
                        "grade": grade(resolved, row["answer"]),
                    }
                    records.append(rec)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if i % 36 == 0:
                        print(name, split, mode, i, flush=True)
            dump(
                path.with_suffix(".summary.json"),
                {
                    "status": "COMPLETED",
                    "metrics": summarise(records),
                    "elapsed_seconds": time.perf_counter() - start,
                    "generation": gen,
                    "hardware": hardware(),
                    "memory": memory(),
                    "model": config(),
                    "adapter": adapter,
                    "dataset": json.loads(Path(ROOT, "manifest.json").read_text()),
                    "tool_calls": sum(r["tool_used"] for r in records),
                    "tool_errors": sum(r["tool_error"] is not None for r in records),
                },
            )


def analyse():
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["baseline"] + [f"{kind}_{seed}" for seed in [42, 43, 44] for kind in ["sft", "dpo"]]
    tables = {}
    paired = {}
    for split in ["test", "final"]:
        rows = []
        for name in names:
            pair = []
            for mode in ["direct", "calculator"]:
                p = Path(f"experiments/toolstudy/{name}_{split}_{mode}.jsonl")
                records = [json.loads(s) for s in p.read_text().splitlines()]
                pair.append(records)
                metrics = summarise(records)
                rows.append({"policy": name, "mode": mode, **metrics})
            paired[f"{name}_{split}"] = paired_ci(*pair)
        tables[split] = rows
        fig, ax = plt.subplots(figsize=(10, 4))
        x = list(range(len(names)))
        direct = [r["accuracy"] for r in rows if r["mode"] == "direct"]
        calc = [r["accuracy"] for r in rows if r["mode"] == "calculator"]
        ax.bar([i - 0.18 for i in x], direct, 0.36, label="Direct")
        ax.bar([i + 0.18 for i in x], calc, 0.36, label="Program + calculator")
        ax.set_xticks(x, names, rotation=25)
        ax.set(ylim=(0, 1), ylabel="Verified accuracy", title=f"Fresh scientific benchmark: {split}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"reports/toolstudy/{split}.png", dpi=160)
        plt.close(fig)
    dump("reports/toolstudy/metrics.json", tables)
    dump("reports/toolstudy/paired.json", paired)
    lines = [
        "# Verified computation study — measured results",
        "",
        "Original scireason-program-v1 data; 384 train, 64 validation, 144 primary and 144 final tasks. Four families held out. Three SFT/DPO training seeds. All policies use greedy decoding, one output and at most one bounded calculator execution; 64 output tokens. No oracle expression is shown to the model.",
        "",
        "The intervention changes both the output instruction and available executor. It is a program-aided pipeline comparison, not an isolated causal effect of a calculator with identical prompts. No multi-turn tool agent or constrained decoding is claimed. OOD final shares templates with OOD primary, with disjoint numerical conditions.",
    ]
    for split, rs in tables.items():
        lines += ["", f"## {split}", "| Policy | Mode | Accuracy | Numeric accuracy |", "|---|---|---:|---:|"]
        lines += [
            f"| {r['policy']} | {r['mode']} | {r['accuracy']:.3f} | {r['numeric_accuracy']:.3f} |" for r in rs
        ]
    lines += [
        "",
        "## Limitations",
        "Small artificial single-step tasks; three seeds do not establish model-family generality. Correct execution does not imply correct programme construction. Base is already instruction-tuned. Numerical conditions and family-level holdouts reduce leakage, but pretraining contamination is unmeasured. All previously published v2 results remain separate. Paired example-bootstrap intervals are conditional on the task set; do not pool three seeds as independent test problems.",
        "",
        "Reproduce: `bash scripts/run_toolstudy.sh`. Training histories are in experiments/toolstudy_*_training.json; per-example responses, programmes and tool errors are retained in experiments/toolstudy/.",
    ]
    Path("reports/toolstudy/research_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "evaluate", "analyse"])
    p.add_argument("--name")
    p.add_argument("--adapter")
    a = p.parse_args()
    if a.action == "prepare":
        prepare()
    elif a.action == "evaluate":
        evaluate_policy(a.name, a.adapter)
    else:
        analyse()
