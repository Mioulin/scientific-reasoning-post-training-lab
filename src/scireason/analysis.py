import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import paired_ci, summarise
from .runtime import dump


def analyse(split="test"):
    root = Path("experiments")
    names = ["baseline", "sft", "dpo", "dpo_beta05"]
    results = {
        name: [json.loads(s) for s in (root / f"{name}_{split}.jsonl").read_text().splitlines()]
        for name in names
        if (root / f"{name}_{split}_summary.json").exists()
    }
    summary = {name: summarise(rows) for name, rows in results.items()}
    comparisons = {
        name: paired_ci(results["baseline"], rows) for name, rows in results.items() if name != "baseline"
    }
    if "dpo" in results and "sft" in results:
        comparisons["dpo_minus_sft"] = paired_ci(results["sft"], results["dpo"])
    if "dpo_beta05" in results and "dpo" in results:
        comparisons["beta05_minus_beta01"] = paired_ci(results["dpo"], results["dpo_beta05"])
    dump(f"reports/comparisons_{split}.json", comparisons)
    dump(f"reports/metrics_{split}.json", summary)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        summary.keys(),
        [r["accuracy"] for r in summary.values()],
        color=["#777777", "#3182bd", "#31a354", "#756bb1"][: len(summary)],
    )
    ax.set(ylim=(0, 1), ylabel="Verified answer accuracy", title=f"Frozen scientific tasks ({split})")
    for i, r in enumerate(summary.values()):
        ax.text(i, r["accuracy"] + 0.02, f"{r['accuracy']:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(f"reports/accuracy_{split}.png", dpi=180)
    plt.close(fig)
    if summary:
        families = list(next(iter(summary.values()))["by_family"])
        fig, ax = plt.subplots(figsize=(10, 5))
        data = np.array([[summary[n]["by_family"][f] for f in families] for n in summary])
        im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(families)), families, rotation=35, ha="right")
        ax.set_yticks(range(len(summary)), list(summary))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.0%}",
                    ha="center",
                    va="center",
                    color="white" if data[i, j] > 0.6 else "black",
                )
        fig.colorbar(im, ax=ax, label="Accuracy")
        fig.tight_layout()
        fig.savefig(f"reports/families_{split}.png", dpi=180)
        plt.close(fig)
    table = [
        "| Model | n | Accuracy | Numeric accuracy | Format validity | Abstention recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in summary.items():
        table.append(
            f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['numeric_accuracy']:.3f} | {r['format_valid_rate']:.3f} | {r['abstention_recall']:.3f} |"
        )
    Path(f"reports/results_{split}.md").write_text("\n".join(table) + "\n")
    failures = {}
    for name, rows in results.items():
        failures[name] = [
            {k: r[k] for k in ["id", "family", "prompt", "answer", "prediction", "grade"]}
            for r in rows
            if not r["grade"]["correct"]
        ][:10]
    dump(f"reports/failure_examples_{split}.json", failures)
    if "baseline" in results:
        regressions = {
            name: [
                {
                    "id": a["id"],
                    "family": a["family"],
                    "prompt": a["prompt"],
                    "baseline": a["prediction"],
                    "trained": b["prediction"],
                }
                for a, b in zip(results["baseline"], rows)
                if a["grade"]["correct"] and not b["grade"]["correct"]
            ]
            for name, rows in results.items()
            if name != "baseline"
        }
        dump(f"reports/regressions_{split}.json", regressions)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, name in zip(axes, ["sft", "dpo", "reward"]):
        path = root / f"{name}_training.json"
        if path.exists():
            history = json.loads(path.read_text())["log_history"]
            for key in ["loss", "eval_loss"]:
                points = [(r["step"], r[key]) for r in history if key in r]
                if points:
                    ax.plot(*zip(*points), label=key)
            ax.legend()
        ax.set(title=name, xlabel="Optimiser step", ylabel="Loss")
    fig.tight_layout()
    fig.savefig("reports/training_curves.png", dpi=180)
    plt.close(fig)
