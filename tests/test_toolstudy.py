import json
import random

import pytest

from scireason.toolstudy import OOD, TRAIN, calculate, make_example, resolve


def test_safe_arithmetic():
    assert calculate("(12+6)/3") == 6
    assert calculate("sqrt(9)") == 3
    for bad in [
        '__import__("os").system("id")',
        "[1]*100000",
        "2**1000000",
        "1/0",
        'float("nan")',
        "(1).__class__",
        "True",
    ]:
        with pytest.raises((ValueError, ZeroDivisionError, SyntaxError)):
            calculate(bad)


def test_independent_oracle():
    for family in TRAIN + OOD:
        for i in range(12):
            r = make_example(family, "test", i, random.Random(i))
            if type(r["answer"]) in (int, float):
                assert calculate(r["oracle_expression"]) == pytest.approx(r["answer"], rel=1e-12)


def test_tool_resolution():
    text = '{"answer":"12/3","abstain":false}'
    result, used, error = resolve(text, "calculator")
    assert json.loads(result)["answer"] == 4 and used and error is None
    assert json.loads(resolve(text, "direct")[0]) == json.loads(text)


def test_generated_split_isolation(tmp_path, monkeypatch):
    from pathlib import Path

    from scireason import toolstudy

    monkeypatch.chdir(tmp_path)
    toolstudy.prepare()
    seen = set()
    for split, count in [("train", 384), ("validation", 64), ("test", 144), ("final", 144)]:
        rows = [json.loads(x) for x in Path(toolstudy.ROOT, split + ".jsonl").read_text().splitlines()]
        assert len(rows) == count
        keys = {r["instance_key"] for r in rows}
        assert len(keys) == len(rows) and not keys.intersection(seen)
        seen.update(keys)
        if split in ("train", "validation"):
            assert not {r["family"] for r in rows}.intersection(toolstudy.OOD)
        from scireason.metrics import grade

        assert all(grade(r["chosen"], r["answer"])["correct"] for r in rows)
        assert not any(grade(r["rejected"], r["answer"])["correct"] for r in rows)


def test_experiment_config_learning_rates_are_numeric():
    from pathlib import Path

    import yaml

    for path in Path("configs/toolstudy").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        assert isinstance(cfg["learning_rate"], float) and 0 < cfg["learning_rate"] < 1
