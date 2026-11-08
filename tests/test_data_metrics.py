import json
import math

from scireason.data import FAMILIES, OOD, prepare, read
from scireason.metrics import grade, paired_ci


def test_reproducible_disjoint_data(tmp_path):
    first = prepare(tmp_path)
    assert first == prepare(tmp_path)
    rows = {s: read(s, tmp_path) for s in first["splits"]}
    ids = [r["id"] for split in rows.values() for r in split]
    assert len(ids) == len(set(ids))
    assert {r["family"] for r in rows["train"]} == set(FAMILIES)
    assert not set(OOD) & {r["family"] for r in rows["train"]}
    assert not {r["template"] for r in rows["train"]} & {r["template"] for r in rows["validation"]}
    for split in rows.values():
        for r in split:
            assert grade(r["chosen"], r["answer"])["correct"]
            assert not grade(r["rejected"], r["answer"])["correct"]


def test_numeric_tolerance_and_types():
    assert grade('{"answer":0.333333,"abstain":false}', 1 / 3)["correct"]
    assert not grade('{"answer":true,"abstain":false}', 1)["correct"]
    assert not grade('{"answer":1.1,"abstain":false}', 1)["correct"]
    assert not grade('{"answer":1,"abstain":true}', 1)["correct"]
    assert not grade("answer: 1", 1)["format_valid"]
    assert not grade('{"answer":NaN,"abstain":false}', 1)["correct"]


def test_abstention_and_evidence():
    assert grade('{"answer":null,"abstain":true}', None)["correct"]
    assert not grade('{"answer":null,"abstain":false}', None)["correct"]
    assert grade('{"answer":"supported","abstain":false}', "unsupported")["unsupported_assertion"]


def test_paired_identical():
    rows = [{"id": str(i), "grade": {"correct": i % 2 == 0}} for i in range(10)]
    assert paired_ci(rows, rows)["ci95"] == [0, 0]


def test_independent_rate_oracle(tmp_path):
    import re

    prepare(tmp_path)
    for r in read("train", tmp_path):
        if r["family"] == "rate":
            count, duration = map(int, re.findall(r"\d+", r["prompt"]))
            assert math.isclose(r["answer"] * duration, count)
        assert set(json.loads(r["chosen"])) == {"answer", "abstain"}
