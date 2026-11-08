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
