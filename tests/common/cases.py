"""Typed loaders for the Coverage and Correctness case configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
from typing import Literal


Mode = Literal["forward", "backward"]
Pattern = Literal[
    "random",
    "ramp",
    "precision_triplet",
    "large_values",
    "cancellation",
]

DOCUMENTED_DTYPES = ("float16", "bfloat16")
CASE_ROOT = Path(__file__).resolve().parents[1] / "st/cases"


@dataclass(frozen=True)
class Case:
    name: str
    mode: Mode
    dtype: str
    s: int
    d: int
    mhc_mult: int
    pattern: Pattern
    seed: int
    tags: tuple[str, ...]
    suite: str

    @property
    def backward(self) -> bool:
        return self.mode == "backward"

    @property
    def input_shape(self) -> tuple[int, ...]:
        if self.mode == "backward":
            return (self.s, self.mhc_mult, self.d)
        return (self.s, self.d)

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.mode == "backward":
            return (self.s, self.d)
        return (self.s, self.mhc_mult, self.d)

    @property
    def largest_tensor_numel(self) -> int:
        return max(prod(self.input_shape), prod(self.output_shape))


def _load_document(suite: str) -> dict:
    path = CASE_ROOT / f"{suite}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("category") != suite or not isinstance(document.get("cases"), list):
        raise ValueError(f"invalid {suite} case configuration: {path}")
    return document


def _load_cases(suite: str) -> tuple[Case, ...]:
    result = []
    for item in _load_document(suite)["cases"]:
        values = dict(item)
        values["tags"] = tuple(values.get("tags", ()))
        result.append(Case(**values, suite=suite))
    return tuple(result)


COVERAGE_CASES = _load_cases("coverage")
CORRECTNESS_CASES = _load_cases("correctness")
CASE_GROUPS = {
    "coverage": COVERAGE_CASES,
    "correctness": CORRECTNESS_CASES,
}
CASES = COVERAGE_CASES + CORRECTNESS_CASES

_case_by_name = {case.name: case for case in CASES}
if len(_case_by_name) != len(CASES):
    raise ValueError("case names must be unique")


def accepted_cases(suite: str = "full") -> tuple[Case, ...]:
    if suite in CASE_GROUPS:
        return CASE_GROUPS[suite]
    if suite == "full":
        return CASES
    raise ValueError(f"unknown suite: {suite}")


def case_by_name(name: str) -> Case:
    try:
        return _case_by_name[name]
    except KeyError as exc:
        raise KeyError(name) from exc
