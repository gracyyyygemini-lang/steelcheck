"""Calculation traces.

The point of this module: every number this library produces carries a record of
where it came from. An engineer stamping a drawing is personally liable, so a
bare number from a black box is worthless to them. A number with the equation,
the table, and the intermediate steps is reviewable.

This is also what makes the LLM layer safe: the model never computes anything,
it only reads traces that the deterministic engine produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """One line of a hand calculation."""

    label: str
    value: float | str | None = None
    unit: str = ""
    ref: str = ""
    detail: str = ""

    def __str__(self) -> str:
        if self.value is None:
            head = self.label
        elif isinstance(self.value, str):
            head = f"{self.label} = {self.value}"
        else:
            head = f"{self.label} = {_sig(self.value)}{(' ' + self.unit) if self.unit else ''}"
        if self.detail:
            head = f"{head}   [{self.detail}]"
        if self.ref:
            head = f"{head}   ({self.ref})"
        return head


@dataclass
class Trace:
    """An ordered list of steps plus the conclusion."""

    title: str
    steps: list[Step] = field(default_factory=list)

    def add(
        self,
        label: str,
        value: float | str | None = None,
        unit: str = "",
        ref: str = "",
        detail: str = "",
    ) -> "Trace":
        self.steps.append(Step(label, value, unit, ref, detail))
        return self

    def extend(self, other: "Trace", prefix: str = "") -> "Trace":
        if prefix:
            self.add(prefix)
        self.steps.extend(other.steps)
        return self

    def render(self) -> str:
        lines = [self.title, "=" * len(self.title)]
        lines += [f"  {s}" for s in self.steps]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _sig(x: float, n: int = 3) -> str:
    """Round to n significant figures the way the AISC Manual does.

    Three significant figures is the convention taught in ARCH 433 and used
    throughout the Manual, so results print the way a grader expects to see them.
    """
    if x == 0:
        return "0"
    if not _finite(x):
        return str(x)
    from math import floor, log10

    mag = floor(log10(abs(x)))
    digits = n - 1 - mag
    r = round(x, digits)
    if digits <= 0:
        return f"{int(r):,}"
    return f"{r:,.{digits}f}"


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def sig(x: float, n: int = 3) -> float:
    """Numeric three-significant-figure rounding (for comparisons, not display)."""
    if x == 0:
        return 0.0
    from math import floor, log10

    mag = floor(log10(abs(x)))
    return round(x, n - 1 - mag)
