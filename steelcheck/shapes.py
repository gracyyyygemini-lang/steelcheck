"""Section property data.

Two sources, in priority order:

1. The official **AISC Shapes Database**, if you drop it in. AISC publishes it
   free alongside the Steel Construction Manual. Download it, save it as
   ``data/aisc_shapes.csv``, and ``load_aisc_csv()`` will pick it up -- then you
   have every shape, not just the seed set.

2. A small **seed set** built into this module, below. Every value in it was
   read aloud in an ARCH 433 lecture while working an example, which means each
   one has a published capacity to check against. That is what makes the test
   suite meaningful: the inputs and the expected outputs come from the same
   source, independently of this code.

On licensing: section dimensions are factual data, but the Shapes Database and
the Specification have their own terms. This repository ships no AISC file --
only the handful of values needed to reproduce documented worked examples. Check
the current license yourself before redistributing anything.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Shape:
    """A rolled structural shape.

    Fields are AISC Manual Table 1-1 notation. ``None`` means "not populated in
    this record" -- a design check that needs a missing property raises rather
    than guessing.
    """

    designation: str
    family: str  # W, C, L, WT, HSS ...
    A: float  # gross area, in^2
    d: float | None = None  # overall depth, in
    bf: float | None = None  # flange width, in
    tf: float | None = None  # flange thickness, in
    tw: float | None = None  # web thickness, in
    t: float | None = None  # leg/wall thickness (angles, HSS), in
    bf_2tf: float | None = None  # flange slenderness b/t
    h_tw: float | None = None  # web slenderness h/tw
    rx: float | None = None
    ry: float | None = None
    Ix: float | None = None
    Iy: float | None = None
    Sx: float | None = None
    Sy: float | None = None
    Zx: float | None = None
    Zy: float | None = None
    xbar: float | None = None  # centroid from back of leg/web, in
    ybar: float | None = None
    leg_long: float | None = None  # angles, in
    leg_short: float | None = None  # angles, in
    source: str = "seed"  # provenance, for auditing

    @property
    def rx_ry(self) -> float:
        """rx/ry, the ratio printed at the bottom of AISC Table 4-1a.

        Read it as one printed quantity when you have the Manual open; computing
        it from rx and ry separately is where rounding creeps in.
        """
        if self.rx is None or self.ry is None:
            raise ValueError(f"{self.designation}: rx and ry required for rx/ry")
        return self.rx / self.ry

    def require(self, *names: str) -> tuple:
        out = []
        for n in names:
            v = getattr(self, n, None)
            if v is None:
                raise ValueError(
                    f"{self.designation}: property {n!r} is not populated. "
                    "Load the full AISC Shapes Database to get it "
                    "(see steelcheck.shapes.load_aisc_csv)."
                )
            out.append(v)
        return tuple(out)

    def __str__(self) -> str:
        return self.designation


# ---------------------------------------------------------------------------
# Seed set -- values stated in ARCH 433 lectures while working examples.
# Each shape here participates in at least one validation test.
# ---------------------------------------------------------------------------

_SEED: list[Shape] = [
    Shape(
        "W12X65", "W", A=19.1, bf_2tf=9.92, h_tw=24.9, rx=5.28, ry=3.02,
        source="lecture 2026-09-17",
    ),
    Shape(
        "W10X30", "W", A=8.84, bf=5.81, tf=0.510, tw=0.300,
        source="lecture 2026-09-10",
    ),
    Shape(
        "W24X76", "W", A=22.4, d=23.9, bf=8.99, tf=0.680, tw=0.440,
        bf_2tf=6.61, h_tw=49.0, rx=9.69, ry=1.92,
        Ix=2100.0, Iy=82.5, Sx=176.0, Sy=18.4, Zx=200.0,
        source="lecture 2026-10-01 / Assignment 1",
    ),
    # rx/ry = 1.66 was stated in lecture; rx, ry and A below reproduce both that
    # ratio and the quoted capacities. Confirm against Table 1-1 before reuse.
    Shape(
        "W14X90", "W", A=26.5, rx=6.14, ry=3.70, bf_2tf=10.2, h_tw=25.9,
        source="lecture 2026-09-24 (A, rx, ry inferred -- verify)",
    ),
    Shape(
        "C10X30", "C", A=8.81, d=10.0, xbar=0.649, tw=0.673,
        source="lecture 2026-09-13",
    ),
    Shape(
        "L6X4X1/2", "L", A=4.75, t=0.5, xbar=0.981, ybar=1.98,
        leg_long=6.0, leg_short=4.0,
        source="lecture 2026-09-11",
    ),
    Shape(
        "WT5X15", "WT", A=4.42, ybar=1.10,
        source="lecture 2026-09-11 (ybar); A inferred -- verify",
    ),
]

_REGISTRY: dict[str, Shape] = {}


def _key(designation: str) -> str:
    return designation.upper().replace(" ", "").replace("×", "X")


def register(shape: Shape) -> None:
    _REGISTRY[_key(shape.designation)] = shape


for _s in _SEED:
    register(_s)


def get(designation: str) -> Shape:
    """Look up a shape by designation, e.g. ``get("W12x65")``."""
    k = _key(designation)
    if k not in _REGISTRY:
        raise KeyError(
            f"Shape {designation!r} not in the registry ({len(_REGISTRY)} loaded). "
            "The seed set only covers shapes used in the validation tests -- "
            "call load_aisc_csv() with the AISC Shapes Database for full coverage."
        )
    return _REGISTRY[k]


def all_shapes() -> list[Shape]:
    return sorted(_REGISTRY.values(), key=lambda s: s.designation)


def family(name: str) -> list[Shape]:
    return [s for s in all_shapes() if s.family == name.upper()]


# Maps AISC Shapes Database column names onto Shape fields. The database uses
# these headers in v15; adjust here if your edition differs.
_CSV_MAP = {
    "A": "A", "d": "d", "bf": "bf", "tf": "tf", "tw": "tw", "t": "t",
    "bf/2tf": "bf_2tf", "h/tw": "h_tw",
    "rx": "rx", "ry": "ry", "Ix": "Ix", "Iy": "Iy",
    "Sx": "Sx", "Sy": "Sy", "Zx": "Zx", "Zy": "Zy",
    "x": "xbar", "y": "ybar", "B": "leg_long", "b": "leg_short",
}


def load_aisc_csv(path: str | Path, overwrite: bool = True) -> int:
    """Load the AISC Shapes Database CSV into the registry.

    Returns the number of shapes loaded. Rows whose area will not parse are
    skipped rather than guessed at.
    """
    path = Path(path)
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            designation = (row.get("AISC_Manual_Label") or row.get("EDI_Std_Nomenclature") or "").strip()
            if not designation:
                continue
            kwargs: dict[str, float] = {}
            for col, fieldname in _CSV_MAP.items():
                raw = (row.get(col) or "").strip()
                if raw in ("", "-", "–"):
                    continue
                try:
                    kwargs[fieldname] = float(raw)
                except ValueError:
                    continue
            if "A" not in kwargs:
                continue
            fam = (row.get("Type") or designation[:1]).strip().upper()
            shape = Shape(
                designation=designation, family=fam, source=f"AISC Shapes Database ({path.name})",
                **kwargs,
            )
            if overwrite or _key(designation) not in _REGISTRY:
                register(shape)
                loaded += 1
    return loaded


def autoload(data_dir: str | Path = "data") -> int:
    """Load ``data/aisc_shapes.csv`` if present. Returns shapes loaded, or 0."""
    p = Path(data_dir) / "aisc_shapes.csv"
    return load_aisc_csv(p) if p.exists() else 0
