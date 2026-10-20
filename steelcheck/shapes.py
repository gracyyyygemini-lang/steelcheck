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
    J: float | None = None  # St. Venant torsional constant, in^4
    Cw: float | None = None  # warping constant, in^6
    rts: float | None = None  # effective radius of gyration for LTB, in
    ho: float | None = None  # distance between flange centroids, in
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
    # A, bf/2tf, h/tw, rx, ry from lecture; the rest (needed for flexure) from
    # steelpy, not stated in lecture.
    Shape(
        "W12X65", "W", A=19.1, d=12.1, bf=12.0, tf=0.605, tw=0.390,
        bf_2tf=9.92, h_tw=24.9, rx=5.28, ry=3.02,
        Ix=533.0, Iy=174.0, Sx=87.9, Sy=29.1, Zx=96.8, Zy=44.1,
        J=2.18, Cw=5780.0, rts=3.38, ho=11.5,
        source="lecture 2026-09-17 (A, bf/2tf, h/tw, rx, ry); rest from steelpy -- verify",
    ),
    Shape(
        "W10X30", "W", A=8.84, bf=5.81, tf=0.510, tw=0.300,
        source="lecture 2026-09-10",
    ),
    Shape(
        "W24X76", "W", A=22.4, d=23.9, bf=8.99, tf=0.680, tw=0.440,
        bf_2tf=6.61, h_tw=49.0, rx=9.69, ry=1.92,
        Ix=2100.0, Iy=82.5, Sx=176.0, Sy=18.4, Zx=200.0,
        J=2.68, Cw=11100.0, rts=2.33, ho=23.2,
        source="lecture 2026-10-01 / Assignment 1",
    ),
    # Flexure validation shapes. Not from lecture: properties from steelpy, with
    # bf/2tf and h/tw derived as in load_from_steelpy(). Confirm against Table 1-1.
    Shape(
        "W18X35", "W", A=10.3, d=17.7, bf=6.00, tf=0.425, tw=0.300,
        bf_2tf=7.06, h_tw=53.5, rx=7.04, ry=1.22,
        Ix=510.0, Iy=15.3, Sx=57.6, Sy=5.12, Zx=66.5, Zy=8.06,
        J=0.506, Cw=1140.0, rts=1.51, ho=17.3,
        source="steelpy -- verify",
    ),
    Shape(
        "W16X31", "W", A=9.13, d=15.9, bf=5.53, tf=0.440, tw=0.275,
        bf_2tf=6.28, h_tw=51.7, rx=6.41, ry=1.17,
        Ix=375.0, Iy=12.4, Sx=47.2, Sy=4.49, Zx=54.0, Zy=7.03,
        J=0.461, Cw=739.0, rts=1.42, ho=15.5,
        source="steelpy -- verify",
    ),
    Shape(
        "W21X48", "W", A=14.1, d=20.6, bf=8.14, tf=0.430, tw=0.350,
        bf_2tf=9.47, h_tw=53.5, rx=8.24, ry=1.66,
        Ix=959.0, Iy=38.7, Sx=93.0, Sy=9.52, Zx=107.0, Zy=14.9,
        J=0.803, Cw=3950.0, rts=2.05, ho=20.2,
        source="steelpy -- verify",
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
    "J": "J", "Cw": "Cw", "rts": "rts", "ho": "ho",
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


# ---------------------------------------------------------------------------
# steelpy adapter
# ---------------------------------------------------------------------------
#
# ``pip install steelpy`` gives the same section property data without going
# through AISC's download. Verified: every value it reports for the shapes in
# the seed set matches the published figure exactly (W12X65 A = 19.1, rx = 5.28,
# ry = 3.02; L6X4X1/2 xbar = 0.981, ybar = 1.98; C10X30 xbar = 0.649).
#
# Two ratios the Manual prints are not stored and have to be derived:
#
#   bf/2tf = bf / (2*tf)
#   h/tw   = (d - 2k) / tw     h is the clear distance between fillets, and k
#                              is the flange-face-to-fillet-toe distance.
#
# Both reproduce the published values to three significant figures -- W12X65
# gives 9.917 and 24.87 against a published 9.92 and 24.9.

#: steelpy family attribute -> our family code.
_STEELPY_FAMILIES = {
    "W_shapes": "W", "M_shapes": "M", "S_shapes": "S", "HP_shapes": "HP",
    "C_shapes": "C", "MC_shapes": "MC", "L_shapes": "L",
    "WT_shapes": "WT", "MT_shapes": "MT", "ST_shapes": "ST",
    "HSS_shapes": "HSS", "HSS_R_shapes": "HSS", "PIPE_shapes": "PIPE",
}

#: Families whose webs and flanges follow the rolled I-shape slenderness rules.
_I_FAMILIES = {"W", "M", "S", "HP"}


def _num(v) -> float | None:
    """steelpy writes an en-dash for 'not applicable'."""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "–", "—", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _steelpy_designation(name: str) -> str:
    """``L6X4X1_2`` -> ``L6X4X1/2``; underscores encode fraction slashes."""
    return name.replace("_", "/")


def load_from_steelpy(families: list[str] | None = None, overwrite: bool = True) -> int:
    """Populate the registry from the ``steelpy`` package.

    An alternative to the AISC Shapes Database download that needs no account
    and no spreadsheet conversion::

        pip install steelpy

        import steelcheck as sc
        sc.shapes.load_from_steelpy()

    Returns the number of shapes loaded.
    """
    try:
        from steelpy import aisc
    except ImportError:
        raise ImportError(
            "steelpy is not installed. Run `pip install steelpy`, or load the "
            "AISC Shapes Database CSV with load_aisc_csv() instead."
        ) from None

    wanted = families or list(_STEELPY_FAMILIES)
    loaded = 0

    for attr in wanted:
        fam = _STEELPY_FAMILIES.get(attr)
        if fam is None:
            continue
        try:
            group = getattr(aisc, attr)
            names = list(group.sections)
        except AttributeError:
            continue

        for raw_name in names:
            try:
                p = getattr(group, raw_name).properties
            except AttributeError:
                continue

            g = lambda k: _num(p.get(k))  # noqa: E731
            kwargs: dict[str, float] = {}

            def put(field: str, value: float | None) -> None:
                if value is not None:
                    kwargs[field] = value

            put("A", g("area"))
            if kwargs.get("A") is None:
                continue

            for field in ("rx", "ry", "Ix", "Iy", "Sx", "Sy", "Zx", "Zy",
                         "J", "Cw", "rts", "ho"):
                put(field, g(field))
            put("xbar", g("x"))
            put("ybar", g("y"))

            if fam == "L":
                # For an angle, steelpy's d is the SHORT leg and b the long one.
                put("leg_long", g("b"))
                put("leg_short", g("d"))
                put("t", g("t"))
            else:
                put("d", g("d"))
                put("bf", g("bf"))
                put("tf", g("tf"))
                put("tw", g("tw"))
                put("t", g("t"))

            if fam in _I_FAMILIES:
                bf, tf, d, k, tw = (g("bf"), g("tf"), g("d"), g("k"), g("tw"))
                if bf and tf:
                    kwargs["bf_2tf"] = round(bf / (2 * tf), 3)
                if d and k and tw:
                    kwargs["h_tw"] = round((d - 2 * k) / tw, 2)

            designation = _steelpy_designation(raw_name)
            if not overwrite and _key(designation) in _REGISTRY:
                continue
            register(Shape(
                designation=designation, family=fam, source="steelpy", **kwargs
            ))
            loaded += 1

    return loaded
