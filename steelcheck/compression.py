"""AISC 360 Chapter E -- Design of Members for Compression.

The spine of a column check is always the same six moves:

1. Classify the section. Chapter E3 is titled "members *without* slender
   elements" -- if the flange or web exceeds lambda_r from Table B4.1a, E3 does
   not apply at all. This is a gate, not a formality.
2. Effective length Lc = K*L, **computed per unbraced segment**. A brace acts as
   a pin, so each segment reads its own two ends off Table C-A-7.1 and gets its
   own K. Keep the largest K*L in each direction -- the longest segment is not
   automatically the critical one.
3. Pick the critical direction. Lcx and Lcy are not comparable directly (one is
   divided by rx, the other by ry), so convert: Lcy_eq = Lcx / (rx/ry).
4. Slenderness ratio Lc/r, with **L in inches**.
5. Short or long: compare Lc/r against 4.71*sqrt(E/Fy). Short buckles
   inelastically (Eq. E3-2, anchored on Fy); long buckles elastically (Eq. E3-3,
   where Fy drops out entirely -- which is why a slender column gains nothing
   from higher grade steel).
6. Pn = Fn*Ag, on the GROSS area. Bolts fill their holes and bear across them,
   so compression deducts nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

from .materials import E, Material
from .shapes import Shape
from .trace import Trace

PHI_C, OMEGA_C = 0.90, 1.67

#: AISC Table C-A-7.1, keyed by (end A, end B) with ends sorted. The value is
#: (theoretical K, recommended design K). Always use the recommended value --
#: ideal end conditions are never achieved in a real structure, because even a
#: "fixed" base rotates a little.
K_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    ("fixed", "fixed"): (0.5, 0.65),       # case (a)
    ("fixed", "pinned"): (0.7, 0.80),      # case (b)
    ("pinned", "pinned"): (1.0, 1.0),      # case (d)
    ("fixed", "free"): (2.0, 2.10),        # case (e), sway permitted
    ("free", "pinned"): (2.0, 2.0),        # case (f)
}

_END_ALIASES = {
    "fix": "fixed", "fixed": "fixed", "f": "fixed",
    "pin": "pinned", "pinned": "pinned", "p": "pinned",
    "brace": "pinned", "braced": "pinned",  # a brace acts as a pin
    "free": "free",
}


def normalize_end(end: str) -> str:
    key = end.strip().lower()
    if key not in _END_ALIASES:
        raise ValueError(
            f"Unknown end condition {end!r}. Use fixed, pinned, braced or free."
        )
    return _END_ALIASES[key]


def K_for(end_a: str, end_b: str, use_recommended: bool = True) -> float:
    """Effective length factor from AISC Table C-A-7.1 (p. 16.1-650)."""
    a, b = normalize_end(end_a), normalize_end(end_b)
    key = (a, b) if (a, b) in K_TABLE else (b, a)
    if key not in K_TABLE:
        raise ValueError(f"No tabulated K for ends ({a}, {b}).")
    theoretical, recommended = K_TABLE[key]
    return recommended if use_recommended else theoretical


@dataclass
class Segment:
    """One unbraced length, with its own end conditions and therefore its own K."""

    length: float  # ft
    top: str = "pinned"
    bottom: str = "pinned"
    use_recommended: bool = True

    @property
    def K(self) -> float:
        return K_for(self.top, self.bottom, self.use_recommended)

    @property
    def Lc(self) -> float:
        """Effective length, ft."""
        return self.K * self.length

    def __str__(self) -> str:
        return (
            f"{self.length:g} ft, {normalize_end(self.top)}-"
            f"{normalize_end(self.bottom)}, K={self.K:g} -> Lc={self.Lc:g} ft"
        )


@dataclass
class Axis:
    """How the column is braced about one axis."""

    segments: list[Segment]
    name: str = ""

    @classmethod
    def unbraced(cls, length: float, top: str, bottom: str, name: str = "") -> "Axis":
        """No intermediate bracing: one segment spanning the full height."""
        return cls([Segment(length, top, bottom)], name)

    @classmethod
    def braced(
        cls,
        total_length: float,
        brace_points: list[float],
        top: str = "pinned",
        bottom: str = "fixed",
        name: str = "",
        measured_from: str = "bottom",
    ) -> "Axis":
        """Split a column at its braces.

        ``brace_points`` are distances to each brace, measured from the bottom by
        default. Each brace acts as a pin for the segments on either side of it,
        so an end segment that touches a fixed base is fixed-pinned (K = 0.80),
        not pinned-pinned.
        """
        pts = sorted(brace_points)
        if measured_from.lower().startswith("top"):
            pts = sorted(total_length - p for p in pts)
        if any(p <= 0 or p >= total_length for p in pts):
            raise ValueError("Brace points must lie strictly inside the column.")

        cuts = [0.0, *pts, total_length]
        segs: list[Segment] = []
        for i in range(len(cuts) - 1):
            lower = bottom if i == 0 else "pinned"
            upper = top if i == len(cuts) - 2 else "pinned"
            segs.append(Segment(cuts[i + 1] - cuts[i], top=upper, bottom=lower))
        return cls(segs, name)

    @property
    def Lc(self) -> float:
        """Governing effective length, ft -- the largest K*L, not the longest L."""
        return max(s.Lc for s in self.segments)

    def trace(self, t: Trace) -> None:
        label = f"{self.name} " if self.name else ""
        for i, s in enumerate(self.segments, 1):
            t.add(f"{label}segment {i}", str(s), "", "AISC Table C-A-7.1")
        t.add(f"{label}Lc (governing)", self.Lc, "ft", "", "largest K*L")


# ---------------------------------------------------------------------------
# Step 1 -- section classification gate
# ---------------------------------------------------------------------------

@dataclass
class SlendernessCheck:
    non_slender: bool
    flange_ratio: float | None
    flange_limit: float | None
    web_ratio: float | None
    web_limit: float | None
    trace: Trace


def classify(shape: Shape, mat: Material) -> SlendernessCheck:
    """Table B4.1a check for a rolled I-shape in uniform compression.

    Both limits scale with sqrt(E/Fy), so they TIGHTEN as the grade rises: a
    section that is non-slender in Gr. 50 can be slender-element in Gr. 80
    without a single dimension changing.
    """
    t = Trace(f"Section classification -- {shape} ({mat.name})")
    root = sqrt(E / mat.Fy)
    t.add("sqrt(E/Fy)", root, "", "", f"E = {E:,.0f} ksi, Fy = {mat.Fy:g} ksi")

    lam_f = 0.56 * root
    lam_w = 1.49 * root
    t.add("lambda_r flange = 0.56*sqrt(E/Fy)", lam_f, "", "AISC Table B4.1a Case 1")
    t.add("lambda_r web = 1.49*sqrt(E/Fy)", lam_w, "", "AISC Table B4.1a Case 5")

    ok = True
    bf_2tf, h_tw = shape.bf_2tf, shape.h_tw
    if bf_2tf is not None:
        passed = bf_2tf < lam_f
        ok &= passed
        t.add("bf/2tf", bf_2tf, "", "AISC Table 1-1",
              f"{'<' if passed else '>='} {lam_f:.1f} -- "
              f"{'non-slender' if passed else 'SLENDER'}")
    if h_tw is not None:
        passed = h_tw < lam_w
        ok &= passed
        t.add("h/tw", h_tw, "", "AISC Table 1-1",
              f"{'<' if passed else '>='} {lam_w:.1f} -- "
              f"{'non-slender' if passed else 'SLENDER'}")

    if bf_2tf is None and h_tw is None:
        t.add("classification", "NOT CHECKED", "", "",
              "bf/2tf and h/tw unavailable for this shape record")
        return SlendernessCheck(True, None, None, None, None, t)

    t.add("result", "non-slender element section" if ok else "SLENDER-ELEMENT",
          "", "AISC Sect. E3" if ok else "AISC Sect. E7")
    return SlendernessCheck(ok, bf_2tf, lam_f, h_tw, lam_w, t)


# ---------------------------------------------------------------------------
# Step 3 -- critical direction
# ---------------------------------------------------------------------------

@dataclass
class DirectionResult:
    governs: str  # 'x' or 'y'
    Lcy_eq: float  # ft
    Lc_design: float  # ft, the value to enter Table 4-1a with
    trace: Trace


def critical_direction(shape: Shape, x: Axis, y: Axis) -> DirectionResult:
    """Equivalent-effective-length method ("method 3").

    Lcy_eq = Lcx / (rx/ry) converts an x-axis length into the y-axis length that
    would be equally critical. Compare it with the real Lcy and the larger
    governs -- and that same number is what you enter Table 4-1a with, since the
    table is indexed on the least radius of gyration.
    """
    t = Trace("Critical direction (equivalent effective length)")
    ratio = shape.rx_ry
    t.add("rx/ry", ratio, "", "bottom of AISC Table 4-1a")
    t.add("Lcx", x.Lc, "ft")
    t.add("Lcy", y.Lc, "ft")

    Lcy_eq = x.Lc / ratio
    t.add("Lcy_eq = Lcx/(rx/ry)", Lcy_eq, "ft")

    if Lcy_eq > y.Lc:
        governs, Lc_design = "x", Lcy_eq
        t.add("governing direction", "X", "", "", f"{Lcy_eq:.3g} ft > {y.Lc:g} ft")
    else:
        governs, Lc_design = "y", y.Lc
        t.add("governing direction", "Y", "", "", f"{Lcy_eq:.3g} ft <= {y.Lc:g} ft")
    t.add("enter Table 4-1a with", Lc_design, "ft")
    return DirectionResult(governs, Lcy_eq, Lc_design, t)


# ---------------------------------------------------------------------------
# Steps 4-6 -- slenderness ratio through nominal strength
# ---------------------------------------------------------------------------

@dataclass
class ColumnCapacity:
    Pn: float  # kips
    Fn: float  # ksi
    Fe: float  # ksi
    Lc_over_r: float
    inelastic: bool
    governing_axis: str
    trace: Trace
    squash_load: float = 0.0  # phi*Fy*Ag, the ceiling no bracing can beat

    @property
    def lrfd(self) -> float:
        return PHI_C * self.Pn

    @property
    def asd(self) -> float:
        return self.Pn / OMEGA_C

    @property
    def buckling_mode(self) -> str:
        return "inelastic" if self.inelastic else "elastic"

    def report(self) -> str:
        return self.trace.render()

    def __str__(self) -> str:
        return (
            f"phi_c*Pn = {self.lrfd:,.1f} kips, Pn/Omega = {self.asd:,.1f} kips "
            f"({self.governing_axis.upper()} governs, {self.buckling_mode} buckling)"
        )


def _slenderness_ratio(Lc_ft: float, r_in: float, t: Trace, label: str) -> float:
    """Lc/r with the foot-to-inch conversion done explicitly.

    Forgetting the x12 makes the ratio 12 times too small and the strength
    absurdly high. A sane Lc/r for a real column is roughly 30 to 150.
    """
    Lc_in = Lc_ft * 12.0
    ratio = Lc_in / r_in
    t.add(f"{label} (in)", Lc_in, "in", "", f"{Lc_ft:g} ft x 12")
    t.add(f"{label}/r", ratio, "", "AISC Sect. E2", f"r = {r_in:g} in")
    if not 5 < ratio < 400:
        t.add("WARNING", f"Lc/r = {ratio:.1f} is outside the usual 30-150 band",
              "", "", "check the foot-to-inch conversion")
    return ratio


def nominal_stress(mat: Material, Lc_over_r: float, t: Trace) -> tuple[float, float, bool]:
    """Fn from AISC Eqs. E3-2 / E3-3. Returns (Fn, Fe, inelastic)."""
    Fe = pi**2 * E / Lc_over_r**2
    t.add("Fe = pi^2*E/(Lc/r)^2", Fe, "ksi", "AISC Eq. E3-4",
          f"pi^2*E = {pi**2 * E:,.0f}")

    threshold = 4.71 * sqrt(E / mat.Fy)
    ratio = mat.Fy / Fe
    t.add("4.71*sqrt(E/Fy)", threshold, "", "AISC Sect. E3")
    t.add("Fy/Fe", ratio, "")

    inelastic = Lc_over_r <= threshold
    if inelastic != (ratio <= 2.25):
        # The two criteria are algebraically equivalent; disagreement means a
        # borderline case where rounding matters. Trust Lc/r and say so.
        t.add("note", "Lc/r and Fy/Fe criteria straddle the boundary", "", "",
              "borderline column; Lc/r criterion used")

    if inelastic:
        Fn = (0.658 ** ratio) * mat.Fy
        t.add("Fn = [0.658^(Fy/Fe)]*Fy", Fn, "ksi", "AISC Eq. E3-2",
              "short column, inelastic buckling")
    else:
        Fn = 0.877 * Fe
        t.add("Fn = 0.877*Fe", Fn, "ksi", "AISC Eq. E3-3",
              "long column, elastic buckling -- Fy does not appear")

    if Fn > mat.Fy:
        t.add("WARNING", f"Fn = {Fn:.1f} ksi exceeds Fy = {mat.Fy:g} ksi",
              "", "", "a column always buckles before it fully yields")
    return Fn, Fe, inelastic


def available_strength(
    shape: Shape,
    mat: Material,
    x: Axis,
    y: Axis,
    check_slenderness: bool = True,
) -> ColumnCapacity:
    """Full Chapter E check for a member without slender elements."""
    t = Trace(f"Compression check -- {shape} ({mat.name})")

    if check_slenderness:
        cls = classify(shape, mat)
        t.extend(cls.trace, prefix="-- Step 1: section classification --")
        if not cls.non_slender:
            raise ValueError(
                f"{shape} in {mat.name} is a slender-element section. AISC "
                "Section E3 covers members WITHOUT slender elements; this member "
                "requires Section E7."
            )

    t.add("-- Step 2: effective lengths --")
    x.name = x.name or "X"
    y.name = y.name or "Y"
    x.trace(t)
    y.trace(t)

    t.add("-- Step 3: critical direction --")
    direction = critical_direction(shape, x, y)
    t.extend(direction.trace)

    t.add("-- Step 4: slenderness ratio --")
    if direction.governs == "x":
        rx, = shape.require("rx")
        ratio = _slenderness_ratio(x.Lc, rx, t, "Lcx")
    else:
        ry, = shape.require("ry")
        ratio = _slenderness_ratio(y.Lc, ry, t, "Lcy")

    t.add("-- Step 5: nominal stress --")
    Fn, Fe, inelastic = nominal_stress(mat, ratio, t)

    t.add("-- Step 6: strength --")
    Pn = Fn * shape.A
    t.add("Ag", shape.A, "in^2", "AISC Table 1-1", "GROSS area -- no hole deduction")
    t.add("Pn = Fn*Ag", Pn, "kips", "AISC Eq. E3-1")
    t.add("phi_c*Pn", PHI_C * Pn, "kips", "AISC Sect. E1", "phi_c = 0.90")
    t.add("Pn/Omega_c", Pn / OMEGA_C, "kips", "AISC Sect. E1", "Omega_c = 1.67")

    squash = PHI_C * mat.Fy * shape.A
    t.add("ceiling phi*Fy*Ag", squash, "kips", "", "no bracing can beat this")

    return ColumnCapacity(
        Pn, Fn, Fe, ratio, inelastic, direction.governs, t, squash_load=squash
    )
