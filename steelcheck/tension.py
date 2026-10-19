"""AISC 360 Chapter D -- Design of Members for Tension.

Two limit states, and the member fails at whichever it reaches first:

* **Tensile yielding** on the gross section, away from the connection.
  Pn = Fy*Ag (Eq. D2-1), phi = 0.90. Ductile, so the factor is generous.
  Note that this is completely independent of how the member is connected.

* **Tensile rupture** on the effective net section, at the connection.
  Pn = Fu*Ae (Eq. D2-2), phi = 0.75. Brittle, so the factor is harsh.

Everything to do with holes, welds, shear lag and connection geometry feeds
rupture only. That asymmetry is the shape of the whole chapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .materials import Material
from .shapes import Shape
from .trace import Trace

PHI_YIELD, OMEGA_YIELD = 0.90, 1.67
PHI_RUPTURE, OMEGA_RUPTURE = 0.75, 2.00

#: Extra diameter added to a bolt when computing net area, in. The standard hole
#: is db + 1/16 on the drawing; calculations add another 1/16 for the ring of
#: metal damaged by punching or drilling.
HOLE_ALLOWANCE = 0.125


@dataclass
class Capacity:
    """One limit state's result, with the work shown."""

    Pn: float  # nominal strength, kips
    phi: float
    omega: float
    limit_state: str
    trace: Trace

    @property
    def lrfd(self) -> float:
        """Design strength phi*Pn, kips."""
        return self.phi * self.Pn

    @property
    def asd(self) -> float:
        """Allowable strength Pn/Omega, kips."""
        return self.Pn / self.omega

    def __str__(self) -> str:
        return (
            f"{self.limit_state}: phi*Pn = {self.lrfd:,.1f} kips, "
            f"Pn/Omega = {self.asd:,.1f} kips"
        )


@dataclass
class Hole:
    """Holes at ONE cross section, through elements of a single thickness.

    The count is how many holes the critical *cross section* passes through --
    not how many bolts the connection has. Five bolts in a line still put one
    hole in any given cross section. A W shape bolted through both flanges and
    the web is two Hole records, because tf and tw differ.
    """

    count: int
    thickness: float  # in


@dataclass
class Bolted:
    """A bolted connection."""

    db: float  # bolt diameter, in
    holes: list[Hole] = field(default_factory=list)
    n_per_line: int = 1  # bolts per line in the direction of the load
    spacing: float = 3.0  # center-to-center, in
    xbar: float | None = None  # connection eccentricity, in; None if Case 1
    all_elements_connected: bool = False  # Table D3.1 Case 1

    @property
    def L(self) -> float:
        """Connection length: first bolt to last, = (n - 1) * spacing."""
        return max(self.n_per_line - 1, 0) * self.spacing


@dataclass
class Welded:
    """A welded connection."""

    longitudinal: bool = False
    transverse: bool = False
    L: float | None = None  # longitudinal weld length, in (average if unequal)
    w: float | None = None  # perpendicular distance between welds, in (Case 4)
    xbar: float | None = None
    connected_area: float | None = None  # in^2, required for transverse-only
    all_elements_connected: bool = False


Connection = Bolted | Welded


# ---------------------------------------------------------------------------
# Limit state 1 -- yielding
# ---------------------------------------------------------------------------

def tensile_yielding(shape: Shape, mat: Material) -> Capacity:
    """Pn = Fy*Ag -- AISC Eq. D2-1.

    Uses the GROSS area: yielding happens on the full section away from the
    connection, where there are no holes. The connection detail is irrelevant
    here, which is why this number is identical across every connection variant
    of the same shape.
    """
    Pn = mat.Fy * shape.A
    t = Trace(f"Tensile yielding -- {shape} ({mat.name})")
    t.add("Ag", shape.A, "in^2", "AISC Table 1-1")
    t.add("Fy", mat.Fy, "ksi", "AISC Table 2-4")
    t.add("Pn = Fy*Ag", Pn, "kips", "AISC Eq. D2-1")
    t.add("phi*Pn", PHI_YIELD * Pn, "kips", "AISC Sect. D2(a)", "phi = 0.90, ductile")
    t.add("Pn/Omega", Pn / OMEGA_YIELD, "kips", "AISC Sect. D2(a)", "Omega = 1.67")
    return Capacity(Pn, PHI_YIELD, OMEGA_YIELD, "tensile yielding", t)


# ---------------------------------------------------------------------------
# Net area and shear lag
# ---------------------------------------------------------------------------

def net_area(shape: Shape, conn: Connection, t: Trace) -> float:
    """An -- gross area less holes, or Ag / connected area for welds."""
    if isinstance(conn, Bolted):
        dh = conn.db + HOLE_ALLOWANCE
        t.add(
            "hole dia. for calc", dh, "in", "AISC Sect. B4.3b",
            f"db + 1/8 = {conn.db:g} + {HOLE_ALLOWANCE:g}",
        )
        deduction = 0.0
        for h in conn.holes:
            d = h.count * dh * h.thickness
            t.add(
                f"deduct {h.count} hole(s) at t = {h.thickness:g} in", d, "in^2", "",
                f"{h.count} x {dh:g} x {h.thickness:g}",
            )
            deduction += d
        An = shape.A - deduction
        t.add("An = Ag - holes", An, "in^2", "AISC Sect. B4.3b")
        return An

    if conn.longitudinal:
        # Longitudinal welds, with or without transverse: no holes exist, and the
        # force has a long run over which to distribute.
        t.add("An = Ag", shape.A, "in^2", "AISC Sect. B4.3a", "welded, no holes")
        return shape.A

    if conn.transverse:
        # Transverse welds only -- Table D3.1 Case 3 restricts An to the elements
        # the weld actually attaches.
        if conn.connected_area is None:
            raise ValueError(
                "Transverse-only weld: set connected_area to the gross area of "
                "the directly connected elements (AISC Table D3.1 Case 3)."
            )
        t.add(
            "An = area of directly connected elements", conn.connected_area, "in^2",
            "AISC Table D3.1 Case 3",
        )
        return conn.connected_area

    raise ValueError("Welded connection must set longitudinal and/or transverse.")


def shear_lag_factor(shape: Shape, conn: Connection, t: Trace) -> float:
    """U -- AISC Table D3.1.

    When load reaches only some of the cross-sectional elements, the rest are
    not yet fully participating at the critical section. U scales An down to the
    area that is actually transmitting load.
    """
    if conn.all_elements_connected:
        t.add("U", 1.0, "", "AISC Table D3.1 Case 1", "all elements connected")
        return 1.0

    if isinstance(conn, Welded) and conn.transverse and not conn.longitudinal:
        t.add("U", 1.0, "", "AISC Table D3.1 Case 3", "transverse welds only")
        return 1.0

    L = conn.L
    if L is None or L <= 0:
        raise ValueError(
            "Connection length L must be positive to compute U. For bolts, "
            "L = (n_per_line - 1) * spacing -- a single bolt per line gives L = 0."
        )
    if conn.xbar is None:
        raise ValueError(
            "xbar is required for this case. It is the distance from the back of "
            "the connected element to the centroidal axis parallel to it: for an "
            "angle use xbar when the LONG leg is connected and ybar when the "
            "SHORT leg is; for a W connected at the flanges use the ybar of the "
            "corresponding WT."
        )

    base = 1.0 - conn.xbar / L
    t.add("L", L, "in", "", "first bolt to last = (n-1) x spacing"
          if isinstance(conn, Bolted) else "longitudinal weld length")
    t.add("xbar", conn.xbar, "in", "AISC Table 1-1/1-5/1-7")

    if isinstance(conn, Welded) and conn.longitudinal and not conn.transverse:
        if conn.w is None:
            raise ValueError(
                "Longitudinal-welds-only (Case 4) needs w, the perpendicular "
                "distance between the two welds."
            )
        factor = 3.0 * L**2 / (3.0 * L**2 + conn.w**2)
        U = factor * base
        t.add("w", conn.w, "in")
        t.add("3L^2/(3L^2+w^2)", factor, "", "AISC Table D3.1 Case 4")
        t.add("1 - xbar/L", base, "")
        t.add("U", U, "", "AISC Table D3.1 Case 4")
        return U

    t.add("U = 1 - xbar/L", base, "", "AISC Table D3.1 Case 2")
    return base


# ---------------------------------------------------------------------------
# Limit state 2 -- rupture
# ---------------------------------------------------------------------------

def tensile_rupture(shape: Shape, mat: Material, conn: Connection) -> Capacity:
    """Pn = Fu*Ae -- AISC Eq. D2-2, with Ae = An*U from Eq. D3-1."""
    t = Trace(f"Tensile rupture -- {shape} ({mat.name})")
    t.add("Ag", shape.A, "in^2", "AISC Table 1-1")
    t.add("Fu", mat.Fu, "ksi", "AISC Table 2-4")

    An = net_area(shape, conn, t)
    U = shear_lag_factor(shape, conn, t)

    U_min = getattr(conn, "connected_area", None)
    if U_min is not None and not conn.all_elements_connected:
        floor = U_min / shape.A
        if floor > U:
            t.add(
                "U raised to minimum", floor, "", "AISC Sect. D3",
                "U >= (connected element area)/Ag",
            )
            U = floor

    Ae = An * U
    Pn = mat.Fu * Ae
    t.add("Ae = An*U", Ae, "in^2", "AISC Eq. D3-1")
    t.add("Pn = Fu*Ae", Pn, "kips", "AISC Eq. D2-2")
    t.add("phi*Pn", PHI_RUPTURE * Pn, "kips", "AISC Sect. D2(b)", "phi = 0.75, brittle")
    t.add("Pn/Omega", Pn / OMEGA_RUPTURE, "kips", "AISC Sect. D2(b)", "Omega = 2.00")
    return Capacity(Pn, PHI_RUPTURE, OMEGA_RUPTURE, "tensile rupture", t)


# ---------------------------------------------------------------------------
# Governing capacity
# ---------------------------------------------------------------------------

@dataclass
class TensionResult:
    yielding: Capacity
    rupture: Capacity
    governing: Capacity

    @property
    def lrfd(self) -> float:
        return self.governing.lrfd

    @property
    def asd(self) -> float:
        return self.governing.asd

    @property
    def limit_state(self) -> str:
        return self.governing.limit_state

    def report(self) -> str:
        return "\n\n".join(
            [
                self.yielding.trace.render(),
                self.rupture.trace.render(),
                (
                    "GOVERNING\n=========\n"
                    f"  phi*Pn   = {self.lrfd:,.1f} kips\n"
                    f"  Pn/Omega = {self.asd:,.1f} kips\n"
                    f"  limit state: {self.limit_state}"
                ),
            ]
        )

    def __str__(self) -> str:
        return (
            f"phi*Pn = {self.lrfd:,.1f} kips "
            f"(governed by {self.limit_state})"
        )


def available_strength(shape: Shape, mat: Material, conn: Connection) -> TensionResult:
    """Full Chapter D check. The smaller of the two limit states governs."""
    y = tensile_yielding(shape, mat)
    r = tensile_rupture(shape, mat, conn)
    gov = y if y.lrfd <= r.lrfd else r
    return TensionResult(y, r, gov)


def required_strength(
    PD: float, PL: float, PLr: float = 0.0, method: str = "LRFD"
) -> tuple[float, Trace]:
    """Required tensile strength from dead and live load.

    LRFD takes the larger of 1.4D and 1.2D + 1.6L + 0.5Lr; ASD takes the larger
    of D and D + L. 1.4D only governs when D > 8L, which real buildings never
    approach.
    """
    t = Trace(f"Required strength ({method})")
    t.add("PD", PD, "kips")
    t.add("PL", PL, "kips")
    if method.upper() == "LRFD":
        c1 = 1.4 * PD
        c2 = 1.2 * PD + 1.6 * PL + 0.5 * PLr
        t.add("1.4D", c1, "kips", "ASCE 7 comb. 1")
        t.add("1.2D + 1.6L + 0.5Lr", c2, "kips", "ASCE 7 comb. 2")
        Pu = max(c1, c2)
        t.add("Pu (governing)", Pu, "kips", "", "1.4D governs only if D > 8L")
        return Pu, t
    c1, c2 = PD, PD + PL
    t.add("D", c1, "kips")
    t.add("D + L", c2, "kips")
    Pa = max(c1, c2)
    t.add("Pa (governing)", Pa, "kips")
    return Pa, t
