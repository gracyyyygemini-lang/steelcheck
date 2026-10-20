"""AISC 360 Chapter F -- Design of Members for Flexure.

Covers doubly-symmetric I-shapes bent about the major axis:

* **Section F2** -- compact sections. Three limit states:
  1. Yielding: Mn = Mp = Fy·Zx (Eq. F2-1). Governs when Lb ≤ Lp.
  2. Inelastic lateral-torsional buckling (LTB): Eq. F2-2. Linear interpolation
     between Mp at Lp and 0.7·Fy·Sx at Lr.
  3. Elastic LTB: Mn = Fcr·Sx ≤ Mp (Eqs. F2-3/F2-4). Governs when Lb > Lr.

* **Section F3** -- noncompact flanges (λpf < bf/2tf ≤ λrf). Adds a fourth
  limit state -- flange local buckling (FLB) -- via Eq. F3-1. The available
  moment is the lesser of the FLB capacity and the LTB capacity. Yielding is
  replaced by FLB as the flat-top plateau: the curve's ceiling drops from Mp to
  Mn(FLB).

  About ten of the 289 W shapes are noncompact at Grade 50, including W21X48,
  W14X99, W14X90, and W12X65. Raising on them would block any downstream sweep
  of a framing plan.

Slender flanges (bf/2tf > λrf) still raise; they need Section F3.2 and no W
shape at Grade 50 reaches that.

The Cb moment-gradient factor (Eq. F1-1) scales the LTB strength, but the
Specification is explicit: "Mn shall not exceed Mp." Omitting that cap is a
classic error -- a Cb of 1.3 on a short Lb can push the inelastic formula above
Mp, and that is not physical.

Units: the public API returns kip-ft, matching compression.py returning kips.
The ÷12 step is traced explicitly, like the ×12 in compression._slenderness_ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

from .materials import E, Material
from .shapes import Shape
from .trace import Trace

PHI_B, OMEGA_B = 0.90, 1.67


# ---------------------------------------------------------------------------
# Compactness check -- Table B4.1b (flexure, NOT B4.1a for compression)
# ---------------------------------------------------------------------------

@dataclass
class CompactnessCheck:
    compact: bool
    noncompact_flange: bool
    slender_flange: bool
    flange_ratio: float | None
    flange_lambda_p: float | None
    flange_lambda_r: float | None
    web_compact: bool
    web_ratio: float | None
    web_lambda_p: float | None
    trace: Trace


def compact_check(shape: Shape, mat: Material) -> CompactnessCheck:
    """Table B4.1b classification for a doubly-symmetric I-shape in flexure.

    The limits here differ from Table B4.1a (compression):

        Compression (B4.1a)    Flexure (B4.1b)
        λr_flange = 0.56       λpf = 0.38, λrf = 1.0
        λr_web    = 1.49       λpw = 3.76, λrw = 5.70

    A W18×35 has h/tw = 53.5 and is slender-element as a column (limit 35.9)
    but compact as a beam (limit 90.6). Same shape, different load, different
    answer -- this is the distinction the classification gate enforces.
    """
    t = Trace(f"Compactness check (flexure) -- {shape} ({mat.name})")
    root = sqrt(E / mat.Fy)
    t.add("sqrt(E/Fy)", root, "", "", f"E = {E:,.0f} ksi, Fy = {mat.Fy:g} ksi")

    # Flange: Table B4.1b Case 10 (rolled I-shapes)
    lam_pf = 0.38 * root
    lam_rf = 1.0 * root
    t.add("lambda_pf = 0.38*sqrt(E/Fy)", lam_pf, "", "AISC Table B4.1b Case 10")
    t.add("lambda_rf = 1.0*sqrt(E/Fy)", lam_rf, "", "AISC Table B4.1b Case 10")

    bf_2tf = shape.bf_2tf
    noncompact_flange = False
    slender_flange = False
    if bf_2tf is not None:
        if bf_2tf <= lam_pf:
            label = "compact"
        elif bf_2tf <= lam_rf:
            label = "noncompact"
            noncompact_flange = True
        else:
            label = "SLENDER"
            slender_flange = True
        t.add("bf/2tf", bf_2tf, "", "AISC Table 1-1",
              f"{'<=' if bf_2tf <= lam_pf else '>'} {lam_pf:.2f} -- {label}")

    # Web: Table B4.1b Case 15 (I-shapes in flexure)
    lam_pw = 3.76 * root
    lam_rw = 5.70 * root
    t.add("lambda_pw = 3.76*sqrt(E/Fy)", lam_pw, "", "AISC Table B4.1b Case 15")

    h_tw = shape.h_tw
    web_ok = True
    if h_tw is not None:
        web_ok = h_tw <= lam_pw
        t.add("h/tw", h_tw, "", "AISC Table 1-1",
              f"{'<=' if web_ok else '>'} {lam_pw:.1f} -- "
              f"{'compact' if web_ok else 'NOT compact'}")

    compact = (not noncompact_flange) and (not slender_flange) and web_ok
    status = "compact" if compact else (
        "SLENDER FLANGE" if slender_flange else
        "noncompact flange" if noncompact_flange else "noncompact web"
    )
    t.add("result", status, "", "AISC Sect. F2" if compact else "AISC Sect. F3")
    return CompactnessCheck(
        compact, noncompact_flange, slender_flange,
        bf_2tf, lam_pf, lam_rf, web_ok, h_tw, lam_pw, t,
    )


# ---------------------------------------------------------------------------
# Cb -- moment-gradient factor (Eq. F1-1)
# ---------------------------------------------------------------------------

def Cb(Mmax: float, MA: float, MB: float, MC: float) -> float:
    """AISC Eq. F1-1 moment-gradient factor.

    All four moments are **absolute values** at the quarter points of the
    unbraced segment:
      Mmax -- maximum moment in the segment
      MA   -- moment at 1/4 point
      MB   -- moment at midpoint (1/2 point)
      MC   -- moment at 3/4 point

    Common values for simply-supported beams:
      uniform load   -> Cb ≈ 1.136
      midpoint load  -> Cb ≈ 1.316
      uniform moment -> Cb = 1.0

    Cb = 1.0 is always conservative.
    """
    return 12.5 * Mmax / (2.5 * Mmax + 3.0 * MA + 4.0 * MB + 3.0 * MC)


# ---------------------------------------------------------------------------
# Limit state parameters: Lp, Lr, Mp
# ---------------------------------------------------------------------------

def _Lp(shape: Shape, mat: Material) -> float:
    """Limiting unbraced length for yielding, in inches.

    Lp = 1.76·ry·sqrt(E/Fy)   (AISC Eq. F2-5)
    """
    ry, = shape.require("ry")
    return 1.76 * ry * sqrt(E / mat.Fy)


def _Lr(shape: Shape, mat: Material) -> float:
    """Limiting unbraced length for inelastic LTB, in inches.

    Lr = 1.95·rts·(E / 0.7Fy)·sqrt(J·c/(Sx·ho) +
         sqrt((J·c/(Sx·ho))^2 + 6.76·(0.7Fy/E)^2))

    (AISC Eq. F2-6)

    c = 1.0 for doubly-symmetric I-shapes.
    """
    rts, J, Sx, ho = shape.require("rts", "J", "Sx", "ho")
    c = 1.0  # doubly-symmetric I-shape
    term = J * c / (Sx * ho)
    ratio = 0.7 * mat.Fy / E
    return 1.95 * rts * (E / (0.7 * mat.Fy)) * sqrt(
        term + sqrt(term ** 2 + 6.76 * ratio ** 2)
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class FlexureCapacity:
    """Available flexural strength, with the full calculation trace."""

    Mn_kipin: float  # nominal strength, kip-in
    Mp_kipin: float  # plastic moment, kip-in
    Lp_ft: float
    Lr_ft: float
    Lb_ft: float
    Cb_used: float
    zone: str  # 'yielding', 'inelastic_ltb', 'elastic_ltb'
    capped: bool  # True if Cb·Mn was capped at Mp
    noncompact_flange: bool  # True if FLB governed over LTB
    trace: Trace

    @property
    def lrfd(self) -> float:
        """Design strength φ_b·Mn, in **kip-ft** (matching Table 3-2)."""
        return PHI_B * self.Mn_kipin / 12.0

    @property
    def asd(self) -> float:
        """Allowable strength Mn/Ω_b, in kip-ft."""
        return self.Mn_kipin / OMEGA_B / 12.0

    @property
    def zone_label(self) -> str:
        labels = {
            "yielding": "yielding",
            "inelastic_ltb": "inelastic LTB",
            "elastic_ltb": "elastic LTB",
            "flb": "flange local buckling",
        }
        return labels.get(self.zone, self.zone)

    def report(self) -> str:
        return self.trace.render()

    def __str__(self) -> str:
        return (
            f"phi_b*Mn = {self.lrfd:,.1f} kip-ft, Mn/Omega = {self.asd:,.1f} kip-ft "
            f"({self.zone_label}"
            f"{', capped at Mp' if self.capped else ''})"
        )


def available_strength(
    shape: Shape,
    mat: Material,
    Lb_ft: float,
    Cb_value: float = 1.0,
) -> FlexureCapacity:
    """Full Chapter F check for a doubly-symmetric I-shape, major axis.

    Parameters
    ----------
    shape : Shape
        Must be a doubly-symmetric I-shape (W, M, S, HP).
    mat : Material
        Steel grade.
    Lb_ft : float
        Unbraced length, **feet**.
    Cb_value : float
        Moment-gradient factor from Eq. F1-1. Default 1.0 (conservative).

    Returns
    -------
    FlexureCapacity
        Available moment strength. ``.lrfd`` is φ_b·Mn in kip-ft.
    """
    t = Trace(f"Flexure check -- {shape} ({mat.name})")

    # ------------------------------------------------------------------
    # Step 1: compactness gate
    # ------------------------------------------------------------------
    cls = compact_check(shape, mat)
    t.extend(cls.trace, prefix="-- Step 1: compactness check (Table B4.1b) --")
    if cls.slender_flange:
        raise ValueError(
            f"{shape} in {mat.name} has a slender flange (bf/2tf = "
            f"{cls.flange_ratio:.2f} > lambda_rf = {cls.flange_lambda_r:.2f}). "
            "This requires AISC Section F3.2, which is not implemented."
        )
    if not cls.web_compact:
        raise ValueError(
            f"{shape} in {mat.name} has a noncompact or slender web (h/tw = "
            f"{cls.web_ratio:.1f} > lambda_pw = {cls.web_lambda_p:.1f}). "
            "This requires AISC Sections F4/F5, which are not implemented."
        )

    # ------------------------------------------------------------------
    # Step 2: plastic moment
    # ------------------------------------------------------------------
    t.add("-- Step 2: plastic moment --")
    Zx, Sx = shape.require("Zx", "Sx")
    Mp = mat.Fy * Zx
    t.add("Fy", mat.Fy, "ksi", "AISC Table 2-4")
    t.add("Zx", Zx, "in^3", "AISC Table 1-1")
    t.add("Mp = Fy*Zx", Mp, "kip-in", "AISC Eq. F2-1")
    t.add("Mp (kip-ft)", Mp / 12.0, "kip-ft", "", "÷ 12 in/ft")

    # ------------------------------------------------------------------
    # Step 2b: FLB check for noncompact flange (Section F3)
    # ------------------------------------------------------------------
    Mn_flb = None
    if cls.noncompact_flange:
        t.add("-- Step 2b: flange local buckling (noncompact flange, F3-1) --")
        lam = cls.flange_ratio
        lam_pf = cls.flange_lambda_p
        lam_rf = cls.flange_lambda_r
        t.add("bf/2tf", lam, "", "AISC Table 1-1")
        t.add("lambda_pf", lam_pf, "", "AISC Table B4.1b Case 10")
        t.add("lambda_rf", lam_rf, "", "AISC Table B4.1b Case 10")

        Mr = 0.7 * mat.Fy * Sx
        t.add("Mr = 0.7*Fy*Sx", Mr, "kip-in", "AISC Eq. F2-2 / F3-1 term")

        Mn_flb = Mp - (Mp - Mr) * ((lam - lam_pf) / (lam_rf - lam_pf))
        t.add(
            "Mn(FLB) = Mp - (Mp - 0.7*Fy*Sx)*((lam - lam_pf)/(lam_rf - lam_pf))",
            Mn_flb, "kip-in", "AISC Eq. F3-1",
        )
        t.add("Mn(FLB) (kip-ft)", Mn_flb / 12.0, "kip-ft", "", "÷ 12 in/ft")
        t.add("note", "FLB replaces yielding as the plateau -- "
              "the flat top is Mn(FLB), not Mp", "", "",
              "limit state is the lesser of FLB and LTB")

    # ------------------------------------------------------------------
    # Step 3: unbraced length and limits
    # ------------------------------------------------------------------
    t.add("-- Step 3: unbraced length --")
    Lb_in = Lb_ft * 12.0
    t.add("Lb", Lb_ft, "ft")
    t.add("Lb (in)", Lb_in, "in", "", f"{Lb_ft:g} ft × 12")

    Lp_in = _Lp(shape, mat)
    Lr_in = _Lr(shape, mat)
    Lp_ft_ = Lp_in / 12.0
    Lr_ft_ = Lr_in / 12.0
    t.add("Lp = 1.76*ry*sqrt(E/Fy)", Lp_in, "in", "AISC Eq. F2-5")
    t.add("Lp (ft)", Lp_ft_, "ft", "", "÷ 12 in/ft")
    t.add("Lr (Eq. F2-6)", Lr_in, "in", "AISC Eq. F2-6")
    t.add("Lr (ft)", Lr_ft_, "ft", "", "÷ 12 in/ft")
    t.add("Cb", Cb_value, "", "AISC Eq. F1-1" if Cb_value != 1.0 else "",
          "1.0 is conservative" if Cb_value == 1.0 else "")

    # ------------------------------------------------------------------
    # Step 4: nominal moment from LTB
    # ------------------------------------------------------------------
    t.add("-- Step 4: nominal moment (LTB) --")
    capped = False

    if Lb_in <= Lp_in:
        # Zone 1: yielding -- Mn = Mp
        Mn_ltb = Mp
        zone = "yielding"
        t.add("Lb <= Lp", f"{Lb_ft:g} ft <= {Lp_ft_:.2f} ft", "", "",
              "no LTB -- full plastic moment")
        t.add("Mn(LTB) = Mp", Mn_ltb, "kip-in", "AISC Eq. F2-1")

    elif Lb_in <= Lr_in:
        # Zone 2: inelastic LTB
        zone = "inelastic_ltb"
        Mr = 0.7 * mat.Fy * Sx
        Mn_raw = Cb_value * (Mp - (Mp - Mr) * ((Lb_in - Lp_in) / (Lr_in - Lp_in)))
        t.add("Lp < Lb <= Lr", f"{Lp_ft_:.2f} ft < {Lb_ft:g} ft <= {Lr_ft_:.2f} ft",
              "", "", "inelastic LTB zone")
        t.add("0.7*Fy*Sx", Mr, "kip-in")
        t.add(
            "Cb*[Mp - (Mp - 0.7FySx)*(Lb-Lp)/(Lr-Lp)]",
            Mn_raw, "kip-in", "AISC Eq. F2-2",
        )

        if Mn_raw > Mp:
            t.add("WARNING: Cb-boosted Mn exceeds Mp", "", "", "",
                  f"{Mn_raw:,.0f} > {Mp:,.0f} kip-in -- capped at Mp")
            Mn_raw = Mp
            capped = True

        Mn_ltb = Mn_raw
        t.add("Mn(LTB)", Mn_ltb, "kip-in", "AISC Eq. F2-2", "shall not exceed Mp")

    else:
        # Zone 3: elastic LTB
        zone = "elastic_ltb"
        rts, J, ho = shape.require("rts", "J", "ho")
        c = 1.0
        Fcr = (
            Cb_value * pi ** 2 * E / (Lb_in / rts) ** 2
            * sqrt(1 + 0.078 * J * c / (Sx * ho) * (Lb_in / rts) ** 2)
        )
        Mn_raw = Fcr * Sx
        t.add("Lb > Lr", f"{Lb_ft:g} ft > {Lr_ft_:.2f} ft", "", "", "elastic LTB zone")
        t.add("rts", rts, "in", "AISC Table 1-1")
        t.add("J", J, "in^4", "AISC Table 1-1")
        t.add("ho", ho, "in", "AISC Table 1-1")
        t.add("c", c, "", "", "1.0 for doubly-symmetric I-shapes")
        t.add("Lb/rts", Lb_in / rts, "")
        t.add("Fcr", Fcr, "ksi", "AISC Eq. F2-4")
        t.add("Fcr*Sx", Mn_raw, "kip-in", "AISC Eq. F2-3")

        if Mn_raw > Mp:
            t.add("WARNING: Cb-boosted Mn exceeds Mp", "", "", "",
                  f"{Mn_raw:,.0f} > {Mp:,.0f} kip-in -- capped at Mp")
            Mn_raw = Mp
            capped = True

        Mn_ltb = Mn_raw
        t.add("Mn(LTB)", Mn_ltb, "kip-in", "AISC Eq. F2-3", "shall not exceed Mp")

    # ------------------------------------------------------------------
    # Step 5: governing Mn (lesser of LTB and FLB if applicable)
    # ------------------------------------------------------------------
    t.add("-- Step 5: governing nominal moment --")
    flb_governs = False
    if Mn_flb is not None:
        t.add("Mn(LTB)", Mn_ltb, "kip-in")
        t.add("Mn(FLB)", Mn_flb, "kip-in")
        if Mn_flb < Mn_ltb:
            Mn = Mn_flb
            flb_governs = True
            t.add("Mn = min(LTB, FLB)", Mn, "kip-in", "",
                  "FLB governs -- noncompact flange controls")
        else:
            Mn = Mn_ltb
            t.add("Mn = min(LTB, FLB)", Mn, "kip-in", "",
                  "LTB governs -- flange does not control")
    else:
        Mn = Mn_ltb
        t.add("Mn", Mn, "kip-in")

    # ------------------------------------------------------------------
    # Step 6: design strength in kip-ft
    # ------------------------------------------------------------------
    t.add("-- Step 6: design strength --")
    Mn_kft = Mn / 12.0
    t.add("Mn (kip-ft)", Mn_kft, "kip-ft", "", "÷ 12 in/ft")
    t.add("phi_b*Mn", PHI_B * Mn_kft, "kip-ft", "AISC Sect. F1", "phi_b = 0.90")
    t.add("Mn/Omega_b", Mn_kft / OMEGA_B, "kip-ft", "AISC Sect. F1",
          "Omega_b = 1.67")

    if flb_governs:
        zone = "flb"

    return FlexureCapacity(
        Mn_kipin=Mn,
        Mp_kipin=Mp,
        Lp_ft=Lp_ft_,
        Lr_ft=Lr_ft_,
        Lb_ft=Lb_ft,
        Cb_used=Cb_value,
        zone=zone,
        capped=capped,
        noncompact_flange=cls.noncompact_flange,
        trace=t,
    )
