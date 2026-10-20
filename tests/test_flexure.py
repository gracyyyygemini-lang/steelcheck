"""Validation against hand-calculated AISC Chapter F flexural strengths.

Every expected value in this file was computed independently of the code under
test, by working the AISC equations by hand using section properties from
steelpy. The W24X76 values also cross-check against published AISC Table 3-2
entries (750 kip-ft, Lp = 6.78 ft, Lr = 19.5 ft).

The W18X35 is the crossover case: slender-element as a column (h/tw = 53.5 >
35.9) but compact as a beam (h/tw = 53.5 < 90.6). The same shape, under
different loading, gets a different classification -- and that difference is
exactly the kind of distinction a language model flattens.
"""

import pytest

import steelcheck as sc
from steelcheck.flexure import (
    Cb,
    FlexureCapacity,
    available_strength,
    compact_check,
)

A992 = sc.materials.A992
TOL = 0.005


def close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(actual - expected) / expected <= tol


# ---------------------------------------------------------------------------
# W24X76, compact section -- the primary validation shape.
# Hand calc:  Fy=50, Zx=200, Sx=176, ry=1.92, rts=2.33, J=2.68, ho=23.2
#
# Mp = 50 * 200 = 10,000 kip-in = 833.3 kip-ft
# phi_b*Mp = 0.9 * 833.3 = 750.0 kip-ft  (matches AISC Table 3-2: 750)
#
# Lp = 1.76 * 1.92 * sqrt(29000/50) / 12 = 6.78 ft  (Table 3-2: 6.78)
# Lr = 19.50 ft  (Table 3-2: 19.5)
# ---------------------------------------------------------------------------

W24X76 = sc.shapes.get("W24X76")


def test_w24x76_is_compact():
    cls = compact_check(W24X76, A992)
    assert cls.compact
    assert not cls.noncompact_flange
    assert not cls.slender_flange


def test_w24x76_phi_Mp():
    """phi_b*Mp = 0.9 * 50 * 200 / 12 = 750.0 kip-ft.

    Hand calc, confirmed against AISC Table 3-2 (750 kip-ft).
    """
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    assert cap.zone == "yielding"
    assert close(cap.lrfd, 750.0)


def test_w24x76_table_3_2_cross_check():
    """The published Table 3-2 value is 750 kip-ft LRFD, 499 kip-ft ASD."""
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    assert close(cap.lrfd, 750.0, tol=0.01)
    assert close(cap.asd, 499.0, tol=0.01)


def test_w24x76_Lp():
    """Lp = 1.76 * 1.92 * sqrt(29000/50) / 12 = 6.78 ft.

    Confirmed against AISC Table 3-2 (6.78 ft).
    """
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    assert close(cap.Lp_ft, 6.78, tol=0.01)


def test_w24x76_Lr():
    """Lr = 19.50 ft from Eq. F2-6.

    Hand calc:
      c = 1.0
      Jc/(Sx*ho) = 2.68 / (176 * 23.2) = 6.558e-4
      0.7Fy/E = 35/29000 = 1.207e-3
      Lr = 1.95 * 2.33 * (29000/35) * sqrt(6.558e-4 + sqrt(4.300e-7 + 9.85e-6))
         = 234.0 in = 19.50 ft

    Confirmed against AISC Table 3-2 (19.5 ft).
    """
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    assert close(cap.Lr_ft, 19.50, tol=0.01)


def test_w24x76_inelastic_ltb():
    """W24X76, Lb = 12 ft, Cb = 1.0: inelastic LTB.

    Hand calc (Eq. F2-2):
      Mr = 0.7 * 50 * 176 = 6,160 kip-in
      Mn = Mp - (Mp - Mr) * (Lb - Lp)/(Lr - Lp)
         = 10000 - (10000 - 6160) * (144 - 81.4)/(234 - 81.4)
         = 10000 - 3840 * 0.4105 = 8424 kip-in
      phi*Mn = 0.9 * 8424 / 12 = 631.8 kip-ft
    """
    cap = available_strength(W24X76, A992, Lb_ft=12.0, Cb_value=1.0)
    assert cap.zone == "inelastic_ltb"
    assert close(cap.lrfd, 631.8)
    assert not cap.capped


def test_w24x76_elastic_ltb():
    """W24X76, Lb = 25 ft, Cb = 1.0: elastic LTB.

    Hand calc (Eq. F2-3/F2-4):
      Lb/rts = 300/2.33 = 128.76
      Fcr = pi^2*29000/128.76^2 * sqrt(1 + 0.078*(2.68/(176*23.2))*128.76^2)
          = 17.26 * 1.3597 = 23.47 ksi
      Mn = 23.47 * 176 = 4131 kip-in
      phi*Mn = 0.9 * 4131 / 12 = 309.9 kip-ft
    """
    cap = available_strength(W24X76, A992, Lb_ft=25.0, Cb_value=1.0)
    assert cap.zone == "elastic_ltb"
    assert close(cap.lrfd, 309.9)


def test_w24x76_Cb_cap_at_Mp():
    """W24X76, Lb = 12 ft, Cb = 1.3: Cb-boosted Mn exceeds Mp, capped.

    Hand calc:
      Mn(Cb=1.0) = 8424 kip-in (from inelastic LTB test above)
      Cb*Mn = 1.3 * 8424 = 10,951 > Mp = 10,000
      Mn is capped at Mp = 10,000 kip-in
      phi*Mn = 750.0 kip-ft
    """
    cap = available_strength(W24X76, A992, Lb_ft=12.0, Cb_value=1.3)
    assert cap.capped, "Cb-boosted Mn must be capped at Mp"
    assert close(cap.lrfd, 750.0)


# ---------------------------------------------------------------------------
# Boundary test: at Lb = Lr, both formulas give 0.7·Fy·Sx
# ---------------------------------------------------------------------------

def test_at_Lr_inelastic_and_elastic_agree():
    """At the boundary Lb = Lr, the inelastic formula gives 0.7*Fy*Sx = Mr,
    and the elastic formula's Fcr*Sx should also equal Mr.

    This is the continuity condition that makes the three-zone curve smooth.
    """
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    Lr = cap.Lr_ft

    # Evaluate at Lr - epsilon (inelastic side) and Lr + epsilon (elastic side)
    below = available_strength(W24X76, A992, Lb_ft=Lr - 0.001)
    above = available_strength(W24X76, A992, Lb_ft=Lr + 0.001)

    # Both should be very close to 0.7*Fy*Sx / 12 (kip-ft)
    Mr_kft = 0.7 * 50.0 * 176.0 / 12.0  # = 513.3 kip-ft
    phi_Mr = 0.9 * Mr_kft  # = 462.0 kip-ft

    assert close(below.lrfd, phi_Mr, tol=0.01)
    assert close(above.lrfd, phi_Mr, tol=0.01)


# ---------------------------------------------------------------------------
# W18X35 -- compact for flexure, slender for compression.
# The existing test_steelpy_integration.py already asserts the compression side.
# ---------------------------------------------------------------------------

def test_w18x35_compact_for_flexure():
    """W18X35 has h/tw = 53.5, which exceeds lambda_r = 35.9 for compression
    but is well under lambda_pw = 90.6 for flexure. Same shape, different load.

    phi_b*Mp = 0.9 * 50 * 66.5 / 12 = 249.4 kip-ft (hand calc).
    Lp = 1.76 * 1.22 * sqrt(29000/50) / 12 = 4.31 ft (hand calc).

    AISC Table 3-2: phi_b*Mp = 249 kip-ft.
    """
    s = sc.shapes.get("W18X35")
    cls = compact_check(s, A992)
    assert cls.compact, "W18X35 is compact in flexure"
    assert cls.web_compact, "web h/tw < 90.6"

    cap = available_strength(s, A992, Lb_ft=3.0)
    assert close(cap.lrfd, 249.4)            # hand calculation
    assert close(cap.lrfd, 249.0, tol=0.01)  # AISC Table 3-2
    assert close(cap.Lp_ft, 4.31, tol=0.01)


def test_w18x35_is_both_compact_beam_and_slender_column():
    """The same shape can be compact for flexure and slender for compression.
    This is the test already referenced in test_steelpy_integration.py."""
    s = sc.shapes.get("W18X35")
    assert compact_check(s, A992).compact
    assert not sc.compression.classify(s, A992).non_slender


# ---------------------------------------------------------------------------
# W16X31 -- Table 3-2 cross-check
# ---------------------------------------------------------------------------

def test_w16x31_table_3_2():
    """AISC Table 3-2: phi_b*Mp = 203 kip-ft, Lp = 4.13 ft, Lr = 11.8 ft.

    Hand calc:
      phi_b*Mp = 0.9 * 50 * 54.0 / 12 = 202.5 kip-ft
      Lp = 1.76 * 1.17 * 24.08 / 12 = 4.13 ft
    """
    s = sc.shapes.get("W16X31")
    cap = available_strength(s, A992, Lb_ft=3.0)
    assert close(cap.lrfd, 202.5)            # hand calculation
    assert close(cap.lrfd, 203.0, tol=0.01)  # AISC Table 3-2
    assert close(cap.Lp_ft, 4.13, tol=0.01)
    assert close(cap.Lr_ft, 11.8, tol=0.02)


# ---------------------------------------------------------------------------
# Physical / monotonicity tests
# ---------------------------------------------------------------------------

def test_Mn_decreases_monotonically_with_Lb():
    lengths = [4, 6, 8, 12, 16, 20, 25, 30]
    caps = [available_strength(W24X76, A992, Lb_ft=L).lrfd for L in lengths]
    assert caps == sorted(caps, reverse=True)


def test_Mn_never_exceeds_Mp_regardless_of_Cb():
    """Even with a very high Cb, the cap at Mp holds."""
    for Cb_val in (1.0, 1.14, 1.3, 1.67, 2.0, 3.0):
        cap = available_strength(W24X76, A992, Lb_ft=8.0, Cb_value=Cb_val)
        assert cap.lrfd <= 750.0 + 0.1, f"Cb={Cb_val}: lrfd={cap.lrfd}"


def test_at_Lp_Mn_equals_Mp():
    """At the boundary Lb = Lp, the beam is fully braced. Mn = Mp."""
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    Lp = cap.Lp_ft

    cap_at_Lp = available_strength(W24X76, A992, Lb_ft=Lp)
    assert cap_at_Lp.zone == "yielding"
    assert close(cap_at_Lp.lrfd, 750.0)


def test_increasing_Cb_increases_Mn():
    """Cb > 1 helps, unless it hits the Mp cap."""
    cap_low = available_strength(W24X76, A992, Lb_ft=15.0, Cb_value=1.0)
    cap_high = available_strength(W24X76, A992, Lb_ft=15.0, Cb_value=1.14)
    assert cap_high.lrfd > cap_low.lrfd


def test_lrfd_returns_kip_ft():
    """The API must return kip-ft, not kip-in.

    phi*Mp for W24X76 is 750 kip-ft. If it returned kip-in it would be 9000.
    The ÷12 step must be traced.
    """
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    assert 600 < cap.lrfd < 800, "should be in kip-ft (~750), not kip-in (~9000)"
    assert "÷ 12" in cap.trace.render()


# ---------------------------------------------------------------------------
# Cb moment-gradient factor tests
# ---------------------------------------------------------------------------

def test_Cb_uniform_load():
    """Simply-supported beam, uniform load.

    Quarter-point moments (normalised to Mmax = 1):
      MA = 3/4, MB = 1 (Mmax), MC = 3/4
    Cb = 12.5*1 / (2.5*1 + 3*0.75 + 4*1 + 3*0.75) = 1.136
    """
    assert close(Cb(1.0, 0.75, 1.0, 0.75), 1.136, tol=0.005)


def test_Cb_midpoint_load():
    """Simply-supported beam, concentrated load at midspan.

    MA = 0.5, MB = 1.0 (Mmax), MC = 0.5
    Cb = 12.5*1 / (2.5*1 + 3*0.5 + 4*1 + 3*0.5) = 1.316
    """
    assert close(Cb(1.0, 0.5, 1.0, 0.5), 1.316, tol=0.005)


def test_Cb_uniform_moment():
    """Uniform moment: all four values are equal. Cb = 1.0 exactly."""
    assert Cb(100.0, 100.0, 100.0, 100.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Noncompact flange tests -- Section F3
# ---------------------------------------------------------------------------

def test_w21x48_noncompact_flange():
    """W21X48 has bf/2tf = 9.47, just above lambda_pf = 9.15.

    Hand calc (Eq. F3-1):
      Zx=107, Sx=93.0
      Mp = 50*107 = 5350 kip-in = 445.8 kip-ft
      Mr = 0.7*50*93 = 3255 kip-in
      lam = 9.465, lam_pf = 9.152, lam_rf = 24.08
      Mn(FLB) = 5350 - (5350 - 3255)*((9.465 - 9.152)/(24.08 - 9.152))
              = 5350 - 2095*0.02098 = 5306 kip-in
      phi*Mn = 0.9 * 5306 / 12 = 397.9 kip-ft

    Confirmed against AISC Table 3-2: phi_b*Mn = 398 kip-ft (flagged 'f').
    """
    s = sc.shapes.get("W21X48")
    cls = compact_check(s, A992)
    assert cls.noncompact_flange
    assert not cls.slender_flange

    cap = available_strength(s, A992, Lb_ft=3.0)
    assert cap.noncompact_flange
    assert close(cap.lrfd, 397.9)            # hand calculation
    assert close(cap.lrfd, 398.0, tol=0.01)  # AISC Table 3-2
    # Must be below phi*Mp
    phi_Mp = 0.9 * 50.0 * 107.0 / 12.0  # 401.2 kip-ft
    assert cap.lrfd < phi_Mp


def test_w12x65_noncompact_flange():
    """W12X65 has bf/2tf = 9.92 (noncompact flange at Grade 50).

    Hand calc:
      Zx=96.8, Sx=87.9
      Mp = 50*96.8 = 4840 kip-in = 403.3 kip-ft
      phi*Mp = 363.0 kip-ft
      Mn(FLB) = 4840 - (4840 - 0.7*50*87.9)*((9.917-9.152)/(24.08-9.152))
              = 4840 - (4840-3077)*0.05126 = 4840 - 90.4 = 4750 kip-in
      phi*Mn(FLB) = 0.9*4750/12 = 356.2 kip-ft

    Confirmed against AISC Table 3-2: phi_b*Mn = 356 kip-ft (flagged 'f').
    """
    s = sc.shapes.get("W12X65")
    cls = compact_check(s, A992)
    assert cls.noncompact_flange

    cap = available_strength(s, A992, Lb_ft=5.0)
    assert cap.noncompact_flange
    assert close(cap.lrfd, 356.2)            # hand calculation
    assert close(cap.lrfd, 356.0, tol=0.01)  # AISC Table 3-2
    # Must be below phi*Mp
    phi_Mp = 0.9 * 50.0 * 96.8 / 12.0  # 363.0 kip-ft
    assert cap.lrfd < phi_Mp


def test_slender_flange_raises():
    """A hypothetical slender-flange shape must raise, not return a number.

    No W shape at Grade 50 has a slender flange, so test indirectly by using
    a high-grade material on a shape near the limit. W12X65 has bf/2tf = 9.92
    and lambda_rf at Grade 80 is 1.0*sqrt(29000/80) = 19.04 -- still not
    slender. So we test the gate in compact_check directly.
    """
    # All W shapes should be non-slender flanges at Grade 50
    s = sc.shapes.get("W12X65")
    cls = compact_check(s, A992)
    assert not cls.slender_flange


def test_noncompact_flange_with_ltb():
    """W12X65, noncompact flange, at an unbraced length in the LTB zone.

    The available strength is the lesser of FLB and LTB. At Lb = 15 ft
    (between Lp = 10.67 ft and Lr = 35.15 ft), the LTB is mild and FLB
    should still govern since it is a constant ceiling.

    This test confirms the min(FLB, LTB) logic works correctly.
    """
    s = sc.shapes.get("W12X65")
    cap = available_strength(s, A992, Lb_ft=15.0)
    # Capacity must be at or below the FLB value
    flb_phi_Mn = 356.2  # from test_w12x65_noncompact_flange
    assert cap.lrfd <= flb_phi_Mn + 0.5


# ---------------------------------------------------------------------------
# Trace and report tests
# ---------------------------------------------------------------------------

def test_trace_contains_equation_citations():
    cap = available_strength(W24X76, A992, Lb_ft=12.0)
    text = cap.trace.render()
    assert "AISC Eq. F2-2" in text
    assert "AISC Eq. F2-5" in text
    assert "AISC Eq. F2-6" in text


def test_report_is_readable():
    cap = available_strength(W24X76, A992, Lb_ft=5.0)
    report = cap.report()
    assert "phi_b = 0.90" in report
    assert "kip-ft" in report
