"""Validation against published and lecture-verified column capacities.

The W12X65 cases are the strongest evidence in the suite: the instructor worked
them by hand from Eqs. E3-1 through E3-4 AND read off the corresponding AISC
Table 4-1a entry, so each one has two independent published values to hit.

The W14X90 cases check the equivalent-effective-length logic, where the quoted
Lcy_eq values (16.9 ft and 24.1 ft) are exact.
"""

import pytest

import steelcheck as sc
from steelcheck.compression import (
    Axis,
    K_for,
    Segment,
    available_strength,
    classify,
    critical_direction,
)

A992 = sc.materials.A992
W12X65 = sc.shapes.get("W12X65")
W14X90 = sc.shapes.get("W14X90")

TOL = 0.005


def close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(actual - expected) / expected <= tol


# ---------------------------------------------------------------------------
# W12X65, pinned both ends, 12 ft. Lecture 2026-09-17.
# Hand calculation gave 727 / 484; AISC Table 4-1a reads 728 / 484.
# ---------------------------------------------------------------------------

def test_w12x65_pinned_12ft():
    x = Axis.unbraced(12, "pinned", "pinned", name="X")
    y = Axis.unbraced(12, "pinned", "pinned", name="Y")
    cap = available_strength(W12X65, A992, x, y)

    assert cap.governing_axis == "y", "equal lengths -> Y always governs"
    assert close(cap.Lc_over_r, 47.7)
    assert close(cap.Fe, 126.0)
    assert close(cap.Fn, 42.3)
    assert close(cap.lrfd, 727.0)
    assert close(cap.asd, 484.0)
    assert cap.inelastic, "Lc/r = 47.7 < 113, so inelastic buckling"


def test_w12x65_table_4_1a_cross_check():
    """phi_c*Pn must land on the published Table 4-1a value of 728 kips."""
    x = y = Axis.unbraced(12, "pinned", "pinned")
    cap = available_strength(W12X65, A992, x, y)
    assert close(cap.lrfd, 728.0, tol=0.01)


# ---------------------------------------------------------------------------
# W12X65, braced differently in each direction. Lecture 2026-09-22.
# Y: pinned top and bottom, braced 14 ft up a 26 ft column -> segments 14 and 12.
# X: fixed top and bottom over the full 26 ft, no intermediate brace.
# Hand calculation gave 686; Table 4-1a at 14 ft reads 685.
# ---------------------------------------------------------------------------

def test_w12x65_mixed_bracing():
    x = Axis.unbraced(26, "fixed", "fixed", name="X")
    y = Axis.braced(26, [14], top="pinned", bottom="pinned", name="Y")

    assert close(x.Lc, 16.9)  # 0.65 x 26
    assert close(y.Lc, 14.0)  # max(14, 12), both segments K = 1.0

    cap = available_strength(W12X65, A992, x, y)
    assert cap.governing_axis == "y"
    assert close(cap.Lc_over_r, 55.6)
    assert close(cap.lrfd, 686.0)


# ---------------------------------------------------------------------------
# W14X90, three bracing scenarios. Lecture 2026-09-24. rx/ry = 1.66.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "Lcx,expected_eq,expected_axis",
    [
        (14.0, 8.44, "y"),   # equal lengths -> Y
        (28.0, 16.9, "x"),   # instructor quoted 16.9 ft
        (40.0, 24.1, "x"),   # instructor quoted 24.1 ft
    ],
)
def test_w14x90_equivalent_length(Lcx, expected_eq, expected_axis):
    x = Axis.unbraced(Lcx, "pinned", "pinned", name="X")
    y = Axis.unbraced(14, "pinned", "pinned", name="Y")
    d = critical_direction(W14X90, x, y)
    assert close(d.Lcy_eq, expected_eq, tol=0.01)
    assert d.governs == expected_axis


def test_w14x90_rx_ry_matches_table():
    assert close(W14X90.rx_ry, 1.66, tol=0.01)


# ---------------------------------------------------------------------------
# Per-segment K -- the error this library exists to prevent.
# ---------------------------------------------------------------------------

def test_brace_acts_as_a_pin_and_each_segment_gets_its_own_K():
    """30 ft column, fixed base, pinned top, braced at mid-height.

    Both segments are 15 ft, but the lower one has a fixed end, so K = 0.80 and
    its Lc is only 12 ft. Equal lengths do NOT mean equal effective lengths.
    """
    y = Axis.braced(30, [15], top="pinned", bottom="fixed")
    lengths = sorted(s.Lc for s in y.segments)
    assert lengths == pytest.approx([12.0, 15.0])
    assert y.Lc == pytest.approx(15.0)

    lower = [s for s in y.segments if s.Lc == pytest.approx(12.0)][0]
    assert lower.K == 0.80, "fixed-pinned segment"


def test_longest_segment_is_not_always_critical():
    """A LONGER segment with a fixed end loses to a SHORTER pinned one.

    30 ft column, fixed base, pinned top, braced 16 ft up:
      lower 16 ft, fixed-pinned, K = 0.80 -> Lc = 12.8 ft
      upper 14 ft, pinned-pinned, K = 1.00 -> Lc = 14.0 ft   <- governs

    Pick by length and you take 16 ft. Pick by K*L and you take the 14 ft
    segment. Only the second is right.
    """
    y = Axis.braced(30, [16], top="pinned", bottom="fixed")
    lower = min(y.segments, key=lambda s: s.Lc)
    upper = max(y.segments, key=lambda s: s.Lc)

    assert lower.length == 16.0 and lower.Lc == pytest.approx(12.8)
    assert upper.length == 14.0 and upper.Lc == pytest.approx(14.0)

    by_length = max(y.segments, key=lambda s: s.length)
    by_Lc = max(y.segments, key=lambda s: s.Lc)
    assert by_length is not by_Lc, "compare K*L, never L"
    assert y.Lc == pytest.approx(14.0)


def test_recommended_K_values():
    assert K_for("pinned", "pinned") == 1.0
    assert K_for("fixed", "fixed") == 0.65
    assert K_for("fixed", "pinned") == 0.80
    assert K_for("braced", "fixed") == 0.80  # a brace is a pin
    # Theoretical values differ and are never used for design.
    assert K_for("fixed", "fixed", use_recommended=False) == 0.5


# ---------------------------------------------------------------------------
# Classification gate and grade dependence
# ---------------------------------------------------------------------------

def test_w12x65_is_non_slender_in_gr50():
    c = classify(W12X65, A992)
    assert c.non_slender
    assert close(c.flange_limit, 13.5, tol=0.01)
    assert close(c.web_limit, 35.9, tol=0.01)


def test_slenderness_limits_tighten_with_grade():
    """Assignment 5 Q1: the geometry never changes, the limits do."""
    limits = {
        m.name: (classify(W12X65, m).flange_limit, classify(W12X65, m).web_limit)
        for m in (sc.materials.A992, sc.materials.A913_65, sc.materials.A913_80)
    }
    f50, w50 = limits["A992"]
    f65, w65 = limits["A913 Gr. 65"]
    f80, w80 = limits["A913 Gr. 80"]
    assert f50 > f65 > f80
    assert w50 > w65 > w80
    assert close(f80, 10.7, tol=0.02)
    assert close(w80, 28.4, tol=0.02)


def test_short_long_threshold_by_grade():
    from math import sqrt

    from steelcheck.materials import E

    def thresh(Fy):
        return 4.71 * sqrt(E / Fy)

    assert close(thresh(36), 134.0, tol=0.01)
    assert close(thresh(50), 113.0, tol=0.01)
    assert close(thresh(80), 89.7, tol=0.01)


# ---------------------------------------------------------------------------
# Elastic branch and physical sanity
# ---------------------------------------------------------------------------

def test_long_column_is_independent_of_grade():
    """Eq. E3-3 contains no Fy, so a slender column gains nothing from a
    higher grade. This is the whole answer to 'should we switch to Gr. 80?'."""
    from steelcheck.compression import nominal_stress
    from steelcheck.trace import Trace

    ratio = 150.0  # well past 113 and past 89.7
    fn50, _, inel50 = nominal_stress(sc.materials.A992, ratio, Trace("t"))
    fn80, _, inel80 = nominal_stress(sc.materials.A913_80, ratio, Trace("t"))
    assert not inel50 and not inel80
    assert fn50 == pytest.approx(fn80)


def test_Fn_never_exceeds_Fy():
    """A column always buckles before it fully yields."""
    for L in (4, 8, 12, 20, 30, 40):
        x = y = Axis.unbraced(L, "pinned", "pinned")
        cap = available_strength(W12X65, A992, x, y)
        assert cap.Fn < A992.Fy


def test_capacity_never_exceeds_the_squash_load():
    x = y = Axis.unbraced(4, "pinned", "pinned")
    cap = available_strength(W12X65, A992, x, y)
    assert cap.lrfd < cap.squash_load


def test_capacity_decreases_monotonically_with_length():
    lengths = [8, 12, 16, 20, 24, 30, 36]
    caps = []
    for L in lengths:
        x = y = Axis.unbraced(L, "pinned", "pinned")
        caps.append(available_strength(W12X65, A992, x, y).lrfd)
    assert caps == sorted(caps, reverse=True)


def test_slenderness_ratio_uses_inches():
    """Guards the single most common error: forgetting the x12."""
    x = y = Axis.unbraced(12, "pinned", "pinned")
    cap = available_strength(W12X65, A992, x, y)
    # 12 ft / 3.02 in would give 3.97 -- off by a factor of 12.
    assert cap.Lc_over_r > 40
    assert "x 12" in cap.trace.render()


def test_compression_uses_gross_area():
    x = y = Axis.unbraced(12, "pinned", "pinned")
    text = available_strength(W12X65, A992, x, y).trace.render()
    assert "GROSS area" in text
