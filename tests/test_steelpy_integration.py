"""Re-run the validation cases against real AISC data instead of the seed set.

``tests/test_tension.py`` and ``tests/test_compression.py`` run against the
small seed set hardcoded in ``shapes.py``. This module loads the full section
property database through ``steelpy`` and checks that the same published
capacities still come out.

That makes it a three-way agreement: the published AISC table value, the
instructor's hand calculation, and this engine running on the official section
properties. Any one of the three being wrong would break it.

Skipped automatically if steelpy is not installed.
"""

import pytest

import steelcheck as sc
from steelcheck.compression import Axis
from steelcheck.compression import available_strength as column_strength
from steelcheck.tension import Bolted, Hole, Welded
from steelcheck.tension import available_strength as tension_strength

steelpy = pytest.importorskip("steelpy", reason="pip install steelpy")

A992 = sc.materials.A992
A572 = sc.materials.A572_50
TOL = 0.005


@pytest.fixture(scope="module", autouse=True)
def load_real_data():
    """Replace the seed set with the full database for this module."""
    n = sc.shapes.load_from_steelpy()
    assert n > 1000, f"expected the full database, got {n} shapes"
    yield


def close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(actual - expected) / expected <= tol


# ---------------------------------------------------------------------------
# The seed values were transcribed from lecture. Confirm they were right.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "designation,field,expected",
    [
        ("W12X65", "A", 19.1), ("W12X65", "rx", 5.28), ("W12X65", "ry", 3.02),
        ("W24X76", "A", 22.4), ("W24X76", "Zx", 200.0), ("W24X76", "Sx", 176.0),
        ("W24X76", "ry", 1.92),
        ("W10X30", "A", 8.84), ("W10X30", "tf", 0.510), ("W10X30", "tw", 0.300),
        ("C10X30", "A", 8.81), ("C10X30", "xbar", 0.649),
        ("L6X4X1/2", "A", 4.75), ("L6X4X1/2", "xbar", 0.981),
        ("L6X4X1/2", "ybar", 1.98), ("L6X4X1/2", "t", 0.5),
        ("WT5X15", "ybar", 1.10),
        # Both of these were inferred rather than transcribed, and flagged as
        # needing verification. They were correct.
        ("W14X90", "A", 26.5),
        ("WT5X15", "A", 4.42),
    ],
)
def test_seed_values_match_official_data(designation, field, expected):
    actual = getattr(sc.shapes.get(designation), field)
    assert actual == pytest.approx(expected, rel=0.01)


def test_derived_slenderness_ratios_match_published():
    """bf/2tf and h/tw are computed, not stored. They must match the Manual."""
    w12 = sc.shapes.get("W12X65")
    assert close(w12.bf_2tf, 9.92, tol=0.01)
    assert close(w12.h_tw, 24.9, tol=0.01)

    w24 = sc.shapes.get("W24X76")
    assert close(w24.bf_2tf, 6.61, tol=0.01)
    assert close(w24.h_tw, 49.0, tol=0.01)


def test_w14x90_rx_ry_matches_quoted_ratio():
    assert close(sc.shapes.get("W14X90").rx_ry, 1.66, tol=0.01)


# ---------------------------------------------------------------------------
# Capacities, recomputed on official data
# ---------------------------------------------------------------------------

def test_w12x65_pinned_12ft_on_real_data():
    x = y = Axis.unbraced(12, "pinned", "pinned")
    cap = column_strength(sc.shapes.get("W12X65"), A992, x, y)
    assert close(cap.lrfd, 727.0)      # instructor's hand calculation
    assert close(cap.lrfd, 728.0, tol=0.01)  # AISC Table 4-1a
    assert close(cap.asd, 484.0)


def test_w12x65_mixed_bracing_on_real_data():
    x = Axis.unbraced(26, "fixed", "fixed")
    y = Axis.braced(26, [14], top="pinned", bottom="pinned")
    cap = column_strength(sc.shapes.get("W12X65"), A992, x, y)
    assert close(cap.lrfd, 686.0)
    assert close(cap.lrfd, 685.0, tol=0.01)  # Table 4-1a at 14 ft


def test_w10x30_tension_series_on_real_data():
    s = sc.shapes.get("W10X30")

    conn = Bolted(
        db=0.75, holes=[Hole(4, s.tf), Hole(2, s.tw)], all_elements_connected=True
    )
    res = tension_strength(s, A992, conn)
    assert close(res.yielding.lrfd, 398.0)
    assert close(res.rupture.lrfd, 318.0)

    conn = Bolted(
        db=0.75, holes=[Hole(4, s.tf)], n_per_line=5, spacing=3.0,
        xbar=sc.shapes.get("WT5X15").ybar,
    )
    assert close(tension_strength(s, A992, conn).rupture.lrfd, 312.0)

    conn = Welded(transverse=True, connected_area=2 * s.bf * s.tf)
    assert close(tension_strength(s, A992, conn).rupture.lrfd, 289.0)


def test_angle_tension_on_real_data():
    s = sc.shapes.get("L6X4X1/2")

    long_leg = Bolted(
        db=0.75, holes=[Hole(2, s.t)], n_per_line=5, spacing=3.0, xbar=s.xbar
    )
    res = tension_strength(s, A572, long_leg)
    assert close(res.yielding.lrfd, 214.0)
    assert close(res.rupture.lrfd, 174.0)

    short_leg = Bolted(
        db=0.75, holes=[Hole(1, s.t)], n_per_line=6, spacing=3.0, xbar=s.ybar
    )
    assert close(tension_strength(s, A572, short_leg).rupture.lrfd, 182.0)


def test_channel_welded_on_real_data():
    s = sc.shapes.get("C10X30")

    both = Welded(longitudinal=True, transverse=True, L=12.0, xbar=s.xbar)
    res = tension_strength(s, A572, both)
    assert close(res.yielding.lrfd, 396.0)
    assert close(res.rupture.lrfd, 406.0)
    assert res.limit_state == "tensile yielding"

    long_only = Welded(longitudinal=True, L=12.0, w=s.d, xbar=s.xbar)
    assert close(tension_strength(s, A572, long_only).rupture.lrfd, 330.0)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_full_database_loaded():
    shapes = sc.shapes.all_shapes()
    assert len(shapes) > 1500
    families = {s.family for s in shapes}
    assert {"W", "C", "L", "WT", "HSS"} <= families


def test_every_w_shape_has_what_a_column_check_needs():
    missing = [
        s.designation
        for s in sc.shapes.family("W")
        if None in (s.A, s.rx, s.ry, s.bf_2tf, s.h_tw)
    ]
    assert not missing, f"incomplete W records: {missing[:5]}"


def test_column_check_runs_across_the_whole_W_series():
    """Smoke test across all 289 W shapes: nothing crashes or goes unphysical."""
    x = y = Axis.unbraced(14, "pinned", "pinned")
    checked = rejected = 0
    for s in sc.shapes.family("W"):
        try:
            cap = column_strength(s, A992, x, y)
        except ValueError:
            rejected += 1  # slender-element, correctly sent to Section E7
            continue
        assert 0 < cap.lrfd < cap.squash_load, s.designation
        assert cap.Fn < A992.Fy, s.designation
        checked += 1

    assert checked > 150
    assert checked + rejected == len(sc.shapes.family("W"))


def test_thin_webbed_beam_shapes_are_rejected_as_columns():
    """About a third of W shapes are slender-element in uniform compression.

    That is the right answer, not a gap in coverage. lambda_r for the web is
    1.49*sqrt(E/Fy) = 35.9 at Grade 50, and a W18X35 has h/tw = 53.5 -- it is a
    beam section, and nobody frames a column out of one.

    Note the same shape is COMPACT in flexure, where the web limits are
    lambda_p = 90.6 and lambda_r = 137. The limits depend on how the member is
    loaded, which is exactly the kind of distinction a language model flattens.
    """
    from steelcheck.compression import classify

    for designation in ("W18X35", "W12X14", "W36X135", "W44X230"):
        s = sc.shapes.get(designation)
        assert s.h_tw > 35.9
        assert not classify(s, A992).non_slender, designation

    # Shapes actually used as columns pass comfortably.
    for designation in ("W12X65", "W14X90", "W14X176", "W10X49"):
        assert classify(sc.shapes.get(designation), A992).non_slender, designation


def test_slender_element_section_raises_rather_than_returning_a_number():
    """Silently applying E3 to a slender section would be unconservative."""
    x = y = Axis.unbraced(14, "pinned", "pinned")
    with pytest.raises(ValueError, match="slender"):
        column_strength(sc.shapes.get("W18X35"), A992, x, y)
