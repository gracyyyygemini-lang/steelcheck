"""Validation against published and lecture-verified tension capacities.

Every expected value below was stated aloud by the instructor while working the
example in an ARCH 433 lecture, which makes it ground truth independent of this
code. The W10X30 series is the same shape under six different connections -- it
is the cleanest demonstration in the whole suite that YIELDING NEVER CHANGES
while rupture moves by 10%.
"""

import pytest

import steelcheck as sc
from steelcheck.tension import Bolted, Hole, Welded, available_strength, tensile_yielding

A992 = sc.materials.A992
A572 = sc.materials.A572_50

TOL = 0.005  # 0.5% -- tighter than the 3-significant-figure answers being checked


def close(actual: float, expected: float, tol: float = TOL) -> bool:
    return abs(actual - expected) / expected <= tol


# ---------------------------------------------------------------------------
# W10X30 -- one shape, four connections. Lecture 2026-09-10 and 2026-09-11.
# ---------------------------------------------------------------------------

W10X30 = sc.shapes.get("W10X30")


def test_w10x30_yielding_is_398():
    cap = tensile_yielding(W10X30, A992)
    assert close(cap.lrfd, 398.0)


@pytest.mark.parametrize(
    "label,conn,expected",
    [
        (
            "bolted through flanges AND web, all elements connected",
            Bolted(
                db=0.75,
                holes=[Hole(4, 0.510), Hole(2, 0.300)],
                all_elements_connected=True,
            ),
            318.0,
        ),
        (
            "bolted through flanges only, 5 bolts per line at 3 in.",
            Bolted(
                db=0.75,
                holes=[Hole(4, 0.510)],
                n_per_line=5,
                spacing=3.0,
                xbar=1.10,  # ybar of the WT5X15, i.e. half the W10X30
            ),
            312.0,
        ),
        (
            "transverse welds only, to the flanges",
            Welded(transverse=True, connected_area=2 * 5.81 * 0.510),
            289.0,
        ),
    ],
)
def test_w10x30_rupture_series(label, conn, expected):
    res = available_strength(W10X30, A992, conn)
    assert close(res.rupture.lrfd, expected), label
    # In every one of these, rupture is the smaller number and therefore governs.
    assert res.limit_state == "tensile rupture", label


def test_yielding_is_independent_of_connection():
    """The structural point of the W10X30 series, asserted directly."""
    connections = [
        Bolted(db=0.75, holes=[Hole(4, 0.510), Hole(2, 0.300)], all_elements_connected=True),
        Bolted(db=0.75, holes=[Hole(4, 0.510)], n_per_line=5, spacing=3.0, xbar=1.10),
        Welded(transverse=True, connected_area=2 * 5.81 * 0.510),
    ]
    values = {available_strength(W10X30, A992, c).yielding.lrfd for c in connections}
    assert len(values) == 1, "yielding must not depend on the connection"


# ---------------------------------------------------------------------------
# L6X4X1/2 -- the leg you connect decides whether you use xbar or ybar, and how
# many bolts fit. Lecture 2026-09-11.
# ---------------------------------------------------------------------------

L6X4 = sc.shapes.get("L6X4X1/2")


def test_angle_long_leg_connected():
    # 6-in. leg takes two bolts (Table 1-7A lists g1 and g2); 5 per line at 3 in.
    conn = Bolted(
        db=0.75, holes=[Hole(2, 0.5)], n_per_line=5, spacing=3.0, xbar=L6X4.xbar
    )
    res = available_strength(L6X4, A572, conn)
    assert close(res.yielding.lrfd, 214.0)
    assert close(res.rupture.lrfd, 174.0)
    assert res.limit_state == "tensile rupture"


def test_angle_short_leg_uses_ybar_and_one_bolt():
    # A 4-in. leg lists a single gage, so only ONE bolt fits -- and the
    # eccentricity becomes ybar, not xbar.
    conn = Bolted(
        db=0.75, holes=[Hole(1, 0.5)], n_per_line=6, spacing=3.0, xbar=L6X4.ybar
    )
    res = available_strength(L6X4, A572, conn)
    assert close(res.rupture.lrfd, 182.0)


def test_swapping_xbar_for_ybar_changes_the_answer():
    """Guards the trap: using the wrong centroid is silently plausible."""
    common = dict(db=0.75, holes=[Hole(2, 0.5)], n_per_line=5, spacing=3.0)
    right = available_strength(L6X4, A572, Bolted(**common, xbar=L6X4.xbar))
    wrong = available_strength(L6X4, A572, Bolted(**common, xbar=L6X4.ybar))
    assert not close(right.rupture.lrfd, wrong.rupture.lrfd, tol=0.02)


# ---------------------------------------------------------------------------
# C10X30 -- the one case in the set where YIELDING governs, plus the Case 4
# longitudinal-only formula. Lecture 2026-09-13.
# ---------------------------------------------------------------------------

C10X30 = sc.shapes.get("C10X30")


def test_channel_longitudinal_plus_transverse_welds():
    conn = Welded(longitudinal=True, transverse=True, L=12.0, xbar=C10X30.xbar)
    res = available_strength(C10X30, A572, conn)
    assert close(res.yielding.lrfd, 396.0)
    assert close(res.rupture.lrfd, 406.0)
    # Rupture does NOT always govern -- this is the counterexample.
    assert res.limit_state == "tensile yielding"


def test_channel_longitudinal_welds_only_case_4():
    conn = Welded(longitudinal=True, L=12.0, w=10.0, xbar=C10X30.xbar)
    res = available_strength(C10X30, A572, conn)
    assert close(res.rupture.lrfd, 330.0)
    assert res.limit_state == "tensile rupture"


def test_longitudinal_welds_give_An_equal_to_Ag():
    conn = Welded(longitudinal=True, transverse=True, L=12.0, xbar=C10X30.xbar)
    text = available_strength(C10X30, A572, conn).rupture.trace.render()
    assert "An = Ag" in text


# ---------------------------------------------------------------------------
# Connection geometry and load combinations
# ---------------------------------------------------------------------------

def test_connection_length_is_n_minus_one_spaces():
    """Five bolts at 3 in. is 12 in., not 15. The classic off-by-one."""
    assert Bolted(db=0.75, n_per_line=5, spacing=3.0).L == 12.0
    assert Bolted(db=0.75, n_per_line=6, spacing=3.0).L == 15.0
    assert Bolted(db=0.75, n_per_line=1, spacing=3.0).L == 0.0


def test_hole_allowance_is_one_eighth():
    conn = Bolted(db=0.75, holes=[Hole(2, 0.5)], n_per_line=5, spacing=3.0, xbar=0.981)
    text = available_strength(L6X4, A572, conn).rupture.trace.render()
    assert "0.875" in text  # 0.75 + 1/8, not 0.75 + 1/16


def test_required_strength_lrfd_and_asd():
    # Quiz 3, 2026-09-25: PD = 35 kips, PL = 95 kips.
    Pu, _ = sc.tension.required_strength(35, 95, method="LRFD")
    Pa, _ = sc.tension.required_strength(35, 95, method="ASD")
    assert close(Pu, 194.0)
    assert close(Pa, 130.0)


def test_missing_xbar_raises_rather_than_guessing():
    conn = Bolted(db=0.75, holes=[Hole(2, 0.5)], n_per_line=5, spacing=3.0)
    with pytest.raises(ValueError, match="xbar"):
        available_strength(L6X4, A572, conn)


def test_transverse_only_weld_requires_connected_area():
    with pytest.raises(ValueError, match="connected_area"):
        available_strength(W10X30, A992, Welded(transverse=True))


def test_preferred_material_lookup():
    assert sc.materials.preferred_for("W12X65").name == "A992"
    assert sc.materials.preferred_for("L6X4X1/2").name == "A572 Gr. 50"
    assert sc.materials.preferred_for("C10X30").name == "A572 Gr. 50"
