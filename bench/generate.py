"""Generate the evaluation set.

Gold answers come from two places, and each problem records which:

* ``lecture`` -- the value was stated by the instructor while working the
  example, and in most cases cross-checked against an AISC Manual table. These
  are independent of this code.
* ``engine`` -- computed by steelcheck, which the ``lecture`` problems validate.

The deliberate design choice is that every generated problem is TAGGED with the
traps it contains, so the benchmark can report accuracy per failure mode rather
than one aggregate number. "LLMs get 61% overall" is mildly interesting;
"LLMs get 94% on single-segment columns and 23% once a brace splits the member"
tells you where the failure actually lives.

Run:  python bench/generate.py > bench/problems.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import steelcheck as sc  # noqa: E402
from steelcheck.compression import Axis, available_strength  # noqa: E402
from steelcheck.tension import (  # noqa: E402
    Bolted,
    Hole,
    Welded,
)
from steelcheck.tension import available_strength as tension_strength  # noqa: E402

A992 = sc.materials.A992
A572 = sc.materials.A572_50


def _p(pid, category, traps, prompt, answer, source, tol=0.02):
    return {
        "id": pid,
        "category": category,
        "traps": traps,
        "prompt": prompt,
        "answer": answer,
        "tolerance": tol,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Lecture-verified anchors. These are the problems whose answers do not depend
# on this code being correct.
# ---------------------------------------------------------------------------

def lecture_anchors() -> list[dict]:
    out = []

    out.append(_p(
        "anchor-col-01", "compression", ["governing_direction", "units"],
        "A W12X65 column of A992 steel is 12 ft long and pinned at both ends "
        "about both axes. Determine the available compressive strength by LRFD, "
        "in kips.",
        {"phi_c_Pn": 727.0},
        "lecture 2026-09-17; AISC Table 4-1a reads 728",
    ))

    out.append(_p(
        "anchor-col-02", "compression", ["per_segment_K", "governing_direction"],
        "A W12X65 column of A992 steel is 26 ft tall. About the x-axis it is "
        "fixed at the top and bottom with no intermediate bracing. About the "
        "y-axis it is pinned at the top and bottom and braced 14 ft above the "
        "base. Determine the available compressive strength by LRFD, in kips.",
        {"phi_c_Pn": 686.0},
        "lecture 2026-09-22; AISC Table 4-1a at 14 ft reads 685",
    ))

    out.append(_p(
        "anchor-ten-01", "tension", ["which_limit_state"],
        "A W10X30 tension member of A992 steel is connected through both "
        "flanges and the web with 3/4 in. bolts, so that load is transmitted "
        "directly to all cross-sectional elements. There are 4 holes in the "
        "flanges and 2 in the web at the critical section. Determine the "
        "available tensile strength by LRFD, in kips, and name the governing "
        "limit state.",
        {"phi_Pn": 318.0, "limit_state": "tensile rupture"},
        "lecture 2026-09-10",
    ))

    out.append(_p(
        "anchor-ten-02", "tension", ["n_minus_1", "hole_count", "xbar_vs_ybar"],
        "A W10X30 tension member of A992 steel is connected through the "
        "FLANGES ONLY with 3/4 in. bolts, 2 per flange per row and 5 bolts per "
        "line at 3 in. on center. Take xbar = 1.10 in. Determine the available "
        "tensile strength by LRFD, in kips.",
        {"phi_Pn": 312.0},
        "lecture 2026-09-11",
    ))

    out.append(_p(
        "anchor-ten-03", "tension", ["xbar_vs_ybar", "bolt_fit"],
        "An L6X4X1/2 tension member of A572 Gr. 50 steel is connected through "
        "the SHORT leg with 3/4 in. bolts at 3 in. on center, 6 bolts per line. "
        "A 4 in. leg accommodates one bolt per row. Determine the available "
        "tensile strength by LRFD, in kips.",
        {"phi_Pn": 182.0},
        "lecture 2026-09-11",
    ))

    out.append(_p(
        "anchor-ten-04", "tension", ["which_limit_state", "weld_net_area"],
        "A C10X30 tension member of A572 Gr. 50 steel is connected at the web "
        "with longitudinal welds 12 in. long in combination with transverse "
        "welds. Take xbar = 0.649 in. Determine the available tensile strength "
        "by LRFD, in kips, and name the governing limit state.",
        {"phi_Pn": 396.0, "limit_state": "tensile yielding"},
        "lecture 2026-09-13 -- the one case where YIELDING governs",
    ))

    out.append(_p(
        "anchor-ten-05", "tension", ["weld_case_4"],
        "A C10X30 tension member of A572 Gr. 50 steel is connected at the web "
        "with LONGITUDINAL WELDS ONLY, 12 in. long, separated by 10 in. Take "
        "xbar = 0.649 in. Determine the available tensile strength by LRFD, in "
        "kips.",
        {"phi_Pn": 330.0},
        "lecture 2026-09-13",
    ))

    out.append(_p(
        "anchor-load-01", "loads", ["lrfd_vs_asd"],
        "A tension member carries PD = 35 kips and PL = 95 kips. Calculate the "
        "required tensile strength by (a) LRFD and (b) ASD, in kips.",
        {"Pu": 194.0, "Pa": 130.0},
        "Quiz 3, 2026-09-25",
    ))

    return out


# ---------------------------------------------------------------------------
# Generated problems, gold answers from the validated engine.
# ---------------------------------------------------------------------------

COLUMN_SHAPES = ["W12X65", "W14X90"]


def generated_columns() -> list[dict]:
    out, n = [], 0

    # Single segment, varying end conditions and length.
    for desig in COLUMN_SHAPES:
        shape = sc.shapes.get(desig)
        for length in (10, 16, 22, 28, 34):
            for top, bottom, word in (
                ("pinned", "pinned", "pinned at both ends"),
                ("fixed", "fixed", "fixed at both ends"),
                ("fixed", "pinned", "fixed at the base and pinned at the top"),
            ):
                n += 1
                x = Axis.unbraced(length, top, bottom)
                y = Axis.unbraced(length, top, bottom)
                cap = available_strength(shape, A992, x, y)
                traps = ["units"]
                if not cap.inelastic:
                    traps.append("elastic_branch")
                if (top, bottom) != ("pinned", "pinned"):
                    traps.append("recommended_K")
                out.append(_p(
                    f"gen-col-{n:03d}", "compression", traps,
                    f"A {desig} column of A992 steel is {length} ft long and "
                    f"{word} about both axes, with no intermediate bracing. "
                    "Determine the available compressive strength by LRFD, in "
                    "kips.",
                    {"phi_c_Pn": round(cap.lrfd, 1)},
                    "engine",
                ))

    # Different bracing in each direction -- the equivalent-length case.
    for desig in COLUMN_SHAPES:
        shape = sc.shapes.get(desig)
        for total, brace in ((30, 15), (30, 12), (36, 20), (24, 10)):
            n += 1
            x = Axis.unbraced(total, "fixed", "fixed")
            y = Axis.braced(total, [brace], top="pinned", bottom="fixed")
            cap = available_strength(shape, A992, x, y)
            out.append(_p(
                f"gen-col-{n:03d}", "compression",
                ["per_segment_K", "governing_direction", "units", "recommended_K"],
                f"A {desig} column of A992 steel is {total} ft tall. About the "
                "x-axis it is fixed at the top and bottom with no intermediate "
                "bracing. About the y-axis it is pinned at the top, fixed at the "
                f"base, and braced {brace} ft above the base. Determine the "
                "available compressive strength by LRFD, in kips, and state "
                "which axis governs.",
                {"phi_c_Pn": round(cap.lrfd, 1), "governing_axis": cap.governing_axis},
                "engine",
            ))

    # Two braces -- three segments, mixed K.
    for desig in COLUMN_SHAPES:
        shape = sc.shapes.get(desig)
        for total, b1, b2 in ((44, 18, 32), (40, 12, 26)):
            n += 1
            x = Axis.unbraced(total, "fixed", "fixed")
            y = Axis.braced(total, [b1, b2], top="pinned", bottom="fixed")
            cap = available_strength(shape, A992, x, y)
            out.append(_p(
                f"gen-col-{n:03d}", "compression",
                ["per_segment_K", "multi_segment", "governing_direction", "units"],
                f"A {desig} column of A992 steel is {total} ft tall. About the "
                "x-axis it is fixed top and bottom with no intermediate bracing. "
                "About the y-axis it is pinned at the top, fixed at the base, "
                f"and braced at {b1} ft and {b2} ft above the base. Determine the "
                "available compressive strength by LRFD, in kips.",
                {"phi_c_Pn": round(cap.lrfd, 1)},
                "engine",
            ))

    # Grade dependence -- the limits move, the geometry does not.
    for grade in (sc.materials.A913_65, sc.materials.A913_80):
        n += 1
        shape = sc.shapes.get("W12X65")
        x = y = Axis.unbraced(14, "pinned", "pinned")
        cap = available_strength(shape, grade, x, y)
        out.append(_p(
            f"gen-col-{n:03d}", "compression",
            ["grade_dependent_lambda", "units"],
            f"A W12X65 column of {grade.name} steel (Fy = {grade.Fy:g} ksi) is "
            "14 ft long, pinned at both ends about both axes. First confirm the "
            "section is non-slender at this grade, then determine the available "
            "compressive strength by LRFD, in kips.",
            {"phi_c_Pn": round(cap.lrfd, 1)},
            "engine",
        ))

    return out


def generated_tension() -> list[dict]:
    out, n = [], 0
    L6 = sc.shapes.get("L6X4X1/2")
    W10 = sc.shapes.get("W10X30")

    for n_bolts in (3, 4, 5, 6, 7):
        for leg, xb, holes, legname in (
            ("long", L6.xbar, 2, "LONG"),
            ("short", L6.ybar, 1, "SHORT"),
        ):
            n += 1
            conn = Bolted(
                db=0.75, holes=[Hole(holes, 0.5)], n_per_line=n_bolts,
                spacing=3.0, xbar=xb,
            )
            res = tension_strength(L6, A572, conn)
            out.append(_p(
                f"gen-ten-{n:03d}", "tension",
                ["xbar_vs_ybar", "n_minus_1", "hole_count", "which_limit_state"],
                f"An L6X4X1/2 tension member of A572 Gr. 50 steel is connected "
                f"through the {legname} leg with 3/4 in. bolts at 3 in. on "
                f"center, {n_bolts} bolts per line, {holes} bolt(s) per row. "
                "Determine the available tensile strength by LRFD, in kips, and "
                "name the governing limit state.",
                {
                    "phi_Pn": round(res.lrfd, 1),
                    "limit_state": res.limit_state,
                },
                "engine",
            ))

    for n_bolts in (3, 5, 7):
        n += 1
        conn = Bolted(
            db=0.875, holes=[Hole(4, 0.510)], n_per_line=n_bolts,
            spacing=3.0, xbar=1.10,
        )
        res = tension_strength(W10, A992, conn)
        out.append(_p(
            f"gen-ten-{n:03d}", "tension",
            ["hole_count", "n_minus_1", "hole_allowance"],
            "A W10X30 tension member of A992 steel is connected through the "
            f"FLANGES ONLY with 7/8 in. bolts, {n_bolts} per line at 3 in. on "
            "center, 2 per flange per row. Take xbar = 1.10 in. Determine the "
            "available tensile strength by LRFD, in kips.",
            {"phi_Pn": round(res.lrfd, 1)},
            "engine",
        ))

    for weld_len in (8.0, 12.0, 18.0):
        n += 1
        C10 = sc.shapes.get("C10X30")
        conn = Welded(longitudinal=True, L=weld_len, w=10.0, xbar=C10.xbar)
        res = tension_strength(C10, A572, conn)
        out.append(_p(
            f"gen-ten-{n:03d}", "tension",
            ["weld_case_4", "weld_net_area", "which_limit_state"],
            "A C10X30 tension member of A572 Gr. 50 steel is connected at the "
            f"web with LONGITUDINAL WELDS ONLY, {weld_len:g} in. long, separated "
            "by 10 in. Take xbar = 0.649 in. Determine the available tensile "
            "strength by LRFD, in kips, and name the governing limit state.",
            {"phi_Pn": round(res.lrfd, 1), "limit_state": res.limit_state},
            "engine",
        ))

    return out


def build() -> dict:
    problems = lecture_anchors() + generated_columns() + generated_tension()
    traps: dict[str, int] = {}
    for p in problems:
        for t in p["traps"]:
            traps[t] = traps.get(t, 0) + 1
    return {
        "schema": 1,
        "n": len(problems),
        "trap_counts": dict(sorted(traps.items(), key=lambda kv: -kv[1])),
        "notes": (
            "Gold answers marked source='lecture' are independent of this code. "
            "Those marked 'engine' were computed by steelcheck, which the "
            "lecture-anchored problems validate (see tests/)."
        ),
        "problems": problems,
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
