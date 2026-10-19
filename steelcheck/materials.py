"""Steel grades.

Values follow AISC Manual Table 2-4 (Applicable ASTM Specifications). Where a
specification gives a range for Fy (A992 is 50-65 ksi), the LOWER bound is the
design value -- using the upper bound would be unconservative.

VERIFY before relying on this in anything real: Fu for the higher A913 grades in
particular should be checked against your edition of Table 2-4. The grades that
matter for a tension or compression check in ARCH 433 are A992 and A572 Gr. 50.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Modulus of elasticity, ksi. The same for every structural steel and every
#: grade -- which is exactly why a slender column gains nothing from higher
#: strength steel (Fy drops out of AISC Eq. E3-3 and E3-4).
E = 29_000.0

#: Shear modulus, ksi. Used by Chapter F; included for completeness.
G = 11_200.0


@dataclass(frozen=True)
class Material:
    name: str
    Fy: float  # ksi, specified minimum yield stress
    Fu: float  # ksi, specified minimum tensile strength

    def __str__(self) -> str:
        return f"{self.name} (Fy={self.Fy:g} ksi, Fu={self.Fu:g} ksi)"


A992 = Material("A992", 50.0, 65.0)
A572_50 = Material("A572 Gr. 50", 50.0, 65.0)
A36 = Material("A36", 36.0, 58.0)
A913_50 = Material("A913 Gr. 50", 50.0, 65.0)
A913_65 = Material("A913 Gr. 65", 65.0, 80.0)
A913_70 = Material("A913 Gr. 70", 70.0, 90.0)
A913_80 = Material("A913 Gr. 80", 80.0, 90.0)
A500_C = Material("A500 Gr. C", 50.0, 62.0)

MATERIALS: dict[str, Material] = {
    m.name: m
    for m in (A992, A572_50, A36, A913_50, A913_65, A913_70, A913_80, A500_C)
}

#: AISC Table 2-4 preferred material specification, by shape family. This is
#: what "assume the preferred material specification" on an exam means.
PREFERRED: dict[str, Material] = {
    "W": A992,
    "M": A992,
    "S": A572_50,
    "HP": A572_50,
    "C": A572_50,
    "MC": A572_50,
    "L": A572_50,
    "WT": A992,
    "MT": A992,
    "ST": A572_50,
    "HSS": A500_C,
}


def preferred_for(designation: str) -> Material:
    """Preferred material spec for a shape designation, e.g. 'W12X65' -> A992."""
    d = designation.upper().strip()
    for family in sorted(PREFERRED, key=len, reverse=True):
        if d.startswith(family) and (
            len(d) == len(family) or not d[len(family)].isalpha()
        ):
            return PREFERRED[family]
    raise KeyError(f"No preferred material known for {designation!r}")


def get(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"Unknown material {name!r}. Known: {', '.join(sorted(MATERIALS))}"
        ) from None
