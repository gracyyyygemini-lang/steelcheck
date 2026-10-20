"""steelcheck -- auditable AISC 360 design checks.

Every result carries a calculation trace with equation and table citations, so
an engineer can review the work rather than trust a number. That is also what
makes the LLM layer in ``bench/`` safe: the model parses and explains, the
engine computes, and the two never swap jobs.

Quick start::

    import steelcheck as sc

    shape = sc.shapes.get("W12X65")
    mat = sc.materials.preferred_for("W12X65")      # A992

    x = sc.Axis.unbraced(30, top="fixed", bottom="fixed", name="X")
    y = sc.Axis.braced(30, [15], top="pinned", bottom="fixed", name="Y")

    cap = sc.compression.available_strength(shape, mat, x, y)
    print(cap)            # phi_c*Pn = ... kips
    print(cap.report())   # the full hand calculation
"""

from . import compression, flexure, materials, shapes, tension
from .compression import Axis, K_for, Segment
from .materials import Material
from .shapes import Shape
from .tension import Bolted, Hole, Welded
from .trace import Trace

__version__ = "0.1.0"

__all__ = [
    "compression",
    "flexure",
    "materials",
    "shapes",
    "tension",
    "Axis",
    "Segment",
    "K_for",
    "Material",
    "Shape",
    "Bolted",
    "Welded",
    "Hole",
    "Trace",
]
