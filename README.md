# steelcheck

**Auditable AISC 360 design checks, and a benchmark for whether language models can do them.**

A structural engineer stamps a drawing and becomes personally liable for it. That makes a confident, uncited number from a language model worse than useless to them — which is why AI adoption in structural engineering is stalled on trust rather than on capability.

This repository is two things:

1. A small, tested Python engine that performs AISC 360 tension, compression and flexure checks and **returns the full hand calculation** — every intermediate value, with its equation and table citation.
2. A benchmark that measures how well a language model does the same job **with and without** that engine, broken down by the specific reasoning step that fails.

The second is the point. The first exists to make it measurable.

---

## The finding this is built to produce

Structural design is a bad fit for a language model working alone, and the failure modes are specific rather than diffuse:

- **Recall.** Section properties have to come from AISC Table 1-1. A model will produce a plausible `r_y` for a W12×65 that is simply wrong, and nothing downstream will catch it.
- **Unit discipline.** The slenderness ratio `L_c/r` needs the length in inches while every other part of the problem is in feet. Miss the ×12 and the answer is twelve times too slender and wildly unconservative.
- **Branch selection.** Short versus long column, bolted versus welded net area, four separate shear-lag cases — each one a fork the model must take correctly before any arithmetic matters.
- **Structural reasoning over geometry.** A lateral brace acts as a pin, so each unbraced segment carries *its own* K factor. Two 15 ft segments can have effective lengths of 15 ft and 12 ft. Picking the longer segment instead of the larger `K·L` is the single most common error, and it is invisible in the output.

`bench/problems.json` tags all 68 problems by which of these traps they contain, so the benchmark reports accuracy **per failure mode** instead of one aggregate number. Those 68 cover compression (46), tension (21) and load combinations (1) only — the benchmark does not yet exercise Chapter F, so nothing it reports says anything about how models handle flexure. "Models score 61%" is a shrug. "Models score 94% on single-segment columns and 23% once a brace splits the member" tells you where the reasoning actually breaks — and that is a result worth writing up.

Run `python bench/run_bench.py --compare` to produce the numbers. They are not checked into this README, because reporting a benchmark you have not run is exactly the failure mode the project is about.

---

## Quick start

```bash
pip install -e .
pytest                       # 93 validation tests
python bench/run_bench.py --dry-run
```

```python
import steelcheck as sc

shape = sc.shapes.get("W12X65")
mat   = sc.materials.preferred_for("W12X65")      # A992, per Table 2-4

# 30 ft column. Fixed top and bottom about x, no intermediate bracing.
# About y: pinned at the top, fixed at the base, braced 15 ft up.
x = sc.Axis.unbraced(30, "fixed", "fixed", name="X")
y = sc.Axis.braced(30, [15], top="pinned", bottom="fixed", name="Y")

cap = sc.compression.available_strength(shape, mat, x, y)
print(cap)          # phi_c*Pn = 662.9 kips (Y governs, inelastic buckling)
print(cap.report()) # the full calculation, step by step, with citations
```

The report is the deliverable, not the number:

```
  -- Step 2: effective lengths --
  X segment 1 = 30 ft, fixed-fixed, K=0.65 -> Lc=19.5 ft   (AISC Table C-A-7.1)
  Y segment 1 = 15 ft, pinned-fixed, K=0.8 -> Lc=12 ft     (AISC Table C-A-7.1)
  Y segment 2 = 15 ft, pinned-pinned, K=1 -> Lc=15 ft      (AISC Table C-A-7.1)
  Y Lc (governing) = 15.0 ft   [largest K*L]
  -- Step 3: critical direction --
  Lcy_eq = Lcx/(rx/ry) = 11.2 ft
  governing direction = Y   [11.2 ft <= 15 ft]
  -- Step 4: slenderness ratio --
  Lcy (in) = 180 in   [15 ft x 12]
  Lcy/r = 59.6   [r = 3.02 in]   (AISC Sect. E2)
```

Note the two segments: same 15 ft length, different effective lengths, because only one of them touches the fixed base.

---

## Validation

The engine is checked against values that do not depend on it.

Every expected number in the tension and compression tests was stated aloud by the instructor of a university steel design course while working an example, and most were cross-checked against a published AISC Manual table in the same breath. Those two independent sources are the ground truth.

```
tests/test_tension.py ............. 17 passed
tests/test_compression.py ......... 19 passed
tests/test_flexure.py ............. 26 passed
tests/test_steelpy_integration.py . 31 passed   (full AISC database)
                                    93 total
```

The flexure tests are the exception: their seed section properties (W18×35, W16×31, W21×48, and the flexure properties of W12×65) come from steelpy, not from lecture, and are tagged for verification against Table 1-1. Because the hand calculations were worked from those same properties, each flexure capacity test also asserts against the published AISC Table 3-2 value as the independent check.

A few that carry real weight:

| Case | Expected | Source |
|---|---|---|
| W12×65, pinned, 12 ft | φ꜀Pₙ = 727 k | hand calc 727, Table 4-1a 728 |
| W12×65, mixed bracing | φ꜀Pₙ = 686 k | hand calc 686, Table 4-1a 685 |
| W10×30, four connections | 398 / 318 / 312 / 289 k | lecture series |
| L6×4×½, long vs short leg | 174 k / 182 k | lecture |
| C10×30, welded | 396 k, **yielding governs** | lecture |

That last one matters: rupture usually governs a tension member, and a test suite that only contains rupture-governed cases would hide a sign error in the comparison. The C10×30 is the counterexample.

Other tests assert physical properties rather than specific numbers — capacity decreases monotonically with length, `F_n` never exceeds `F_y`, no amount of bracing beats the squash load `φF_yA_g`, and a long column's strength is identical in Grade 50 and Grade 80 because `F_y` does not appear in AISC Eq. E3-3.

---

## Architecture: the model never computes

In `tools` mode the language model gets three tools — `lookup_shape`, `column_capacity`, `tension_capacity` — and the division of labour is strict:

```
natural language  ──▶  LLM  ──▶  structured inputs
                                       │
                                       ▼
                             steelcheck  (deterministic)
                                       │
                                       ▼
                        result + full citation trace
                                       │
                        LLM ◀──────────┘   explains, never recalculates
```

The model parses and explains. The engine computes. Neither does the other's job. This is the only arrangement that produces a number an engineer can actually review, and demonstrating *why* is a large part of what the benchmark is for.

Tool errors are returned to the model rather than raised, so a wrong designation or a missing `x̄` comes back as a correctable message instead of a crash.

---

## Scope

Implemented:

- **Chapter D** — tensile yielding (Eq. D2-1) and rupture (Eq. D2-2), net area with hole allowances, all four Table D3.1 shear lag cases, LRFD and ASD.
- **Chapter E** — Table B4.1a classification gate, per-segment effective lengths from Table C-A-7.1, the equivalent-effective-length method for the critical direction, Eqs. E3-1 through E3-4.
- **Chapter F** — doubly symmetric I-shapes bent about the major axis. Table B4.1b compactness gate, yielding (Eq. F2-1), inelastic and elastic lateral-torsional buckling with Lp and Lr (Eqs. F2-2 through F2-6), the Cb moment-gradient factor (Eq. F1-1) capped at Mp, and flange local buckling for noncompact flanges (Eq. F3-1), LRFD and ASD.
- **Load combinations** — ASCE 7 basic dead and live combinations for both methods.

Not implemented: flexure beyond major-axis I-shapes with compact or noncompact flanges (slender flanges, minor-axis bending, Sections F4–F13), Chapter G (shear), block shear, connection design, slender-element members (Section E7), and anything seismic. The scope is deliberately the part that could be validated against published values rather than the part that would look most complete.

### Section property data

The seed set in `steelcheck/shapes.py` holds only the handful of shapes needed to reproduce the documented worked examples. For the full 1,660-shape database:

```bash
pip install steelpy
```

```python
sc.shapes.load_from_steelpy()      # W, C, L, WT, HSS, S, M, HP, MT, ST, PIPE
```

Two ratios the Manual prints are not stored in that dataset and are derived:

```
bf/2tf = bf / (2*tf)
h/tw   = (d - 2k) / tw
```

Both reproduce the published figures to three significant figures — W12×65 gives 9.917 and 24.87 against a published 9.92 and 24.9.

Alternatively, the **AISC Shapes Database** is a direct download if you have access (free to AISC members; [student membership](https://www.aisc.org/aisc-membership/member-types/student/) is the cheap route). Save it as `data/aisc_shapes.csv` and call `sc.shapes.load_aisc_csv(...)` instead.

No AISC file is redistributed here, and `data/*.csv` is gitignored. Section dimensions are factual data, but the Shapes Database and the *Specification* carry their own terms — **check them before redistributing anything.** Implementing the Specification's equations is ordinary practice; republishing its tables is not.

`tests/test_steelpy_integration.py` re-runs every validation case against the full database. That makes it a three-way agreement: the published AISC table value, the instructor's hand calculation, and this engine running on official section properties. It also confirmed the two seed values that had been inferred rather than transcribed (`W14X90` area 26.5 in², `WT5X15` area 4.42 in²) were correct.

---

## Related work

Several Python packages already expose AISC section properties — [`steelpy`](https://pypi.org/project/steelpy/), [`pyaisc`](https://github.com/mwhit74/pyaisc), [`steel-helper`](https://pypi.org/project/steel-helper/), among others. Most are lookup layers. This project does not try to replace them; it adds the parts they generally leave out:

- limit-state **design checks** rather than property lookup,
- a **citation trace** on every result, so the output is reviewable,
- a **benchmark and eval set** for measuring model performance on the domain.

If you only need section properties, use one of those.

---

## Not a substitute for engineering judgement

This is a study and research tool. It has not been independently verified, it implements a deliberately narrow slice of the Specification, and it must not be used for the design of anything that will be built. Structural design requires a licensed engineer.
