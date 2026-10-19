"""Benchmark: can a language model do structural steel design?

Two conditions over the same problem set:

``raw``
    The model answers from its own knowledge. It has to recall section
    properties, pick the right limit state, convert units, and carry a
    multi-step calculation -- all in-context, all unverifiable.

``tools``
    The model gets tool access to steelcheck. It parses the problem into
    structured inputs and reads back a result. **It never does arithmetic.**

The gap between the two is the finding. Reporting it per trap category is what
makes it useful: an aggregate score says models are mediocre at this, a
per-trap breakdown says exactly which reasoning step breaks.

Usage::

    export ANTHROPIC_API_KEY=sk-...
    python bench/run_bench.py --mode raw   --limit 20
    python bench/run_bench.py --mode tools --limit 20
    python bench/run_bench.py --compare              # both, then the delta

    python bench/run_bench.py --dry-run              # no API calls; checks wiring
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import steelcheck as sc  # noqa: E402
from steelcheck.compression import Axis  # noqa: E402
from steelcheck.compression import available_strength as column_strength  # noqa: E402
from steelcheck.tension import Bolted, Hole, Welded  # noqa: E402
from steelcheck.tension import available_strength as tension_strength  # noqa: E402

PROBLEMS = Path(__file__).parent / "problems.json"
DEFAULT_MODEL = "claude-sonnet-4-5"

ANSWER_INSTRUCTION = """
End your reply with a fenced json block holding only the requested quantities,
using these keys where they apply: phi_c_Pn, phi_Pn, Pu, Pa (numbers, in kips),
limit_state ("tensile yielding" or "tensile rupture"), governing_axis ("x" or "y").

```json
{"phi_c_Pn": 123.4}
```
""".strip()


# ---------------------------------------------------------------------------
# The tools the model may call in `tools` mode
# ---------------------------------------------------------------------------

TOOL_SCHEMA = [
    {
        "name": "lookup_shape",
        "description": (
            "Look up AISC Table 1-1 section properties for a rolled shape. "
            "Returns gross area, slenderness ratios, radii of gyration and "
            "centroid distances. Use this instead of recalling values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "designation": {
                    "type": "string",
                    "description": "e.g. 'W12X65', 'L6X4X1/2', 'C10X30'",
                }
            },
            "required": ["designation"],
        },
    },
    {
        "name": "column_capacity",
        "description": (
            "Full AISC Chapter E compression check. Handles the section "
            "classification gate, per-segment effective lengths (a brace acts as "
            "a pin), the critical direction, and the inelastic/elastic branch. "
            "Returns the available strength and the full calculation trace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "designation": {"type": "string"},
                "grade": {
                    "type": "string",
                    "description": "e.g. 'A992', 'A572 Gr. 50', 'A913 Gr. 80'",
                    "default": "A992",
                },
                "total_length_ft": {"type": "number"},
                "x_top": {"type": "string", "enum": ["pinned", "fixed", "free"]},
                "x_bottom": {"type": "string", "enum": ["pinned", "fixed", "free"]},
                "x_braces_ft": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Brace heights above the base, x-axis. Empty if none.",
                },
                "y_top": {"type": "string", "enum": ["pinned", "fixed", "free"]},
                "y_bottom": {"type": "string", "enum": ["pinned", "fixed", "free"]},
                "y_braces_ft": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Brace heights above the base, y-axis. Empty if none.",
                },
            },
            "required": [
                "designation", "total_length_ft",
                "x_top", "x_bottom", "y_top", "y_bottom",
            ],
        },
    },
    {
        "name": "tension_capacity",
        "description": (
            "Full AISC Chapter D tension check: yielding on the gross section "
            "and rupture on the effective net section, returning whichever "
            "governs. Handles hole deductions and all four Table D3.1 shear lag "
            "cases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "designation": {"type": "string"},
                "grade": {"type": "string", "default": "A992"},
                "connection": {"type": "string", "enum": ["bolted", "welded"]},
                "db": {"type": "number", "description": "Bolt diameter, in."},
                "holes": {
                    "type": "array",
                    "description": (
                        "Holes at ONE cross section. Each entry is "
                        "[count, element thickness in inches]. Bolts along the "
                        "line do not add holes to the same cross section."
                    ),
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "n_per_line": {"type": "integer"},
                "spacing": {"type": "number", "default": 3.0},
                "xbar": {
                    "type": "number",
                    "description": (
                        "Connection eccentricity. Angle: xbar for the long leg, "
                        "ybar for the short leg."
                    ),
                },
                "all_elements_connected": {"type": "boolean", "default": False},
                "weld_longitudinal": {"type": "boolean", "default": False},
                "weld_transverse": {"type": "boolean", "default": False},
                "weld_L": {"type": "number", "description": "Longitudinal weld length, in."},
                "weld_w": {"type": "number", "description": "Distance between welds, in."},
                "connected_area": {
                    "type": "number",
                    "description": "Required for transverse-only welds (Case 3).",
                },
            },
            "required": ["designation", "connection"],
        },
    },
]


def _grade(name: str | None):
    if not name:
        return sc.materials.A992
    try:
        return sc.materials.get(name)
    except KeyError:
        for key, mat in sc.materials.MATERIALS.items():
            if key.lower().replace(" ", "").startswith(name.lower().replace(" ", "")[:6]):
                return mat
        return sc.materials.A992


def run_tool(name: str, args: dict) -> dict:
    """Execute a tool call against the engine. Errors are returned, not raised."""
    try:
        if name == "lookup_shape":
            s = sc.shapes.get(args["designation"])
            return {
                k: v
                for k, v in {
                    "designation": s.designation, "A": s.A, "d": s.d, "bf": s.bf,
                    "tf": s.tf, "tw": s.tw, "t": s.t, "bf_2tf": s.bf_2tf,
                    "h_tw": s.h_tw, "rx": s.rx, "ry": s.ry, "xbar": s.xbar,
                    "ybar": s.ybar, "Zx": s.Zx, "Sx": s.Sx,
                }.items()
                if v is not None
            }

        if name == "column_capacity":
            s = sc.shapes.get(args["designation"])
            m = _grade(args.get("grade"))
            total = args["total_length_ft"]

            def axis(top, bottom, braces, nm):
                braces = [b for b in (braces or []) if 0 < b < total]
                if braces:
                    return Axis.braced(total, braces, top=top, bottom=bottom, name=nm)
                return Axis.unbraced(total, top, bottom, name=nm)

            x = axis(args["x_top"], args["x_bottom"], args.get("x_braces_ft"), "X")
            y = axis(args["y_top"], args["y_bottom"], args.get("y_braces_ft"), "Y")
            cap = column_strength(s, m, x, y)
            return {
                "phi_c_Pn_kips": round(cap.lrfd, 1),
                "Pn_over_omega_kips": round(cap.asd, 1),
                "governing_axis": cap.governing_axis,
                "Lcx_ft": round(x.Lc, 3),
                "Lcy_ft": round(y.Lc, 3),
                "Lc_over_r": round(cap.Lc_over_r, 1),
                "buckling": cap.buckling_mode,
                "trace": cap.report(),
            }

        if name == "tension_capacity":
            s = sc.shapes.get(args["designation"])
            m = _grade(args.get("grade"))
            if args["connection"] == "bolted":
                conn = Bolted(
                    db=args.get("db", 0.75),
                    holes=[Hole(int(c), float(t)) for c, t in args.get("holes", [])],
                    n_per_line=args.get("n_per_line", 1),
                    spacing=args.get("spacing", 3.0),
                    xbar=args.get("xbar"),
                    all_elements_connected=args.get("all_elements_connected", False),
                )
            else:
                conn = Welded(
                    longitudinal=args.get("weld_longitudinal", False),
                    transverse=args.get("weld_transverse", False),
                    L=args.get("weld_L"),
                    w=args.get("weld_w"),
                    xbar=args.get("xbar"),
                    connected_area=args.get("connected_area"),
                    all_elements_connected=args.get("all_elements_connected", False),
                )
            res = tension_strength(s, m, conn)
            return {
                "phi_Pn_kips": round(res.lrfd, 1),
                "Pn_over_omega_kips": round(res.asd, 1),
                "limit_state": res.limit_state,
                "yielding_kips": round(res.yielding.lrfd, 1),
                "rupture_kips": round(res.rupture.lrfd, 1),
                "trace": res.report(),
            }

        return {"error": f"unknown tool {name}"}
    except Exception as exc:  # surfaced to the model so it can correct itself
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_answer(text: str) -> dict:
    blocks = _JSON_BLOCK.findall(text or "")
    for blk in reversed(blocks):
        try:
            return json.loads(blk)
        except json.JSONDecodeError:
            continue
    # Fall back to a bare trailing object.
    m = re.search(r"\{[^{}]*\}\s*$", text or "")
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def score(expected: dict, got: dict, tol: float) -> tuple[bool, str]:
    if not got:
        return False, "no parseable answer"
    for key, want in expected.items():
        have = got.get(key)
        if have is None:
            return False, f"missing {key}"
        if isinstance(want, (int, float)):
            try:
                have_f = float(have)
            except (TypeError, ValueError):
                return False, f"{key}: {have!r} not numeric"
            if want == 0:
                ok = abs(have_f) < 1e-9
            else:
                ok = abs(have_f - want) / abs(want) <= tol
            if not ok:
                return False, f"{key}: got {have_f:g}, want {want:g}"
        else:
            if str(have).strip().lower() != str(want).strip().lower():
                return False, f"{key}: got {have!r}, want {want!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def solve(client, model: str, problem: dict, mode: str, max_turns: int = 8) -> str:
    system = (
        "You are checking structural steel members to AISC 360. Work carefully "
        "and show your reasoning.\n\n" + ANSWER_INSTRUCTION
    )
    if mode == "tools":
        system = (
            "You are checking structural steel members to AISC 360. You have "
            "tools that perform the calculations exactly. Use them rather than "
            "computing or recalling values yourself: look up section properties, "
            "then call the capacity tool with the geometry as stated in the "
            "problem. Report the number the tool returns.\n\n" + ANSWER_INSTRUCTION
        )

    messages = [{"role": "user", "content": problem["prompt"]}]
    tools = TOOL_SCHEMA if mode == "tools" else []

    for _ in range(max_turns):
        kwargs = dict(model=model, max_tokens=2000, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.create(**kwargs)

        messages.append({"role": "assistant", "content": resp.content})
        calls = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not calls:
            return "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )

        results = []
        for call in calls:
            out = run_tool(call.name, call.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(out)[:6000],
            })
        messages.append({"role": "user", "content": results})

    return "".join(
        b.text for b in messages[-1].get("content", []) if getattr(b, "type", None) == "text"
    )


def run(mode: str, model: str, limit: int | None, verbose: bool) -> dict:
    data = json.loads(PROBLEMS.read_text())
    problems = data["problems"][:limit] if limit else data["problems"]

    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY to run the benchmark (or use --dry-run).")
    client = Anthropic()

    rows = []
    for i, p in enumerate(problems, 1):
        try:
            reply = solve(client, model, p, mode)
            got = extract_answer(reply)
            ok, why = score(p["answer"], got, p["tolerance"])
        except Exception as exc:
            ok, why, got = False, f"error: {type(exc).__name__}: {exc}", {}
        rows.append({**p, "passed": ok, "why": why, "got": got})
        mark = "PASS" if ok else "FAIL"
        print(f"[{i:>3}/{len(problems)}] {mark}  {p['id']:<16} {why}", flush=True)
        if verbose and not ok:
            print(f"        expected {p['answer']}")

    return summarize(rows, mode, model)


def summarize(rows: list[dict], mode: str, model: str) -> dict:
    n = len(rows)
    passed = sum(r["passed"] for r in rows)

    by_trap: dict[str, list[bool]] = defaultdict(list)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["passed"])
        for t in r["traps"]:
            by_trap[t].append(r["passed"])

    print(f"\n{'=' * 60}")
    print(f"{mode.upper()}  model={model}")
    print(f"overall: {passed}/{n} = {100 * passed / n:.0f}%")
    print(f"\n{'by category':<30} {'acc':>6}  n")
    for k, v in sorted(by_cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {k:<28} {100 * sum(v) / len(v):>5.0f}%  {len(v)}")
    print(f"\n{'by trap (worst first)':<30} {'acc':>6}  n")
    for k, v in sorted(by_trap.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {k:<28} {100 * sum(v) / len(v):>5.0f}%  {len(v)}")

    return {
        "mode": mode,
        "model": model,
        "n": n,
        "passed": passed,
        "accuracy": passed / n if n else 0.0,
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
        "by_trap": {k: sum(v) / len(v) for k, v in by_trap.items()},
        "rows": rows,
    }


def dry_run() -> None:
    """Exercise the tools and scorer without touching the API."""
    data = json.loads(PROBLEMS.read_text())
    print(f"loaded {data['n']} problems, {len(data['trap_counts'])} trap categories\n")

    print("tool: lookup_shape(W12X65)")
    print(" ", run_tool("lookup_shape", {"designation": "W12X65"}), "\n")

    print("tool: column_capacity -- 30 ft, X fixed-fixed, Y braced at 15 ft")
    out = run_tool("column_capacity", {
        "designation": "W12X65", "grade": "A992", "total_length_ft": 30,
        "x_top": "fixed", "x_bottom": "fixed", "x_braces_ft": [],
        "y_top": "pinned", "y_bottom": "fixed", "y_braces_ft": [15],
    })
    print("  phi_c_Pn =", out["phi_c_Pn_kips"], "kips |",
          "Lcx =", out["Lcx_ft"], "ft |", "Lcy =", out["Lcy_ft"], "ft |",
          "governs:", out["governing_axis"], "\n")

    print("tool: tension_capacity -- L6X4X1/2, long leg, 5 bolts")
    out = run_tool("tension_capacity", {
        "designation": "L6X4X1/2", "grade": "A572 Gr. 50", "connection": "bolted",
        "db": 0.75, "holes": [[2, 0.5]], "n_per_line": 5, "spacing": 3.0,
        "xbar": 0.981,
    })
    print("  phi_Pn =", out["phi_Pn_kips"], "kips, governed by", out["limit_state"], "\n")

    print("tool error handling (bad shape):")
    print(" ", run_tool("lookup_shape", {"designation": "W99X999"})["error"][:80], "\n")

    print("scorer:")
    for reply, label in [
        ('Working... ```json\n{"phi_c_Pn": 663.0}\n```', "correct"),
        ('```json\n{"phi_c_Pn": 745.0}\n```', "10% high"),
        ("I think it is about 663 kips.", "no json block"),
    ]:
        got = extract_answer(reply)
        ok, why = score({"phi_c_Pn": 662.9}, got, 0.02)
        print(f"  {label:<14} -> {'PASS' if ok else 'FAIL'}  ({why})")

    print("\nwiring OK. Set ANTHROPIC_API_KEY and drop --dry-run to benchmark.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["raw", "tools"], default="raw")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compare", action="store_true", help="run both and show the delta")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    if args.compare:
        raw = run("raw", args.model, args.limit, args.verbose)
        tools = run("tools", args.model, args.limit, args.verbose)
        d = tools["accuracy"] - raw["accuracy"]
        print(f"\n{'=' * 60}")
        print(f"raw   {raw['accuracy']:.0%}")
        print(f"tools {tools['accuracy']:.0%}")
        print(f"delta {d:+.0%}")
        result = {"raw": raw, "tools": tools, "delta": d}
    else:
        result = run(args.mode, args.model, args.limit, args.verbose)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
