# LLM-GENERATED — UNVERIFIED (tier: simulation)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""
Collate per-run output into a single collated_results.npz per planet.

Reads either the LSF per-run directory tree (``--source tree``) or the one-file-per-run
output produced by run_one.py on HTCondor (``--source npz``).

Usage:
    python collate.py bin_inj                        # LSF tree under output/
    python collate.py bin_inj --source npz --root staging/

Why runs go missing, and why that now matters
---------------------------------------------
``summary.json`` is the last file run_model writes, so it is exactly what a killed or
preempted run lacks.  The previous version caught FileNotFoundError and silently
dropped such runs, leaving no record anywhere -- the row order became "sorted run dirs,
minus whatever was dropped", so the mapping from row to run identity was lost and the
survival denominator was quietly wrong.  That was tolerable while jobs ran to
completion; under preemption it is not.  Every dropped run is now recorded in
``missing_runs``/``missing_reasons`` and the exit status is non-zero if a planet came
up short of the run count in run_plan.json.
"""

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from binary_planets.plan import MODE_CODES, load_plan  # noqa: E402

LEGACY_KEYS = ("summaries", "cfgs", "elements", "distances")


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def _iter_tree(planet_dir, is_binary):
    """Yield (run_index, record | None, reason) from the LSF directory tree."""
    run_dirs = sorted((d for d in planet_dir.iterdir()
                       if d.is_dir() and d.name.isdigit()),
                      key=lambda d: int(d.name))
    for d in run_dirs:
        idx = int(d.name)
        try:
            with open(d / "summary.json") as f:
                summary = json.load(f)
            with open(d / "config.json") as f:
                cfg = json.load(f)
            element = np.load(d / "elements.npy")
            distance = np.load(d / "distances.npy") if is_binary else None
        except FileNotFoundError as exc:
            yield idx, None, f"missing {pathlib.Path(exc.filename).name}"
            continue
        except Exception as exc:
            yield idx, None, f"unreadable: {type(exc).__name__}"
            continue
        yield idx, (summary, cfg, element, distance), None


def _iter_npz(planet_dir, is_binary):
    """Yield (run_index, record | None, reason) from run_one.py result files."""
    for f in sorted(planet_dir.glob("run_*.npz"),
                    key=lambda p: int(p.stem.split("_")[-1])):
        idx = int(f.stem.split("_")[-1])
        try:
            z = np.load(f, allow_pickle=True)
            if "summary" not in z.files:
                yield idx, None, "no summary in npz"
                continue
            summary = json.loads(str(z["summary"]))
            cfg = json.loads(str(z["config"]))
            element = z["elements"]
            distance = z["distances"] if is_binary and "distances" in z.files else None
        except Exception as exc:
            yield idx, None, f"unreadable: {type(exc).__name__}"
            continue
        yield idx, (summary, cfg, element, distance), None


def collate_planet(planet_dir, *, mode, n_expected=None, source="tree",
                   plan_sha=""):
    """Collate one planet.  Returns (payload_dict, n_ok, missing_indices)."""
    is_binary = (mode == 2)
    summaries, cfgs, elements, distances = [], [], [], []
    run_index, n_rejected = [], []
    missing, reasons = [], []

    it = _iter_tree if source == "tree" else _iter_npz
    seen = set()
    for idx, rec, reason in tqdm(list(it(planet_dir, is_binary)),
                                 desc="  runs", leave=False):
        seen.add(idx)
        if rec is None:
            missing.append(idx)
            reasons.append(reason)
            continue
        summary, cfg, element, distance = rec
        summaries.append(summary)
        cfgs.append(cfg)
        elements.append(element)
        if is_binary and distance is not None:
            distances.append(distance)
        run_index.append(idx)
        n_rejected.append(int(summary.get("n_rejected", -1)))

    # Runs that never produced anything at all are invisible to the iterators above;
    # only run_plan.json knows how many there should have been.
    if n_expected is not None:
        for idx in range(n_expected):
            if idx not in seen:
                missing.append(idx)
                reasons.append("absent")

    order = np.argsort(missing) if missing else []
    payload = {
        "summaries": summaries,
        "cfgs": cfgs,
        "elements": elements,
        "distances": distances,
        # --- added; the legacy four above keep their shapes ---
        "run_index": np.asarray(run_index, dtype=np.int64),
        "n_rejected": np.asarray(n_rejected, dtype=np.int64),
        "missing_runs": np.asarray([missing[i] for i in order], dtype=np.int64),
        "missing_reasons": np.asarray([reasons[i] for i in order], dtype=object),
        "n_expected": np.int64(-1 if n_expected is None else n_expected),
        "git_hash": _git_hash(),
        "plan_sha256": plan_sha,
        "platform": "osg" if source == "npz" else "ris",
        "rebound_version": __import__("rebound").__version__,
    }
    return payload, len(summaries), payload["missing_runs"]


def needs_update(planet_dir, collated_path, source):
    if not collated_path.exists():
        return True
    mtime = collated_path.stat().st_mtime
    if source == "tree":
        return any(d.stat().st_mtime > mtime for d in planet_dir.iterdir()
                   if d.is_dir() and d.name.isdigit())
    return any(f.stat().st_mtime > mtime for f in planet_dir.glob("run_*.npz"))


def main():
    ap = argparse.ArgumentParser(description="Collate simulation results.")
    ap.add_argument("run_name", help="e.g. bin_inj")
    ap.add_argument("--mode", choices=["sin_inj", "bin_inj"], default=None,
                    help="injection mode; defaults to run_name if it is a mode name")
    ap.add_argument("--source", choices=["tree", "npz"], default="tree")
    ap.add_argument("--root", default="output",
                    help="root containing <run_name>/<planet>/")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # "bin" in run_name silently mislabels any run name containing that substring.
    mode_name = args.mode or args.run_name
    if mode_name not in MODE_CODES:
        print(f"Error: cannot infer mode from {args.run_name!r}; pass --mode.",
              file=sys.stderr)
        sys.exit(2)
    mode = MODE_CODES[mode_name]

    run_dir = pathlib.Path(args.root) / args.run_name
    if not run_dir.is_dir():
        print(f"Error: '{run_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        plan, plan_sha = load_plan(args.plan)
        n_expected = int(plan["n_runs"])
    except Exception as exc:
        print(f"warning: no run plan ({exc}); completeness not checked",
              file=sys.stderr)
        plan_sha, n_expected = "", None

    short = []
    for planet_dir in tqdm(sorted(d for d in run_dir.iterdir() if d.is_dir()),
                           desc="planets"):
        out = planet_dir / "collated_results.npz"
        if not args.force and not needs_update(planet_dir, out, args.source):
            tqdm.write(f"  skip  {planet_dir.name}")
            continue
        tqdm.write(f"collate {planet_dir.name}")
        try:
            payload, n_ok, missing = collate_planet(
                planet_dir, mode=mode, n_expected=n_expected,
                source=args.source, plan_sha=plan_sha)
        except Exception as exc:
            tqdm.write(f"  ERROR {planet_dir.name}: {exc}")
            short.append((planet_dir.name, -1, -1))
            continue
        with open(out, "wb") as fh:
            np.savez_compressed(fh, **payload)
        exp = "?" if n_expected is None else n_expected
        tqdm.write(f"   -> {n_ok}/{exp} runs"
                   + (f", {len(missing)} missing" if len(missing) else ""))
        if n_expected is not None and n_ok < n_expected:
            short.append((planet_dir.name, n_ok, n_expected))

    if short:
        print("\nIncomplete planets:", file=sys.stderr)
        for name, n_ok, exp in short:
            print(f"  {name}: {n_ok}/{exp}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
