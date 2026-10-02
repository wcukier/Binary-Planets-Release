# LLM-GENERATED — UNVERIFIED (tier: simulation)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Resolve run identity against data/run_plan.json.

The plan is the single source of truth for the physics knobs that vary per system --
timestep, integrator, integration time, log cadence and run count -- so that neither
driver hardcodes them.  ``run_planets.py`` (LSF) resolves by ``array_index`` because
LSF hands it ``${LSB_JOBINDEX}``; ``run_one.py`` (HTCondor) resolves by system *name*,
which is stable across regenerations of the plan whereas array_index is not.
"""

import hashlib
import json
import os
import pathlib

import numpy as np

MODE_NAMES = {0: "no_inj", 1: "sin_inj", 2: "bin_inj"}
MODE_CODES = {v: k for k, v in MODE_NAMES.items()}

DEFAULT_PLAN = pathlib.Path(__file__).resolve().parent.parent / "data" / "run_plan.json"

# Fallbacks for plans written before these fields existed.  n_log was hardcoded at
# run_planets.py:218 and the entropy did not exist at all (seeds came from a fresh
# SeedSequence at import, which is why old runs are not reproducible).
DEFAULT_N_LOG = 10000
DEFAULT_ENTROPY = 0x9E3779B97F4A7C15


def load_plan(path=None):
    """Read the plan and return (plan_dict, sha256_hex)."""
    path = pathlib.Path(path or os.environ.get("BP_RUN_PLAN", DEFAULT_PLAN))
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def safe_name(name):
    """System name with spaces replaced, for argv and paths (e.g. HD_108236)."""
    return name.replace(" ", "_")


def entry_by_name(plan, mode, name):
    """Look up a system's plan entry by name.  Preferred for HTCondor.

    Accepts either the catalogue name or its safe_name() form."""
    mode_name = MODE_NAMES[mode] if isinstance(mode, int) else mode
    try:
        rows = plan["modes"][mode_name]
    except KeyError:
        raise SystemExit(f"run_plan.json has no ordering for mode {mode!r}")
    for row in rows:
        if safe_name(row["name"]) == safe_name(name):
            return row
    raise SystemExit(f"system {name!r} not in run_plan.json for mode {mode_name}")


def entry_by_index(plan, mode, array_index):
    """Look up by 1-based array_index.  Used by the LSF driver."""
    mode_name = MODE_NAMES[mode] if isinstance(mode, int) else mode
    rows = plan["modes"][mode_name]
    if not 1 <= array_index <= len(rows):
        raise SystemExit(
            f"array index {array_index} outside 1..{len(rows)} for mode {mode_name}"
        )
    return next(r for r in rows if r["array_index"] == array_index)


def run_seed(plan, mode, system_index, run_index):
    """Derive this run's seed from its identity rather than from wall-clock entropy.

    ``spawn_key`` makes the seed a pure function of (mode, system, run), which
    guarantees no two of the 132,000 independent jobs collide -- something a fresh
    ``SeedSequence()`` per job does not -- and makes any single run re-executable
    from its config alone.
    """
    mode_code = MODE_CODES[mode] if isinstance(mode, str) else mode
    entropy = int(plan.get("seed_entropy", DEFAULT_ENTROPY))
    ss = np.random.SeedSequence(
        entropy=entropy, spawn_key=(mode_code, int(system_index), int(run_index))
    )
    return int(ss.generate_state(1)[0])
