"""Look up run settings in data/run_plan.json.

The plan holds the timestep, integrator, integration time, log cadence and run count
for each system, so neither driver hardcodes them. ``run_planets.py`` (LSF) looks
systems up by ``array_index``, which LSF passes as ``${LSB_JOBINDEX}``. ``run_one.py``
(HTCondor) looks them up by name, because array_index changes when the plan is
regenerated.
"""

import hashlib
import json
import os
import pathlib

import numpy as np

MODE_NAMES = {0: "no_inj", 1: "sin_inj", 2: "bin_inj"}
MODE_CODES = {v: k for k, v in MODE_NAMES.items()}

DEFAULT_PLAN = pathlib.Path(__file__).resolve().parent.parent / "data" / "run_plan.json"

# Used when the plan does not set n_log or seed_entropy.
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
    """Look up a system's plan entry by name. Used by the HTCondor driver.

    Accepts either the catalogue name or its safe_name() form.
    """
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
    """Look up by 1-based array_index. Used by the LSF driver."""
    mode_name = MODE_NAMES[mode] if isinstance(mode, int) else mode
    rows = plan["modes"][mode_name]

    if not 1 <= array_index <= len(rows):
        raise SystemExit(
            f"array index {array_index} outside 1..{len(rows)} for mode {mode_name}"
        )

    return next(r for r in rows if r["array_index"] == array_index)


def run_seed(plan, mode, system_index, run_index):
    """Seed for one run, computed from (mode, system, run).

    The same run always gets the same seed, so any run can be re-executed from its
    config. The seed is a single 32-bit word, so a few of the 132,000 runs share one.
    """
    mode_code = MODE_CODES[mode] if isinstance(mode, str) else mode
    entropy = int(plan.get("seed_entropy", DEFAULT_ENTROPY))

    ss = np.random.SeedSequence(
        entropy=entropy, spawn_key=(mode_code, int(system_index), int(run_index))
    )
    return int(ss.generate_state(1)[0])
