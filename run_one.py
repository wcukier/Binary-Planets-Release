#!/usr/bin/env python3
# LLM-GENERATED — UNVERIFIED (tier: simulation)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Run exactly one simulation.  Entry point for OSG/HTCondor.

One HTCondor job = one *accepted* run.  The 5-year startup rejection stays inside the
job (it fires on roughly 70% of draws), so a job that exits 0 has produced one accepted
run, not one attempt.

Self-checkpointing: after --checkpoint-seconds of wall clock in the main integration
loop the run writes a checkpoint and exits with --checkpoint-exit-code.  HTCondor
transfers the checkpoint files, restarts this script from the top, and we resume.  Only
the ~1.2% of runs that survive ever live long enough to reach the timer.

Exit codes
----------
0   run completed; result written
3   draw rejected --max-attempts times (loud failure, not a silent drop)
85  checkpoint written, please restart (configurable)
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from binary_planets.checkpoint import pack as ckpt_pack           # noqa: E402
from binary_planets.checkpoint import unpack as ckpt_unpack       # noqa: E402
from binary_planets.checkpoint import exists as ckpt_exists       # noqa: E402
from binary_planets.draw import draw_config                      # noqa: E402
from binary_planets.plan import (                                # noqa: E402
    MODE_CODES, entry_by_name, load_plan, run_seed, safe_name,
)
from binary_planets.run_model import (                           # noqa: E402
    CHECKPOINTED, DONE, REJECTED, run_model_resumable,
)
from binary_planets.utils import DATA_DIR                        # noqa: E402


def git_hash():
    # Set by container/bp.def; the image has no .git.
    if os.environ.get("BP_GIT_HASH"):
        return os.environ["BP_GIT_HASH"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["sin_inj", "bin_inj"])
    ap.add_argument("--system", required=True,
                    help="system name as in run_plan.json, spaces optionally as '_'")
    ap.add_argument("--run", required=True, type=int,
                    help="run index within (mode, system); fixes this run's seed")
    ap.add_argument("--plan", default=None, help="path to run_plan.json")
    ap.add_argument("--outdir", default=".", help="root for the run directory")
    ap.add_argument("--ckpt-dir", default=None,
                    help="where checkpoint files live (default: --outdir)")
    ap.add_argument("--checkpoint-seconds", type=float, default=10800.0,
                    help="wall seconds before writing a checkpoint and exiting")
    ap.add_argument("--checkpoint-exit-code", type=int, default=85)
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="give up after this many startup rejections")
    ap.add_argument("--t-end", type=float, default=None, help="override plan t_end")
    ap.add_argument("--cpu-budget-hours", type=float, default=None,
                    help="total CPU budget over all executions; when reached, write a "
                         "handoff record to --result-npz and exit 0")
    ap.add_argument("--resume-from", default=None,
                    help="handoff record (a result .npz with 'handoff') to continue")
    ap.add_argument("--t-snapshot", type=float, default=None,
                    help="extended run: log the production schedule to this time, save a "
                         "snapshot there, then continue to --t-end")
    ap.add_argument("--n-log-ext", type=int, default=9000,
                    help="extended run: log steps from --t-snapshot to --t-end")
    ap.add_argument("--replay-draw", type=int, default=None,
                    help="rerun a completed run: take draw N of the seeded sequence (N = its "
                         "n_rejected) without integrating the rejected draws")
    ap.add_argument("--result-npz", default="result.npz",
                    help="collected output; '' to write only the legacy file tree")
    return ap.parse_args(argv)


def write_result(path, base, config, plan_sha, seed, mode, system, run_index):
    """Collect the run's products into one compressed file.

    One file per job rather than the five-file directory the LSF path writes: 132,000
    directories on an access point is a filesystem problem, and transfer_output_remaps
    does not reliably create intermediate directories.  Compression matters because
    `elements` is allocated at full n_log and is mostly NaN for the ~98.7% of runs that
    halt early.
    """
    b = pathlib.Path(base)
    payload = {
        "config": json.dumps(config),
        "mode": mode,
        "system": system,
        "run_index": run_index,
        "seed": seed,
        "plan_sha256": plan_sha,
        "git_hash": git_hash() or "",
        "rebound_version": __import__("rebound").__version__,
    }
    for name in ("elements", "distances"):
        f = b / f"{name}.npy"
        if f.exists():
            payload[name] = np.load(f)
    for name in ("summary", "config"):
        f = b / f"{name}.json"
        if f.exists():
            payload[name] = f.read_text()
    snap = b / "snapshot.json"
    if snap.exists():
        payload["snapshot_summary"] = snap.read_text()
    final = b / "final.sim.bin"
    if final.exists():
        payload["final_sim"] = np.frombuffer(final.read_bytes(), dtype=np.uint8)
    notes = b / "run_notes.out"
    if notes.exists():
        payload["run_notes"] = notes.read_text()
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **payload)


# LLM-GENERATED — UNVERIFIED (tier: simulation)
def write_handoff(path, ckpt_dir, meta, plan_sha, mode, system, run_index):
    """A result file carrying the checkpoint instead of results; resume it elsewhere
    with --resume-from.  collate.py counts it as incomplete."""
    payload = {"handoff": True, "mode": mode, "system": system,
               "run_index": run_index, "seed": meta["seed"],
               "cpu_accum": meta["cpu_accum"], "plan_sha256": plan_sha,
               "git_hash": git_hash() or "",
               "rebound_version": __import__("rebound").__version__}
    payload.update(ckpt_pack(ckpt_dir))
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **payload)


def main(argv=None):
    args = parse_args(argv)
    mode = MODE_CODES[args.mode]

    plan, plan_sha = load_plan(args.plan)
    entry = entry_by_name(plan, mode, args.system)
    t_end = args.t_end if args.t_end is not None else float(plan["t_end"])
    n_log = int(plan.get("n_log", 10000))
    if args.t_snapshot is not None:
        n_log += args.n_log_ext

    systems = np.load(DATA_DIR / "compact_systems_run_composite.npy", allow_pickle=True)
    system = systems[entry["system_index"]]

    system_name = entry["name"]
    name = f"{args.mode}/{safe_name(system_name)}/{args.run}"
    base = f"{args.outdir}/{name}"
    os.makedirs(base, exist_ok=True)
    ckpt_dir = args.ckpt_dir or args.outdir

    seed = run_seed(plan, mode, entry["system_index"], args.run)

    # After a self-checkpoint restart the sandbox holds a newer checkpoint than the
    # handoff; unpacking again would roll the run back.
    if args.resume_from and not ckpt_exists(ckpt_dir):
        rec = np.load(args.resume_from, allow_pickle=True)
        if "handoff" not in rec.files or int(rec["seed"]) != seed:
            raise SystemExit(f"{args.resume_from} is not a handoff record for {name}")
        ckpt_unpack(rec, ckpt_dir)

    deadline = time.monotonic() + args.checkpoint_seconds

    def should_checkpoint():
        return time.monotonic() >= deadline

    # Seed once.  Each rejected draw advances the stream rather than resetting it, so
    # the realised prior matches the LSF path exactly -- the rejection loop is part of
    # the prior, not a retry around it.
    np.random.seed(seed)

    status, attempt = REJECTED, 0
    if args.replay_draw is not None:
        # LLM-GENERATED — UNVERIFIED (tier: simulation).  Rejection integrations consume
        # no random numbers, so draw N here is the config the run accepted on attempt N.
        for _ in range(args.replay_draw + 1):
            cfg = draw_config(system, mode, dt=entry["dt"], t_end=t_end,
                              integrator=plan["integrator"], name=name, n_log=n_log)
        if args.t_snapshot is not None:
            cfg["t_snapshot"] = args.t_snapshot
            cfg["n_log_ext"] = args.n_log_ext
        attempt = args.replay_draw
        status = run_model_resumable(
            cfg, mode, outdir=args.outdir, ckpt_dir=ckpt_dir,
            should_checkpoint=should_checkpoint, attempt=attempt,
            seed=seed, git_hash=git_hash(),
            cpu_budget_s=None if args.cpu_budget_hours is None
            else args.cpu_budget_hours * 3600,
        )
        if status == REJECTED:
            print(f"replayed draw {args.replay_draw} for {name} halted inside the startup "
                  "window", file=sys.stderr)
            return 4
    while status == REJECTED and attempt < args.max_attempts:
        cfg = draw_config(system, mode, dt=entry["dt"], t_end=t_end,
                          integrator=plan["integrator"], name=name, n_log=n_log)
        if args.t_snapshot is not None:
            cfg["t_snapshot"] = args.t_snapshot
            cfg["n_log_ext"] = args.n_log_ext
        status = run_model_resumable(
            cfg, mode, outdir=args.outdir, ckpt_dir=ckpt_dir,
            should_checkpoint=should_checkpoint, attempt=attempt,
            seed=seed, git_hash=git_hash(),
            cpu_budget_s=None if args.cpu_budget_hours is None
            else args.cpu_budget_hours * 3600,
        )
        if status == REJECTED:
            attempt += 1

    if status == CHECKPOINTED and args.cpu_budget_hours is not None:
        meta = json.loads((pathlib.Path(ckpt_dir) / "checkpoint.json").read_text())
        if meta["cpu_accum"] >= args.cpu_budget_hours * 3600:
            write_handoff(args.result_npz, ckpt_dir, meta, plan_sha,
                          args.mode, system_name, args.run)
            print(f"cpu budget {args.cpu_budget_hours} h reached; handoff written",
                  file=sys.stderr)
            return 0

    if status == CHECKPOINTED:
        print(f"checkpoint written after {args.checkpoint_seconds:.0f}s; "
              f"exiting {args.checkpoint_exit_code} for restart", file=sys.stderr)
        return args.checkpoint_exit_code

    if status == REJECTED:
        print(f"draw rejected {args.max_attempts} times for {args.mode}/"
              f"{args.system}/{args.run}; giving up", file=sys.stderr)
        return 3

    if args.result_npz:
        write_result(args.result_npz, base, cfg, plan_sha, seed,
                     args.mode, system_name, args.run)
    # Not `attempt`: on resume the count comes from the checkpoint.
    n_rej = json.loads((pathlib.Path(base) / "summary.json").read_text())["n_rejected"]
    print(f"done: {name} after {n_rej} rejected draws", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
