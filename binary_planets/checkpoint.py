# LLM-GENERATED — UNVERIFIED (tier: simulation)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Serialize and restore a mid-flight simulation so a preempted job can resume.

Format is three files, written together and committed by the last one:

    checkpoint.sim.bin   rebound binary snapshot (~4 KB)
    checkpoint.npz       the log arrays + cursor (savez_compressed)
    checkpoint.json      scalars, config and provenance -- THE COMMIT MARKER

``checkpoint.json`` is written last and via os.replace, so a job killed mid-write
leaves either a complete checkpoint or one that ``load`` refuses.  On refusal the
caller restarts the run from scratch rather than resuming into a corrupt state --
a wrong answer is far worse than a repeated one.

Two things here are easy to get wrong and expensive to get wrong:

* **E0/L0 must be carried.**  run_model captures them *before* integration; recomputing
  them after a restart would silently make E_Err and L_err meaningless (they would be
  measured against the resumed state rather than the initial one).
* **The loop phase and cursor must be explicit.**  Neither is recoverable from the log
  arrays -- the startup loop resets the cursor every iteration, and get_log_len returns
  capacity rather than fill.

``states`` is deliberately not carried: it is never written to disk and never read by
collate.py, so restoring it would cost ~3.9 MB per checkpoint for nothing.
"""

import json
import os
import pathlib

import numpy as np
import rebound

SCHEMA_VERSION = 1

SIM_NAME = "checkpoint.sim.bin"
NPZ_NAME = "checkpoint.npz"
JSON_NAME = "checkpoint.json"


class CheckpointMismatch(Exception):
    """Checkpoint is unusable -- caller must restart the run from scratch."""


def _paths(outdir):
    d = pathlib.Path(outdir)
    return d / SIM_NAME, d / NPZ_NAME, d / JSON_NAME


def exists(outdir):
    """True only if the commit marker is present."""
    return _paths(outdir)[2].exists()


def save(outdir, sim, log, *, phase, i_next, config, mode, energy0, angmom0,
         t_wall_accum, attempt, seed, git_hash=None, cpu_accum=0.0, extra=None):
    """Write a checkpoint.  Commit marker (`checkpoint.json`) is written last."""
    sim_p, npz_p, json_p = _paths(outdir)

    for p in (sim_p, npz_p, json_p):          # never leave a stale marker
        if p.exists():
            p.unlink()

    sim.save_to_file(str(sim_p))

    # Write through a handle: np.savez_compressed appends ".npz" to a *path* that does
    # not already end in it, which would silently defeat the atomic rename.
    tmp_npz = npz_p.with_name(npz_p.name + ".tmp")
    with open(tmp_npz, "wb") as fh:
        np.savez_compressed(fh, elements=log[0], distances=log[1],
                            t_step=np.int64(log[3]))
    os.replace(tmp_npz, npz_p)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "i_next": int(i_next),
        "mode": int(mode),
        "config": config,
        "energy0": float(energy0),
        "angmom0": [float(x) for x in angmom0],
        "t_wall_accum": float(t_wall_accum),
        "cpu_accum": float(cpu_accum),
        "extra": extra or {},
        "attempt": int(attempt),
        "seed": int(seed),
        "rebound_version": rebound.__version__,
        "git_hash": git_hash,
    }
    tmp_json = json_p.with_name(json_p.name + ".tmp")
    tmp_json.write_text(json.dumps(meta))
    os.replace(tmp_json, json_p)               # <- commit
    return meta


def load(outdir, *, expect_mode=None):
    """Restore a checkpoint.

    Returns ``(sim, log, meta)``.  Raises CheckpointMismatch if anything is missing,
    unreadable, or was written by a different rebound version or schema -- the caller
    must then start the run over.
    """
    sim_p, npz_p, json_p = _paths(outdir)
    if not json_p.exists():
        raise CheckpointMismatch("no commit marker")
    if not (sim_p.exists() and npz_p.exists()):
        raise CheckpointMismatch("commit marker present but payload missing")

    try:
        meta = json.loads(json_p.read_text())
    except Exception as exc:
        raise CheckpointMismatch(f"unreadable metadata: {exc}")

    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointMismatch(
            f"schema {meta.get('schema_version')} != {SCHEMA_VERSION}")
    if meta.get("rebound_version") != rebound.__version__:
        # Resuming a trace state across rebound versions is not something we have
        # verified, and a silent trajectory change is exactly what must not happen.
        raise CheckpointMismatch(
            f"rebound {meta.get('rebound_version')} != {rebound.__version__}")
    if expect_mode is not None and int(meta["mode"]) != int(expect_mode):
        raise CheckpointMismatch("mode mismatch")

    try:
        sim = rebound.Simulation(str(sim_p))
        if sim.collision != "none":
            sim.collision_resolve = "halt"
        z = np.load(npz_p)
        elements, distances, t_step = z["elements"], z["distances"], int(z["t_step"])
    except Exception as exc:
        raise CheckpointMismatch(f"unreadable payload: {exc}")

    # states is not carried -- reallocate empty; nothing reads it.
    states = np.zeros((elements.shape[0], elements.shape[1], 7)) * np.nan
    log = [elements, distances, states, t_step]
    return sim, log, meta


def pack(outdir):
    """The checkpoint's three files as a dict of arrays/strings, for a handoff record."""
    sim_p, npz_p, json_p = _paths(outdir)
    return {"ckpt_sim": np.frombuffer(sim_p.read_bytes(), dtype=np.uint8),
            "ckpt_npz": np.frombuffer(npz_p.read_bytes(), dtype=np.uint8),
            "ckpt_json": json_p.read_text()}


def unpack(record, outdir):
    """Inverse of pack: write the checkpoint files, commit marker last."""
    sim_p, npz_p, json_p = _paths(outdir)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    sim_p.write_bytes(np.asarray(record["ckpt_sim"], dtype=np.uint8).tobytes())
    npz_p.write_bytes(np.asarray(record["ckpt_npz"], dtype=np.uint8).tobytes())
    json_p.write_text(str(record["ckpt_json"]))


def clear(outdir):
    """Remove checkpoint files after a run completes normally."""
    for p in _paths(outdir):
        if p.exists():
            p.unlink()
