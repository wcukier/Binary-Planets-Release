from binary_planets.run_model import run_model
from binary_planets.draw import draw_config
from binary_planets.plan import MODE_NAMES, entry_by_index, load_plan
from binary_planets.sim import get_hill_radius, orbits_cross
from binary_planets.utils import *  # noqa

from os import sys
import json
import pathlib
import time
import os
import numpy as np
from multiprocessing import Pool


global cfg_name
global mode
global pl_num
global dt
global t_end
global integrator

def load_plan_entry(mode, array_index):
    """The plan orders systems by descending estimated cost so the longest job starts
    first, and carries a per-system timestep.  Returns (entry, plan).

    Thin wrapper over binary_planets.plan so this driver and run_one.py resolve run
    identity through one implementation.
    """
    plan, _sha = load_plan()
    return entry_by_index(plan, mode, array_index), plan


def _init_worker(cfg, m, p, d, t, integ):
    global cfg_name, mode, pl_num, dt, t_end, integrator
    cfg_name = cfg
    mode     = m
    pl_num   = p
    dt       = d
    t_end    = t
    integrator = integ

compact_sys = np.load(DATA_DIR / "compact_systems_run_composite.npy", allow_pickle=True)
# compact_sys = np.load("data/TOI-178.npy", allow_pickle=True)
# compact_sys = np.load("data/Kepler-11.npy", allow_pickle=True)
n_sys = len(compact_sys)
n_runs = 1000
MAX_DRAW_ATTEMPTS = 200
seq = np.random.SeedSequence().generate_state(n_runs)


# def one_run(run_num, config, mode, pl_num, debug=0):
def one_run(run_num, cfg):
    """Draw a config and run it.  Returns 1 if the draw was rejected in the 5-yr window.
    """
    sys_num = pl_num
    batch_num = int(int(run_num) / n_sys) + 1
    system = compact_sys[sys_num]
    print(
        f"Sys_Num: {sys_num} Run Number: {run_num}. Run Name: {cfg['name']}. Batch #: {batch_num}",
        file=sys.stderr,
    )

    config = draw_config(
        system, mode,
        dt=dt, t_end=t_end, integrator=integrator,
        name=cfg["name"], n_log=cfg["n_log"],
    )

    print(f"system configuration: {config}", file=sys.stderr)
    return run_model(config, mode)
        

def run_dispatcher(run_num):
    seed = seq[run_num]
    print(f"Seed: {seed}", flush=True)
    np.random.seed(seed)

    cfg = {
        "name": cfg_name,
        "binary": {},
        "n_log": 10000,
    }

    sys_num = pl_num  # np.random.randint(n_sys)
    batch_num = int(int(run_num) / n_sys) + 1

    system = compact_sys[sys_num]
    cfg["name"] = f"{cfg['name']}/{system['name']}"
    
    i = 0
    try:
        os.mkdir(f"output/{cfg['name']}")
    except:
        pass
    while True:
        try:
            os.mkdir(f"output/{cfg['name']}/{i}")
            break
        except Exception as e:
            i += 1
            if i > 40000:
                raise(e)
    cfg["name"] = f"{cfg['name']}/{i}"
    

        
    # Bounded, matching run_one.py's --max-attempts.  ~70% of draws are rejected in the
    # 5-yr startup window so this normally runs 1-3 times
    for attempt in range(MAX_DRAW_ATTEMPTS):
        early_stop = one_run(run_num, cfg)
        if not early_stop:
            break
        print(f"{run_num}: Early stopped.  Retrying...", file=sys.stderr)
    else:
        print(f"{run_num}: draw rejected {MAX_DRAW_ATTEMPTS} times; giving up",
              file=sys.stderr)
    

if __name__ == "__main__":
    try:
        cfg_name = sys.argv[1]
        mode = int(sys.argv[2])  # 0 for no inj, 1 for single inj, 2 for binary inj
        index = int(sys.argv[3])  # bsub doesn't allow for an index of 0
        dt_arg = sys.argv[4] if len(sys.argv) > 4 else None

        if dt_arg == "auto":
            # Plan-driven: the index is a position in the cost-ordered job list,
            # NOT a direct index into compact_systems_run_composite.npy, and dt and
            # the integrator come from the plan rather than the command line.
            entry, plan = load_plan_entry(mode, index)
            pl_num     = entry["system_index"]
            dt         = entry["dt"]
            integrator = plan["integrator"]
            t_end      = float(sys.argv[5]) if len(sys.argv) > 5 else float(plan["t_end"])
            print(
                f"[plan] array {index} -> {entry['name']} (system {pl_num}), "
                f"dt={dt:.3e}, integrator={integrator}, t_end={t_end:.3e}",
                file=sys.stderr,
            )
        else:
            pl_num     = index - 1
            dt         = float(dt_arg) if dt_arg is not None else 1e-5
            t_end      = float(sys.argv[5]) if len(sys.argv) > 5 else 1_000_000
            integrator = "leapfrog"

        print(
            f"Simulating {compact_sys[pl_num]['name']}, mode = {mode}", file=sys.stderr
        )

    except Exception as e:
        print("Error parsing arguments")
        print(f"Error was {e}")
        raise

    os.makedirs(f"output/{cfg_name}", exist_ok=True)

    # for i in range(1000):
    #     one_run(i)


    n_proc = int(os.environ.get("LSB_DJOB_NUMPROC", 0)) or os.cpu_count()

    with Pool(processes=n_proc,
              initializer=_init_worker,
              initargs=(cfg_name, mode, pl_num, dt, t_end, integrator)) as p:
        for _ in p.imap_unordered(run_dispatcher, range(0, n_runs), chunksize=1):
            pass
