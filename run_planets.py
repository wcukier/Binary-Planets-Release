from binary_planets.run_model import run_model
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

MODE_NAMES = {1: "sin_inj", 2: "bin_inj"}
RUN_PLAN = pathlib.Path(__file__).resolve().parent / "data" / "run_plan.json"


def load_plan_entry(mode, array_index):
    """Resolve ${LSB_JOBINDEX} through data/run_plan.json.

    The plan orders systems by descending estimated cost so the longest job starts
    first, and carries a per-system timestep.  Returns (entry, plan)."""
    plan = json.loads(RUN_PLAN.read_text())
    try:
        rows = plan["modes"][MODE_NAMES[mode]]
    except KeyError:
        raise SystemExit(f"run_plan.json has no ordering for mode {mode}")
    if not 1 <= array_index <= len(rows):
        raise SystemExit(
            f"array index {array_index} outside 1..{len(rows)} for mode {mode}"
        )
    entry = next(r for r in rows if r["array_index"] == array_index)
    return entry, plan


def _init_worker(cfg, m, p, d, t, integ):
    global cfg_name, mode, pl_num, dt, t_end, integrator
    cfg_name = cfg
    mode     = m
    pl_num   = p
    dt       = d
    t_end    = t
    integrator = integ

compact_sys = np.load("data/compact_systems_run_composite.npy", allow_pickle=True)
# compact_sys = np.load("data/TOI-178.npy", allow_pickle=True)
# compact_sys = np.load("data/Kepler-11.npy", allow_pickle=True)
n_sys = len(compact_sys)
n_runs = 1000
seq = np.random.SeedSequence().generate_state(n_runs)


# def one_run(run_num, config, mode, pl_num, debug=0):
def one_run(run_num, cfg):

    
#     with open(cfg_file) as f:
#         config = json.load(f)
#     if cfg_name:
#         config["name"] = cfg_name

#     cfg = dict(config)
    sys_num = pl_num  # np.random.randint(n_sys)
    batch_num = int(int(run_num) / n_sys) + 1

    system = compact_sys[sys_num]
#     cfg["name"] = f"{cfg['name']}/{system['name']}"
    print(
        f"Sys_Num: {sys_num} Run Number: {run_num}. Run Name: {cfg['name']}. Batch #: {batch_num}",
        file=sys.stderr,
    )

    n_secondary = len(system["a"])

    cfg["n_secondary"] = n_secondary
    cfg["t_end"] = t_end

    if mode == 1:
        cfg["n_secondary"] += 1

    # cfg["n_secondary"] = 0 #DEBUG

    name = cfg["name"]
    for i in range(1):
        cfg["name"] = name
        cfg["m_star"] = float(get_stellar_mass(system))
        print(f"m_star: {cfg['m_star']}", file=sys.stderr)
        # The axes must be drawn against the same stellar mass stored in the config: for
        # planets without published error bars they scale rigidly with it.  See task 0.8.
        semi_majors = get_semimajor(system, cfg["m_star"])
        # Gap edges as *simulated*, not as catalogued.  Drawing the injected planet
        # between the catalog edges while the bounding planets moved is what put 13.6% of
        # runs outside the gap they were supposed to be in.
        i_in, i_out = gap_indices(system)
        gap_lo, gap_hi = semi_majors[i_in], semi_majors[i_out]
        es = get_eccen(system)
        incs = get_inc(system)
        masses = get_mass(system)
        for j in range(n_secondary):
            cfg[f"secondary_{j}"] = {}
            cfg[f"secondary_{j}"]["m"] = masses[j]
            cfg[f"secondary_{j}"]["a"] = semi_majors[j]
            cfg[f"secondary_{j}"]["e"] = es[j]
            cfg[f"secondary_{j}"]["inc"] = incs[j] - np.pi / 2
            cfg[f"secondary_{j}"]["omega"] = np.random.uniform(-np.pi, np.pi)
            cfg[f"secondary_{j}"]["Omega"] = np.random.uniform(-np.pi, np.pi)

        if mode == 1:
            j = n_secondary
            cfg[f"secondary_{j}"] = {}
            # cfg[f"secondary_{j}"]["m"] = np.random.normal(
            #     np.mean(masses), np.std(masses)
            # )
            cfg[f"secondary_{j}"]["m"] = np.random.uniform(0.38, 72)

            # Draw the injected planet, redrawing if its orbit crosses any of
            # the existing planets' orbits.
            while True:
                inj_a = 10 ** np.random.uniform(
                    np.log10(gap_lo), np.log10(gap_hi)
                )
                inj_e = np.random.uniform(0, 1)
                if not any(
                    orbits_cross(inj_a, inj_e, semi_majors[k], es[k])
                    for k in range(n_secondary)
                ):
                    break
            cfg[f"secondary_{j}"]["a"] = inj_a
            cfg[f"secondary_{j}"]["e"] = inj_e
            cfg[f"secondary_{j}"]["inc"] = np.arccos(np.random.uniform(-1, 1))
            cfg[f"secondary_{j}"]["omega"] = np.random.uniform(-np.pi, np.pi)
            cfg[f"secondary_{j}"]["Omega"] = np.random.uniform(-np.pi, np.pi)

        if mode == 2:
            mass_total = np.random.uniform(0.38, 72)
            q = np.random.uniform(0, 0.5)
            cfg["binary"]["m1"] = q * mass_total
            cfg["binary"]["m2"] = (1 - q) * mass_total
            cfg["binary"]["e"] = np.random.uniform(0, 1)
            cfg["binary"]["phase"] = np.random.uniform(-np.pi, np.pi)

            cfg["binary"]["Omega"] = np.random.uniform(-np.pi, np.pi)
            cfg["binary"]["inc"] = np.arccos(np.random.uniform(-1, 1))
            cfg["binary"]["bin_inc"] = np.arccos(np.random.uniform(-1, 1))


            # Built from the *drawn* gap edges, for the same reason as the a draw below:
            # e_max bounds the COM eccentricity that keeps the pair inside the gap it is
            # actually being injected into, not the catalog one.  See task 0.8.
            emax = min(gap_hi/gap_lo - 1,
                    1- gap_lo/gap_hi
            )
            
            # Draw the injected binary, redrawing if the pair's heliocentric
            # orbit (widened by the binary separation so the whole pair is
            # accounted for) crosses any of the existing planets' orbits.
            while True:
                
                cfg["binary"]["e_sys"] = np.random.uniform(0, emax)

                bin_a = 10 ** np.random.uniform(
                    np.log10(gap_lo), np.log10(gap_hi)
                )
                r_hill = get_hill_radius(
                    bin_a, cfg["binary"]["e_sys"], mass_total, cfg["m_star"]
                )
                bin_d = r_hill * np.random.uniform(0, 1.0)
                if not any(
                    orbits_cross(
                        bin_a, cfg["binary"]["e_sys"], semi_majors[k], es[k],
                        buffer1=bin_d,
                    )
                    for k in range(n_secondary)
                ):
                    break
            cfg["binary"]["a"] = bin_a
            cfg["binary"]["d"] = bin_d

        else:
            cfg["binary"]["m1"] = 1e-20
            cfg["binary"]["m2"] = 1e-20

        cfg["dt"] = dt
        cfg["integrator"] = integrator

#         i = 1
#         try:
#             os.mkdir(f"output/{cfg['name']}")
#         except:
#             pass
#         while True:
#             try:
#                 os.mkdir(f"output/{cfg['name']}/{i}")
#                 break
#             except:
#                 i += 1
#         cfg["name"] = f"{cfg['name']}/{i}"

        print(f"system configuration: {cfg}", file=sys.stderr)
        return run_model(cfg, mode)
        

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
    

        
    early_stop = 1
    while early_stop:
        early_stop = one_run(run_num, cfg)
        if early_stop:
            print(f"{run_num}: Early stopped.  Retrying...", file=sys.stderr)
    

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

    with Pool(initializer=_init_worker,
              initargs=(cfg_name, mode, pl_num, dt, t_end, integrator)) as p:
        p.map(run_dispatcher, range(0, n_runs))
