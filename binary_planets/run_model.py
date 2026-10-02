# binary_planets.anim is not used here -- both of its functions (animate_separation,
# animate_simulation) are only referenced from commented-out lines below.  Importing
# it pulls matplotlib and imageio into every job, which on OSG means shipping ~100 MB
# of unused dependencies to 132,000 jobs.  Import it explicitly if you re-enable the
# animation calls.
from binary_planets.constants import *
from binary_planets.log import *
from binary_planets.sim import *

import sys
import json
import time
import os
import datetime





# LLM-GENERATED — UNVERIFIED (tier: simulation): build_sim, summarize and
# run_model_resumable, plus run_model's `outdir` parameter.  See POLICY.md §2.
def build_sim(config, mode, run_notes=None):
    """Construct the rebound Simulation and empty log from a config.

    Deterministic and free of RNG: given the same config this produces the same initial
    state every time, which is what makes a fresh start and a checkpoint resume
    interchangeable.  Split out of run_model so run_model_resumable (used by run_one.py
    on HTCondor) shares the setup rather than reimplementing it.
    """
    def note(msg):
        if run_notes is not None:
            run_notes.write(msg)

    binary = config["binary"]
    n_secondary = config["n_secondary"]

    n_particles = n_secondary + 1
    if mode == 2:
        n_particles += 2

    sim, log = init_sim(config["m_star"],
                        config["n_log"],
                        config["integrator"],
                        config["dt"],
                        n_particles)

    if mode == 2:
        sim = init_binary_planet(sim,
                                 config["m_star"],
                                 binary["m1"],
                                 binary["m2"],
                                 binary["d"],
                                 binary["a"],
                                 binary["e"],
                                 binary["e_sys"],
                                 binary["phase"],
                                 binary["Omega"],
                                 binary["inc"],
                                 binary["bin_inc"],
                                 run_notes=run_notes)

    for i in range(int(n_secondary)):
        sec = config[f"secondary_{i}"]
        sim = init_single_planet(sim,
                                 sec["m"],
                                 sec["a"],
                                 sec["e"],
                                 sec["inc"],
                                 sec["omega"],
                                 sec["Omega"])
        note(f"Added secondary_{i}.\n")
        note(f"# of particles: sim")
    note("*************************************************\n\n")
    if "r_star" in config:
        set_radii(sim, config, mode)
    # LLM-GENERATED — UNVERIFIED (tier: simulation).  Stored in the config so a resumed
    # run uses the t = 0 value; see log_elements.
    if mode == 2:
        binary["a_com0"] = com_orbit(sim).a
    return sim, log


# LLM-GENERATED — UNVERIFIED (tier: simulation): pair_state (task 1.4d).
def pair_state(sim):
    """State of the injected pair (particles 1, 2): is it bound, and has it left?

    pair_E2         specific two-body energy, ignoring the star
    pair_bound      pair_E2 < 0
    pair_r_hill     separation / Hill radius at the pair's current distance from the
                    star; > 1 means pair_bound is not meaningful
    pair_com_a      semi-major axis of the pair's centre of mass about the star
    pair_com_bound  centre of mass bound to the star
    """
    p0, p1, p2 = sim.particles[0], sim.particles[1], sim.particles[2]
    m12 = p1.m + p2.m
    dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
    dvx, dvy, dvz = p1.vx - p2.vx, p1.vy - p2.vy, p1.vz - p2.vz
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    E2 = 0.5 * (dvx**2 + dvy**2 + dvz**2) - sim.G * m12 / r

    cx = (p1.m * p1.x + p2.m * p2.x) / m12 - p0.x
    cy = (p1.m * p1.y + p2.m * p2.y) / m12 - p0.y
    cz = (p1.m * p1.z + p2.m * p2.z) / m12 - p0.z
    cvx = (p1.m * p1.vx + p2.m * p2.vx) / m12 - p0.vx
    cvy = (p1.m * p1.vy + p2.m * p2.vy) / m12 - p0.vy
    cvz = (p1.m * p1.vz + p2.m * p2.vz) / m12 - p0.vz
    R = np.sqrt(cx**2 + cy**2 + cz**2)
    r_hill = R * (m12 / (3 * p0.m))**(1 / 3)
    E_com = 0.5 * (cvx**2 + cvy**2 + cvz**2) - sim.G * (p0.m + m12) / R
    com_a = -sim.G * (p0.m + m12) / (2 * E_com)

    return {"pair_E2": float(E2), "pair_bound": bool(E2 < 0),
            "pair_r_hill": float(r / r_hill),
            "pair_com_a": float(com_a), "pair_com_bound": bool(E_com < 0)}


# LLM-GENERATED — UNVERIFIED (tier: simulation).
def set_radii(sim, config, mode):
    """Give every body its physical radius and halt the integration on contact.

    Particle order matches build_sim: star, the pair (mode 2), then the secondaries.
    Configs without radii (drawn before collisions existed) run without detection.
    """
    r_e = RADIUS_EARTH / AU_TO_M
    ps = sim.particles
    ps[0].r = config["r_star"] * RADIUS_SUN / AU_TO_M
    k = 1
    if mode == 2:
        ps[1].r = config["binary"]["radius1"] * r_e
        ps[2].r = config["binary"]["radius2"] * r_e
        k = 3
    for i in range(int(config["n_secondary"])):
        ps[k + i].r = config[f"secondary_{i}"]["radius"] * r_e
    for p in ps:
        p.last_collision = -1.0
    sim.collision = "direct"
    sim.collision_resolve = "halt"


def summarize(sim, halt, t_end, t_elapsed, energy0, angmom0, mode=None):
    """Build the summary dict.  energy0/angmom0 must be the values captured BEFORE
    integration -- after a checkpoint resume they come from the checkpoint, not from
    the resumed simulation, or E_Err and L_err become meaningless.

    Mode 2 adds pair_state() at the final time."""
    E0 = energy0
    Lx0, Ly0, Lz0 = angmom0
    E1 = sim.energy()
    Lx1, Ly1, Lz1 = sim.angular_momentum()

    E_err = (E0 - E1) / E0
    # Task 2.0 fix: the denominator squared Lx0 twice and omitted Lz0.
    L_err = np.sqrt(((Lx0 - Lx1)**2 + (Ly0 - Ly1)**2 + (Lz0 - Lz1)**2)
                    / (Lx0**2 + Ly0**2 + Lz0**2 + 1e-20))

    try:
        o = sim.particles[2].orbit(primary=sim.particles[1])
    except Exception as e:
        print(e)
        o = sim.particles[2].orbit(primary=sim.particles[0])

    out = {
        "t_elapsed": t_elapsed,
        "E_Err": E_err,
        "E_0": E0,
        "E_1": E1,
        "L_err": L_err,
        "L_i": [Lx0, Ly0, Lz0],
        "L_f": [Lx1, Ly1, Lz1],
        "Final_elements": [o.a, o.e, o.inc, o.Omega, o.omega],
        "t_end": t_end,
        "halt": halt,
    }
    if mode == 2:
        out.update(pair_state(sim))
    return out


def run_model(config, mode, debug=0, outdir="output"):
    # `outdir` is the root the run's directory hangs off.  On LSF this stays "output"
    # (relative to the repo, which is the CWD there); on an HTCondor execute node the
    # CWD is the job scratch dir, so run_one.py passes outdir=".".
    base = f"{outdir}/{config['name']}"
    with open(f"{base}/run_notes.out", 'w+') as run_notes:
        run_notes.write(f"{sys.argv}\n")
        run_notes.write(f"Run start time: {datetime.datetime.now()}\n")
        for key in config.keys():
            run_notes.write(f"{key}: {config[key]}\n")
        run_notes.write("*************************************************\n\n")

        sim, log = build_sim(config, mode, run_notes)


    with open(f"{base}/config.json", "w+") as cf:
        json.dump(config, cf)

    E0 = sim.energy()
    Lx0, Ly0, Lz0, = sim.angular_momentum()

    if debug == 1:
        return sim

    t0 = time.time()
    _, _, halt, t_end = simulate(sim, log, config["t_end"], mode,
                                 a_com0=config["binary"]["a_com0"] if mode == 2 else None)
    if t_end < 5: return 1
    t1 = time.time()

    save_log(log, base)

    sum_data = summarize(sim, halt, t_end, t1 - t0, E0, (Lx0, Ly0, Lz0), mode=mode)
    E_err, L_err = sum_data["E_Err"], sum_data["L_err"]
    E1 = sum_data["E_1"]
    Lx1, Ly1, Lz1 = sum_data["L_f"]

    with open(f"{base}/summary.json", "w+") as cf:
        json.dump(sum_data, cf)
    
    
    

    with open(f"{base}/run_notes.out", 'a+') as summary:
        summary.write(f"{halt}\n")
        summary.write(f"t_end: {t_end}")
        summary.write(f"Time elapsed: {t1-t0}\n")

        summary.write(f"Energy Error: {E_err:.2e}\tE_i: {E0:.2e}\
            \tE_f: {E1:.2e}\n")

        summary.write(f"Angular Mom Err: {L_err:.2e}\t\
                      L_i: ({Lx0:.2e}, {Ly0:.2e}, {Lz0:.2e})\t\
                          L_f: ({Lx1:.2e}, {Ly1:.2e}, {Lz1:.2e})\n")
        
        summary.write(f"Simulation Time: {sim.t}")

    #     summary.write("\nStatistical Moments:\n")
    #     summary.write(f"{moments}\n")
    #     summary.write("\n====Derivatives====\n")
    #     summary.write(f"Initial first derivative: {first_i}\n")
    #     summary.write(f"Final first derivative: {first_f}\n")
    #     summary.write(f"Initial second derivative: {second_i}\n")
    #     summary.write(f"Final second derivative: {second_f}\n")


    #     o = sim.particles[2].orbit(primary=sim.particles[1])
    #     summary.write(f"Binary system orbital elements: a:{o.a}, e:{o.e}\n")
    
    del sim, log
    return 0


    # plot_corner(log, f"output/{config['name']}/corner.png")





if __name__ == "__main__":
    try:
        print(f"Config file: {sys.argv[1]}", file=sys.stderr)
        with open(sys.argv[1]) as f:
            config = json.load(f)
        if len(sys.argv) > 2:
            config["name"] = sys.argv[2]
        print(f"Run Name: {config['name']}", file=sys.stderr)

    except Exception as e:
        print("Error in loading config file")
        print(f"Error was {e}")
        raise()

    if len(sys.argv) > 4:
        key = sys.argv[3]
        val = sys.argv[4]
        # print(f"{key}: {val}", file=sys.stderr)
        # print(f"{config[key]}", file=sys.stderr)
        print(type(val), file=sys.stderr)
        config[key] = float(val)

    

    i = 1
    try:
        os.mkdir(f"output/{config['name']}")
    except:
        pass
    while(True):
        try:
            os.mkdir(f"output/{config['name']}/{i}")
            break
        except:
            i += 1
    config["name"] = f"{config['name']}/{i}"
    


    # T = get_period(config["m1"], config["m2"], config["d"],
    #                config["e"], config["phase"])
    # print(T)
    # animate_separation(f"output/{config['name']}/end", sim,
    #                    config['t_end'], config['t_end']+.2)

# --------------------------------------------------------------------------------
# Resumable variant, used by run_one.py on HTCondor.  Shares build_sim() and
# summarize() with run_model above so the two paths cannot diverge on the physics.
# --------------------------------------------------------------------------------

DONE = 0        # completed; summary.json written
REJECTED = 1    # draw halted inside the 5-yr startup window -- caller redraws
CHECKPOINTED = 2  # checkpoint on disk; caller should exit with the checkpoint code


def run_model_resumable(config, mode, outdir=".", *, ckpt_dir=None,
                        should_checkpoint=None, attempt=0, seed=0, git_hash=None):
    """Run one simulation, able to resume from and write checkpoints.

    Returns one of DONE / REJECTED / CHECKPOINTED.

    On resume, ``config`` is ignored in favour of the config stored in the checkpoint,
    and E0/L0 come from the checkpoint rather than being recomputed -- recomputing them
    against the resumed state would silently invalidate E_Err and L_err.
    """
    from binary_planets import checkpoint as ckpt
    from binary_planets.sim import CheckpointRequested, simulate_resumable

    ckpt_dir = ckpt_dir or outdir
    base = f"{outdir}/{config['name']}"
    os.makedirs(base, exist_ok=True)

    resume = None
    t_wall_accum = 0.0

    if ckpt.exists(ckpt_dir):
        try:
            sim, log, meta = ckpt.load(ckpt_dir, expect_mode=mode)
            if meta["seed"] != seed or meta["config"]["name"] != config["name"]:
                raise ckpt.CheckpointMismatch(
                    f"checkpoint is for {meta['config']['name']} (seed {meta['seed']})")
            config = meta["config"]
            E0 = meta["energy0"]
            Lx0, Ly0, Lz0 = meta["angmom0"]
            resume = (meta["phase"], meta["i_next"])
            t_wall_accum = meta["t_wall_accum"]
            attempt = meta["attempt"]
            seed = meta["seed"]
        except ckpt.CheckpointMismatch as exc:
            # Never resume into a state we cannot vouch for: a repeated run is cheap,
            # a silently corrupt one is not.
            print(f"checkpoint unusable ({exc}); restarting run", file=sys.stderr)
            ckpt.clear(ckpt_dir)
            resume = None

    if resume is None:
        with open(f"{base}/run_notes.out", 'w+') as run_notes:
            run_notes.write(f"{sys.argv}\n")
            run_notes.write(f"Run start time: {datetime.datetime.now()}\n")
            for key in config.keys():
                run_notes.write(f"{key}: {config[key]}\n")
            run_notes.write("*************************************************\n\n")
            sim, log = build_sim(config, mode, run_notes)

        with open(f"{base}/config.json", "w+") as cf:
            json.dump(config, cf)

        E0 = sim.energy()
        Lx0, Ly0, Lz0 = sim.angular_momentum()

    t0 = time.time()
    try:
        _, log, halt, t_end, _phase, _i = simulate_resumable(
            sim, log, config["t_end"], mode,
            resume=resume, should_checkpoint=should_checkpoint,
            a_com0=config["binary"]["a_com0"] if mode == 2 else None,
        )
    except CheckpointRequested as c:
        ckpt.save(ckpt_dir, sim, c.log, phase=c.phase, i_next=c.i_next,
                  config=config, mode=mode, energy0=E0, angmom0=(Lx0, Ly0, Lz0),
                  t_wall_accum=t_wall_accum + (time.time() - t0),
                  attempt=attempt, seed=seed, git_hash=git_hash)
        return CHECKPOINTED

    if t_end < 5:
        ckpt.clear(ckpt_dir)
        return REJECTED

    t_elapsed = t_wall_accum + (time.time() - t0)
    save_log(log, base)

    sum_data = summarize(sim, halt, t_end, t_elapsed, E0, (Lx0, Ly0, Lz0), mode=mode)
    sum_data["n_rejected"] = attempt
    sum_data["seed"] = seed
    with open(f"{base}/summary.json", "w+") as cf:
        json.dump(sum_data, cf)

    ckpt.clear(ckpt_dir)
    del sim, log
    return DONE
