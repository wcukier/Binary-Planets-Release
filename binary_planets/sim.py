import numpy as np
import rebound
from .constants import *
from .log import *
import sys

def orbital_charcteristics(m1, m2, d, e=0, phase=0, Omega=0, inc=0):
    a1 = m2/(m1+m2)*d
    a2 = m1/(m1+m2)*d
    

    
    r1 = get_separation(a1, e, phase)
    r2 = get_separation(a2, e, phase)

    T1 = np.sqrt(4*np.pi**2/(G_AU_YR_MSUN * (m1+m2)) * (a1+a2)**3)
    T2 = T1
    
    
    v1 = get_velocity(r1, a1, e, T1)
    v2 = get_velocity(r2, a2, e, T2)
    

    
    xy1 = np.array([r1*np.cos(phase), r1*np.sin(phase), 0])
    xy2 = np.array([r2*np.cos(phase+np.pi), r2*np.sin(phase+np.pi), 0])
    
    

    rot = np.array([[np.cos(Omega), -np.sin(Omega)*np.cos(inc), np.sin(Omega)*np.sin(inc)],
                    [np.sin(Omega), np.cos(Omega)*np.cos(inc), -np.cos(Omega)*np.sin(inc)],
                    [0,             np.sin(inc),                np.cos(inc)]])
    
    x_vec1 = np.matmul(rot, xy1)
    x_vec2 = np.matmul(rot, xy2)
    
    v_dir1 = np.cross(x_vec1, np.matmul(rot, [0,0,1]))
    v_dir2 = np.cross(x_vec2, np.matmul(rot, [0,0,1]))
    
    
    v_hat1 = v_dir1/np.linalg.norm(v_dir1)
    v_hat2 = v_dir2/np.linalg.norm(v_dir2)
    
    
    v_vec1 = v_hat1 * v1
    v_vec2 = v_hat2 * v2

    return [x_vec1, v_vec1], [x_vec2, v_vec2]
    
    
def get_period(m1, m2, d, e=0, phase=0):
    m1=m1*MASS_E/MASS_SUN
    m2=m2*MASS_E/MASS_SUN
    
    r1 = m2/(m1+m2)*d
    r2 = m1/(m1+m2)*d
    
    a1 = get_semi_major(r1, e, phase)
    a2 = get_semi_major(r2, e, phase)

    return np.sqrt(4*np.pi**2/(G_AU_YR_MSUN * (m1+m2)) * (a1+a2)**3)
    
    
    
def get_semi_major(r, e, theta):
    return r*(1 + e*np.cos(theta))/(1 - e**2)

def get_separation(a, e, theta):
    return (1 - e**2)/(1- e* np.cos(theta)) * a


def get_velocity(r, a , e, T):
    return 2* np.pi * (a**2) * np.sqrt(1-e**2)/(r*T)


def get_hill_radius(a, e, m, m_star):
    return a * (1-e) * ((m*MASS_E)/(3*m_star*MASS_SUN))**(1/3)


def orbits_cross(a1, e1, a2, e2, buffer1=0, buffer2=0):
    """Return True if two orbits cross, i.e. their periapsis/apoapsis radial
    ranges (about the star) overlap so that one is neither entirely inside nor
    entirely outside the other.

    ``buffer1``/``buffer2`` widen each orbit's radial range to account for the
    physical extent of the body about its own barycenter -- e.g. for a binary
    planet, pass the maximum displacement of a component from the binary
    barycenter so the whole pair is checked, not just its center of mass."""
    peri1, apo1 = a1 * (1 - e1) - buffer1, a1 * (1 + e1) + buffer1
    peri2, apo2 = a2 * (1 - e2) - buffer2, a2 * (1 + e2) + buffer2
    return max(peri1, peri2) <= min(apo1, apo2)




def known_planets_cross(semi_majors, eccentricities):
    """True if any two of the known planets are drawn onto crossing orbits.

    The known planets are observed and have coexisted for Gyr, so a draw that puts two
    of them on crossing orbits describes a system that cannot exist.  Used to reject
    such draws before they are simulated -- see the note in binary_planets.draw.
    """
    n = len(semi_majors)
    return any(orbits_cross(semi_majors[j], eccentricities[j],
                            semi_majors[k], eccentricities[k])
               for j in range(n) for k in range(j + 1, n))


def init_sim(m_star, n_log=1000, integrator="whfast",
             dt=1e-4, n_particles=2, run_notes=False):
    sim=rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = integrator
    sim.dt = dt
    sim.ri_whfast.safe_mode = 0
    sim.ri_whfast.corrector = 11
    sim.add(m=m_star)
    log = init_log(n_log, n_particles)

    return sim, log
     


def init_binary_planet(sim, m_star, m1, m2, d, a=1, e=0,
                       e_sys=0, phase=0, Omega=0, inc=0,
                       bin_inc=0, run_notes=None):
    """Add a bound planet pair on an orbit about the star.

    NOTE ON THE ANGLE NAMES -- they read backwards:

        inc      inclination of the pair's MUTUAL orbit (m1 about m2).  Carries Omega.
        bin_inc  inclination of the pair's CENTRE-OF-MASS orbit about the star.
                 Its node is fixed at 0, which is harmless: the known planets are
                 coplanar with random Omega, so the system is azimuthally symmetric.

    Both are measured from the reference plane, so neither is the mutual inclination
    between the two orbital planes.
    """
    inc = np.pi - inc
    bin_inc = np.pi - bin_inc

    m1=m1*MASS_E/MASS_SUN
    m2=m2*MASS_E/MASS_SUN

    sim.move_to_hel()
    [x1, v1], [x2, v2] = orbital_charcteristics(m1, m2, d, inc=inc, e=e, 
                                                phase=phase, Omega=Omega)

    # oribtal charateristics of the combined planetary system
    _, [[x0, y0, z0], [vx0, vy0, vz0]] = orbital_charcteristics(m_star, m1+m2, a,
                                                                inc=bin_inc,
                                                                e=e_sys)
    if run_notes:
        run_notes.write(f"e: {e}\t e_sys:{e_sys}")


    sim.add(m=m1, 
            x=x0+x1[0], y=y0+x1[1], z=z0+x1[2], 
            vx=vx0+v1[0],vy=vy0+v1[1], vz=vz0+v1[2])
    
    sim.add(m=m2,
            x=x0+x2[0], y=y0+x2[1], z=z0+x2[2], 
            vx=vx0+v2[0], vy=vy0+v2[1], vz=vz0+v2[2])
    
    return sim


def init_single_planet(sim, m, a, e, inc, omega, Omega):
    # inc = np.pi-inc
    sim.move_to_hel()
    mass = m * MASS_E/MASS_SUN
    sim.add(m=mass, a=a, e=e, inc=inc, omega=omega, Omega=Omega)
    sim.move_to_com()
    return sim

# LLM-GENERATED — UNVERIFIED (tier: simulation): CheckpointRequested,
# simulate_resumable, and simulate()'s delegation to it.  The rest of this module
# predates that change.  See Assistant/POLICY.md §2.
# The 5-year startup window: 100 samples over [0, 10] yr.  A run that halts anywhere in
# here is a *rejected draw*, not a failure -- run_model returns 1 and the caller redraws.
STARTUP_T_END = 10
STARTUP_STEPS = 100


class CheckpointRequested(Exception):
    """Raised out of simulate_resumable when should_checkpoint() asks to stop.

    Carries the exact resume point so the caller can serialize it.  Only ever raised
    from the main loop -- the startup window is under 5 yr of integration and is where
    draw rejection happens, so interrupting it would be pointless and would complicate
    the retry semantics.
    """

    def __init__(self, phase, i_next, t, log):
        super().__init__(f"checkpoint requested at {phase}[{i_next}], t={t}")
        self.phase = phase
        self.i_next = i_next
        self.t = t
        self.log = log        # carries the live cursor; see the note in the main loop


def simulate_resumable(sim, log, t_end, mode, *, resume=None, should_checkpoint=None):
    """Integrate, optionally resuming mid-run and optionally stopping to checkpoint.

    Parameters
    ----------
    resume : tuple or None
        ``(phase, i_next)`` from a checkpoint, where phase is "startup" or "main" and
        i_next indexes the loop's own linspace.  None starts from the beginning.
    should_checkpoint : callable or None
        Polled once per log step in the main loop.  If it returns True,
        CheckpointRequested is raised.

    Returns
    -------
    (sim, log, halt, t, phase, i_next)
        `phase`/`i_next` describe where integration stopped, so a halt and a completed
        run are distinguishable from a checkpoint without inspecting the arrays.

    Notes
    -----
    Neither the loop phase nor the write cursor is recoverable from the log arrays:
    the startup loop resets ``log[3] = 1`` on every iteration, so all but the first
    startup sample overwrite row 1, and ``get_log_len`` returns capacity rather than
    fill.  Both must be carried explicitly in the checkpoint.
    """
    # init_log returns a tuple; log_elements returns a list.  Normalise once so the
    # cursor is assignable regardless of which the caller passed.
    log = list(log)
    n_log = get_log_len(log)
    startup_ts = np.linspace(0, STARTUP_T_END, STARTUP_STEPS)
    main_ts = np.linspace(0, t_end, n_log)[1:]
    # Main targets below STARTUP_T_END would integrate backwards (TRACE segfaults).
    if len(main_ts) and main_ts[0] < STARTUP_T_END:
        raise ValueError(f"log cadence t_end/(n_log-1) = {main_ts[0]:.3g} yr is below the "
                         f"{STARTUP_T_END} yr startup window; raise t_end or lower n_log")

    phase, i0 = resume if resume is not None else ("startup", 0)
    halt = ""
    t = 0.0

    # log_elements returns a *new* list with t_step rebound while mutating the three
    # arrays in place.  Write the cursor back into our own list rather than rebinding,
    # and hand that list out on every exit path -- including CheckpointRequested, which
    # otherwise unwinds leaving the caller with a stale cursor and a silently truncated
    # log on resume.
    if phase == "startup":
        for i in range(i0, len(startup_ts)):
            t = startup_ts[i]
            sim.integrate(t, exact_finish_time=0)
            stepped, halt = log_elements(sim, log, mode)
            log[3] = stepped[3]
            if halt:
                print("Unstable system.  Stopping...", file=sys.stderr)
                return sim, log, halt, t, "startup", i + 1
            log[3] = 1
        phase, i0 = "main", 0

    for i in range(i0, len(main_ts)):
        t = main_ts[i]
        sim.integrate(t, exact_finish_time=0)
        stepped, halt = log_elements(sim, log, mode)
        log[3] = stepped[3]
        if halt:
            print("Unstable system.  Stopping...", file=sys.stderr)
            return sim, log, halt, t, "main", i + 1
        if should_checkpoint is not None and should_checkpoint():
            raise CheckpointRequested("main", i + 1, t, log)

    return sim, log, halt, t, "main", len(main_ts)


def simulate(sim, log, t_end, mode):
    """Original entry point: integrate straight through, no checkpointing.

    Delegates to simulate_resumable so the LSF and OSG paths cannot drift apart.
    """
    sim, log, halt, t, _phase, _i_next = simulate_resumable(sim, log, t_end, mode)
    return sim, log, halt, t

    
    

