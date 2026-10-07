# LLM-GENERATED — UNVERIFIED (tier: simulation)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Draw one run configuration from the priors.

Extracted verbatim from ``run_planets.py:one_run`` so that the LSF driver
(``run_planets.py``) and the single-simulation driver (``run_one.py``, used on
OSG/HTCondor) consume *the same* prior code.  Two copies of this logic would make the
single-vs-binary survival comparison the paper turns on impossible to verify.

The RNG consumption order is load-bearing and must not be reordered: the realised prior
is the result of the draws *and* the rejection loops together, and changing the order
changes the realised posterior even though every individual distribution is unchanged.
See revision-plan task 0.1 item 4, where the realised binary separation prior comes out
at median d/r_hill = 0.32 rather than the nominal 0.5.

Units follow the simulation: AU, yr, Msun.
"""

import numpy as np

# Rejection guards.  Matches the AXIS_MAX_ATTEMPTS / MAX_MASS_ATTEMPTS convention in
# binary_planets.utils: bound the loop, and raise loudly rather than spin.
SYSTEM_MAX_ATTEMPTS = 10_000
INJECT_MAX_ATTEMPTS = 20_000


class NoRoomForInjection(RuntimeError):
    """This realisation of the known system cannot host the injected object.

    Raised when the known planets are drawn onto crossing orbits, or when the gap they
    leave admits no acceptable injected orbit.  Both mean the *drawn system* is the
    problem, not the injected draw, so draw_config redraws the whole system.
    """

from binary_planets.sim import (get_hill_radius, known_planets_cross,
                                orbits_cross)
from binary_planets.forecaster import mr_forecast as mr
from binary_planets.utils import (
    draw_rv,
    gap_indices,
    get_eccen,
    get_inc,
    get_mass,
    get_semimajor,
    get_stellar_mass,
)


def _draw_once(system, mode, *, dt, t_end, integrator, name, n_log):
    """Draw one run configuration.

    Pure function of ``system``, ``mode`` and the state of ``np.random``.  Performs no
    I/O and creates no directories -- the caller owns naming and output layout.

    Parameters
    ----------
    system : dict-like
        One entry of ``data/compact_systems_run_composite.npy``.
    mode : int
        0 = no injection, 1 = single injection, 2 = binary injection.
    dt, t_end : float
        Timestep and integration time, in years.
    integrator : str
        Rebound integrator name, e.g. ``"trace"``.
    name : str
        Run name; becomes the output path stem.
    n_log : int
        Number of log samples to allocate.

    Returns
    -------
    dict
        The config consumed by ``binary_planets.run_model.run_model``.
    """
    cfg = {
        "name": name,
        "binary": {},
        "n_log": n_log,
    }

    n_secondary = len(system["a"])
    cfg["n_secondary"] = n_secondary
    cfg["t_end"] = t_end

    if mode == 1:
        cfg["n_secondary"] += 1

    # The known planets are observed and have coexisted for Gyr, so a draw placing any
    # two of them on crossing orbits describes a system that cannot exist.  This is the
    # same argument that motivates the ordering rejection in get_semimajor (task 0.8);
    # rejecting only ordering left the subtler case where the planets keep their order
    # but their radial ranges overlap.  Measured over 300 draws per system that happens
    # in 3.76% of draws catalogue-wide, concentrated in the most compact systems --
    # K2-72 91.7%, Kepler-223 84.7%, Kepler-338 64.3% -- which are precisely the three
    # systems that came up short at 0% survival in the leapfrog pass.
    #
    # When the crossing pair is the pair bounding the gap, the injection region is
    # closed outright and the rejection loops below had no acceptable draw at all: they
    # spun forever.  Rejecting here removes that at the source, so those loops now
    # terminate by construction rather than by their guard.
    #
    # The whole system is redrawn rather than the offending planet, for the reason
    # get_semimajor gives: redrawing one planet biases it toward its error-bar edge.
    m_star = float(get_stellar_mass(system))
    # The axes must be drawn against the same stellar mass stored in the config: for
    # planets without published error bars they scale rigidly with it.  See task 0.8.
    semi_majors = get_semimajor(system, m_star)
    es = get_eccen(system)
    if known_planets_cross(semi_majors, es):
        raise NoRoomForInjection(f"{system['name']}: known planets drawn crossing")

    cfg["m_star"] = m_star
    # Gap edges as *simulated*, not as catalogued.  Drawing the injected planet
    # between the catalog edges while the bounding planets moved is what put 13.6% of
    # runs outside the gap they were supposed to be in.
    i_in, i_out = gap_indices(system)
    gap_lo, gap_hi = semi_majors[i_in], semi_majors[i_out]
    incs = get_inc(system)
    masses, mass_radii = get_mass(system, return_radii=True)

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
        cfg[f"secondary_{j}"]["m"] = np.random.uniform(0.38, 72)

        # Draw the injected planet, redrawing if its orbit crosses any of
        # the existing planets' orbits.
        for _ in range(INJECT_MAX_ATTEMPTS):
            inj_a = 10 ** np.random.uniform(
                np.log10(gap_lo), np.log10(gap_hi)
            )
            inj_e = np.random.uniform(0, 1)
            if not any(
                orbits_cross(inj_a, inj_e, semi_majors[k], es[k])
                for k in range(n_secondary)
            ):
                break
        else:
            raise NoRoomForInjection(
                f"{system['name']}: no acceptable injected planet in "
                f"{INJECT_MAX_ATTEMPTS} attempts; gap [{gap_lo:.6f}, {gap_hi:.6f}]"
            )
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
        for _ in range(INJECT_MAX_ATTEMPTS):

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
        else:
            raise NoRoomForInjection(
                f"{system['name']}: no acceptable injected binary in "
                f"{INJECT_MAX_ATTEMPTS} attempts; gap [{gap_lo:.6f}, {gap_hi:.6f}]"
            )
        cfg["binary"]["a"] = bin_a
        cfg["binary"]["d"] = bin_d

    else:
        cfg["binary"]["m1"] = 1e-20
        cfg["binary"]["m2"] = 1e-20

    _draw_radii(cfg, system, mode, mass_radii)
    cfg["dt"] = dt
    cfg["integrator"] = integrator

    return cfg


# LLM-GENERATED — UNVERIFIED (tier: simulation): physical radii for collision detection.
FORECAST_MIN_MASS = 3e-4   # M_earth; Mpost2R returns None below this


def _forecast_radius(m_earth):
    return float(mr.Mpost2R([max(m_earth, FORECAST_MIN_MASS)], "Earth", classify="No")[0])


def _draw_radii(cfg, system, mode, mass_radii):
    """Radii in Earth radii (planets) and solar radii (star).

    Known planets keep the radius their mass was forecast from, if any (``mass_radii``,
    NaN otherwise), and otherwise draw uniformly between the catalogue bounds like the
    other parameters.  Injected bodies are forecast from their drawn mass.  The
    catalogue has no stellar radius, so the star uses the main-sequence R = M^0.8.
    Drawn last so the other parameters' draws are unchanged.
    """
    n_known = len(system["a"])
    for j in range(n_known):
        r = mass_radii[j]
        if np.isnan(r):
            r = draw_rv(system["r"][j], system["r_lower"][j], system["r_upper"][j])
        cfg[f"secondary_{j}"]["radius"] = float(r)
    if mode == 1:
        j = n_known
        cfg[f"secondary_{j}"]["radius"] = _forecast_radius(cfg[f"secondary_{j}"]["m"])
    if mode == 2:
        cfg["binary"]["radius1"] = _forecast_radius(cfg["binary"]["m1"])
        cfg["binary"]["radius2"] = _forecast_radius(cfg["binary"]["m2"])
    cfg["r_star"] = cfg["m_star"] ** 0.8


def draw_config(system, mode, *, dt, t_end, integrator, name, n_log=10000):
    """Draw one run configuration, redrawing the system when it admits no injection.

    Two things make a drawn realisation unusable, and both are properties of the *known*
    system rather than of the injected object:

    * the known planets land on crossing orbits -- they are observed to coexist, so such
      a draw describes a system that cannot exist (the same argument as the ordering
      rejection in get_semimajor, task 0.8);
    * the gap they leave admits no injected orbit at all.

    Either way the whole system is redrawn, for the reason get_semimajor gives: redrawing
    one element biases it toward the edge of its error bar.  Both used to be unbounded
    ``while True`` loops that spun forever on the most compact systems.

    Rejection is negligible for 61 of the 66 systems and heavy for three -- K2-72,
    Kepler-223 and Kepler-338 -- whose known-planet eccentricities are drawn conditional
    on the planets being mutually non-crossing.  That conditioning is a real change to
    the prior and needs stating in the methods.
    """
    for _ in range(SYSTEM_MAX_ATTEMPTS):
        try:
            return _draw_once(system, mode, dt=dt, t_end=t_end,
                              integrator=integrator, name=name, n_log=n_log)
        except NoRoomForInjection:
            continue
    raise RuntimeError(
        f"could not draw a usable configuration for {system['name']} in "
        f"{SYSTEM_MAX_ATTEMPTS} attempts"
    )
