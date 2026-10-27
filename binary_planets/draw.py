# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Draw one run configuration from the priors.

Both drivers use this module, ``run_planets.py`` on LSF and ``run_one.py`` on
OSG/HTCondor, so every run is drawn by the same code.

Keep the draws in their current order. Reordering them changes which configuration a
given seed produces, and moving a draw into or out of a rejection loop changes the
prior the runs actually sample. That prior already differs from the nominal
distributions: the binary separation has median d/r_hill = 0.32, not 0.5.

Units are AU, yr and Msun.
"""

import numpy as np

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

SYSTEM_MAX_ATTEMPTS = 10_000
INJECT_MAX_ATTEMPTS = 20_000


class NoRoomForInjection(RuntimeError):
    """This realisation of the known system cannot host the injected object.

    Raised when the known planets are drawn onto crossing orbits, or when the gap they
    leave admits no acceptable injected orbit.  Both mean the *drawn system* is the
    problem, not the injected draw, so draw_config redraws the whole system.
    """



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

    m_star = float(get_stellar_mass(system))

    # Axes without published errors scale with m_star, so use the stored value.
    semi_majors = get_semimajor(system, m_star)
    es = get_eccen(system)


    # The known planets are observed to coexist, so reject draws where any two cross.
    # The caller redraws the whole system; redrawing one planet would bias it toward
    # the edge of its error bar.
    if known_planets_cross(semi_majors, es):
        raise NoRoomForInjection(f"{system['name']}: known planets drawn crossing")

    cfg["m_star"] = m_star

    # Gap edges from the drawn axes, not the catalog.
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

    A draw of the known system is rejected if two known planets cross, or if the gap
    between them has no room for the injected orbit. Either way the whole system is
    redrawn, since redrawing one planet would bias it toward the edge of its error bar.

    Rejection is rare for most systems but common for K2-72, Kepler-223 and Kepler-338.
    For those three, the known-planet eccentricities are effectively drawn conditional
    on the planets not crossing.
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
