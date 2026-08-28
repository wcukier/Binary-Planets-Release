from scipy.stats import skewnorm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import numpy as np
from .forecaster import mr_forecast as mr
import sys as system
def _cdf(x,a,b,c):
    return skewnorm.cdf(x,a,b,c)

mass, cdf = np.load("data/st_mass_dist.npy")
mass_func = interp1d(cdf, mass)
MASS_CDF_MIN = np.min(cdf)
MASS_CDF_MAX = np.max(cdf)

e, cdf = np.load("data/e_dist.npy")
e_func = interp1d(cdf, e)
E_CDF_MIN = np.min(cdf)
E_CDF_MAX = np.max(cdf)

def draw_rv(mid, lo, hi, num=1):
    if (np.isnan(lo) * np.isnan(hi)* (mid==0)): return 0
    if np.isnan(lo): lo = -.2*mid
    if np.isnan(hi): hi = .2*mid
    # try:
    return np.random.uniform(mid+lo, mid+hi)
        # [a,b,c], _ = curve_fit(_cdf, [mid+lo, mid, mid+hi], [.16, .5, .84]) 
        # print(a,b,c)
        # return skewnorm.rvs(a,b,c, size=num)   
    # except:
    #     return np.random.normal(mid, np.max([lo, hi]))
    
def get_stellar_mass(sys):
    # Fallback is for missing masses only; a `mass < 0.1` guard also caught TRAPPIST-1.
    mass = sys["st_mass"]
    if np.isnan(mass) or mass <= 0:
        x = np.random.uniform(MASS_CDF_MIN, MASS_CDF_MAX)
        return mass_func(x)
    return draw_rv(mass, sys["st_lower"], sys["st_upper"])
    
def get_eccen(sys):
    mids = sys["e"]
    los = sys["e_lower"]
    his = sys["e_upper"]
    es = np.zeros(len(mids))
    for i in range(len(es)):
        if (np.isnan(mids[i]) + np.isnan(los[i]) + np.isnan(his[i])):
            x = np.random.uniform(E_CDF_MIN, E_CDF_MAX)
            # es[i] = e_func(x)
            es[i]=0
        else:
            # e[i] = 0
            while True:
                es[i] = draw_rv(mids[i], los[i], his[i])
#                 print(es[i])
                if es[i]>=0: break

    return es

def get_inc(sys):
    mids = np.array(sys["inc"])
    
    # if np.all(np.isnan(mids)):
    return np.ones(len(mids)) *np.pi/2
    
    # incs = np.zeros(len(mids))
    # los = sys["inc_lower"]
    # his = sys["inc_upper"]
    
    # mean = np.mean(mids[~np.isnan(mids)])
    # std = np.std(mids[~np.isnan(mids)])
    
    # for i in range(len(mids)):
    #     if np.isnan(mids[i]):
    #         incs[i] = np.random.normal(mean, std)
    #     else:
    #         incs[i] = draw_rv(mids[i], los[i], his[i])
            
    # return incs * np.pi/180

MAX_MASS_ATTEMPTS = 10_000


def get_mass(sys):
    # Use the measured mass when there is one; fall back to the mass-radius relation
    # only when the mass is absent or flagged as an upper limit (mass_lim == 1).
    mass_mid = sys["mass"]
    mass_upper = sys["mass_upper"]
    mass_lower = sys["mass_lower"]
    mass_lim = sys["mass_lim"]
    r_mid = sys["r"]
    r_upper = sys["r_upper"]
    r_lower = sys["r_lower"]
    mass = np.zeros(len(mass_mid))
    for i in range(len(mass)):
        no_measurement = np.isnan(mass_mid[i]) or mass_lim[i] == 1
        for attempt in range(MAX_MASS_ATTEMPTS):
            override_break = 0
            if no_measurement:
                r = draw_rv(r_mid[i], r_lower[i], r_upper[i])
                mass[i] = mr.Rpost2M([r], "Earth", 1e5, classify="No")[0]
                # Reject only against a genuine upper limit, never a central value.
                if mass_lim[i] == 1 and mass[i] > mass_mid[i]:
                    override_break = 1
            else:
                mass[i] = draw_rv(mass_mid[i], mass_lower[i], mass_upper[i])

            if mass[i] > 0 and override_break == 0:
                break
        else:
            raise RuntimeError(
                f"could not draw a valid mass for planet {i} of {sys['name']} in "
                f"{MAX_MASS_ATTEMPTS} attempts (mass={mass_mid[i]}, lim={mass_lim[i]}, "
                f"r={r_mid[i]})"
            )
    return mass
            
# --- LLM-GENERATED — UNVERIFIED (tier: simulation) -----------------------------------
# get_semimajor / stellar_axis_scale / gap_indices below were rewritten 2026-08-28 to fix
# the defect recorded as task 0.8: the old version drew every axis independently, falling
# back to U(0.8a, 1.2a) for the 63% of planets with no published error bars.  That was wide
# enough to re-order the known planets in 4.95% of runs and to put the injected planet
# outside the drawn gap in 13.6%.  Remove this banner only after POLICY.md §2 checks.
AXIS_MAX_ATTEMPTS = 10_000  # ordering-rejection guard; see get_semimajor


def stellar_axis_scale(sys, m_star):
    """Factor by which a system's semi-major axes scale with the drawn stellar mass.

    Parameters
    ----------
    sys : dict-like
        One entry of ``compact_systems_run_composite.npy``.
    m_star : float
        Stellar mass drawn for this run, in Msun (from :func:`get_stellar_mass`).

    Returns
    -------
    float
        ``(m_star / st_mass)**(1/3)``, dimensionless; 1.0 when the catalog has no stellar
        mass to anchor against.

    Notes
    -----
    For a transiting planet the semi-major axis is not measured directly.  It is derived
    from the period and the stellar mass, ``a = (G M_star P^2 / 4 pi^2)^(1/3)``, and the
    period is known to many significant figures.  The *ratios* ``a_i/a_j = (P_i/P_j)^(2/3)``
    are therefore essentially exact, and the uncertainty is almost entirely an uncertainty
    in the overall scale, carried by ``M_star``.

    So the architecture scales rigidly.  Applying this common factor keeps the planets in
    their observed order by construction, which independent per-planet draws do not.
    """
    m_cat = sys["st_mass"]
    if np.isnan(m_cat) or m_cat <= 0:
        # get_stellar_mass fell back to the IMF draw, which is not tied to this system, so
        # there is no meaningful catalog anchor to scale against.
        return 1.0
    return float(m_star / m_cat) ** (1.0 / 3.0)


def gap_indices(sys):
    """Indices of the two known planets bounding the dynamical gap.

    Parameters
    ----------
    sys : dict-like
        One entry of ``compact_systems_run_composite.npy``; uses ``a`` and ``gap``.

    Returns
    -------
    tuple of (int, int)
        ``(i_inner, i_outer)`` into the system's planet arrays.

    Notes
    -----
    ``gap`` stores the two bounding planets' *catalog* semi-major axes, so the indices are
    recovered by matching against ``a``.  Callers need the indices, not the catalog values,
    because the axes actually simulated are redrawn per run.
    """
    a = np.asarray(sys["a"], dtype=float)
    return (int(np.argmin(np.abs(a - sys["gap"][0]))),
            int(np.argmin(np.abs(a - sys["gap"][1]))))


def get_semimajor(sys, m_star):
    """Draw the known planets' semi-major axes, preserving their observed ordering.

    Parameters
    ----------
    sys : dict-like
        One entry of ``compact_systems_run_composite.npy``.
    m_star : float
        Stellar mass drawn for this run, in Msun.  Must be the same value stored in the
        run config, so that the axes and the stellar mass are mutually consistent.

    Returns
    -------
    np.ndarray
        Semi-major axes in AU, shape ``(n_planets,)``, strictly increasing.

    Raises
    ------
    RuntimeError
        If ordering could not be satisfied in ``AXIS_MAX_ATTEMPTS`` draws.

    Notes
    -----
    Two cases, per Wolf's decision of 2026-08-28:

    - A planet **with** published error bars on both sides is drawn from them, as before.
      A published value reflects a real analysis and is trusted over a derived one.
    - A planet **without** them is placed by rigidly scaling its catalog axis with the
      drawn stellar mass (:func:`stellar_axis_scale`), rather than by the old
      ``U(0.8a, 1.2a)`` fallback, which was some fourteen times wider than the stellar mass
      justifies and was the root cause of task 0.8.

    Mixing the two can still invert an adjacent pair, so ordering is enforced by rejection.
    Only 4 of 226 adjacent pairs have published ranges wide enough to overlap at all
    (in KOI-351 and Kepler-186), so the rejection rate is small and the loop is not a
    practical cost.  The whole system is redrawn on rejection rather than the offending
    planet alone, which would bias the survivor toward the edge of its error bar.
    """
    a_cat = np.asarray(sys["a"], dtype=float)
    lo = np.asarray(sys["a_lower"], dtype=float)
    hi = np.asarray(sys["a_upper"], dtype=float)
    published = ~(np.isnan(lo) | np.isnan(hi))
    scaled = a_cat * stellar_axis_scale(sys, m_star)

    for _ in range(AXIS_MAX_ATTEMPTS):
        semi_majors = scaled.copy()
        for i in np.flatnonzero(published):
            semi_majors[i] = draw_rv(a_cat[i], lo[i], hi[i])
        if np.all(np.diff(semi_majors) > 0):
            return semi_majors

    raise RuntimeError(
        f"could not draw an ordered set of semi-major axes for {sys['name']} in "
        f"{AXIS_MAX_ATTEMPTS} attempts"
    )
# --- end LLM-GENERATED block ---------------------------------------------------------


# ---------------------------------------------------------------------------
# Orbital geometry
# ---------------------------------------------------------------------------

# Lidov-Kozai critical inclination, arccos(sqrt(3/5)).
# Grundy+2019 doi:10.1016/j.icarus.2019.03.035; Li+2014 doi:10.1088/0004-637X/791/2/86
LK_CRIT_DEG = 39.2315


def mutual_inclination(inc, bin_inc, Omega):
    """Angle (rad) between the pair's mutual orbit and its orbit about the star.

    Takes raw config values; neither stored inclination is this angle.
    """
    i1 = np.pi - np.asarray(inc, dtype=float)        # mutual orbit
    i2 = np.pi - np.asarray(bin_inc, dtype=float)    # COM orbit about the star
    d_node = np.asarray(Omega, dtype=float)          # COM orbit node is fixed at 0
    cos_i = (np.cos(i1) * np.cos(i2)
             + np.sin(i1) * np.sin(i2) * np.cos(d_node))
    return np.arccos(np.clip(cos_i, -1.0, 1.0))


def mutual_inclination_from_cfgs(cfgs):
    """Vectorised mutual_inclination() over a list of run configs. Returns radians."""
    b = [c["binary"] for c in cfgs]
    return mutual_inclination(
        np.array([x["inc"] for x in b]),
        np.array([x["bin_inc"] for x in b]),
        np.array([x["Omega"] for x in b]),
    )


def in_lk_window(i_mut_rad):
    """True where the mutual inclination lies in the quadrupole LK window."""
    deg = np.degrees(i_mut_rad)
    return (deg > LK_CRIT_DEG) & (deg < 180.0 - LK_CRIT_DEG)
