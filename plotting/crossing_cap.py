
"""Largest eccentricity the draw's orbit-crossing test allows for the injected body.

draw.py redraws the injected orbit while its radial range [a(1-e) - b, a(1+e) + b]
overlaps any known planet's [a_k(1-e_k), a_k(1+e_k)], with b = 0 for a single planet and
b = d (the drawn pair separation) for a binary's centre of mass.  At fixed a (and d) the
accepted e is therefore uniform on [0, e_max), so e / e_max is uniform in the prior.
"""
import numpy as np


def e_max(cfg, mode):
    """e_max for a stored config; mode 1 = single injection, 2 = binary."""
    n_known = cfg["n_secondary"] - 1 if mode == 1 else cfg["n_secondary"]
    ak = np.array([cfg[f"secondary_{k}"]["a"] for k in range(n_known)])
    ek = np.array([cfg[f"secondary_{k}"]["e"] for k in range(n_known)])
    if mode == 1:
        a, b = cfg[f"secondary_{n_known}"]["a"], 0.0
    else:
        a, b = cfg["binary"]["a"], cfg["binary"]["d"]
    inner = ak < a
    lim = [1.0]
    if (~inner).any():
        lim.append(np.min((ak[~inner] * (1 - ek[~inner]) - b) / a - 1))
    if inner.any():
        lim.append(np.min(1 - (ak[inner] * (1 + ek[inner]) + b) / a))
    return min(lim)
