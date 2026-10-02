# LLM-GENERATED — UNVERIFIED (tier: figure-pub)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Survival fraction against separation from the nearest known planet, in mutual Hill radii.

This is the variable the gap-edge structure is actually a function of. The rescaled
semi-major axis used in Figures 5 and 6 is a *catalog* coordinate: it is built from the
catalog gap edges, while the bounding planets' axes are redrawn every run. Mutual Hill
separation instead measures the configuration that was simulated.

The dashed line marks ``2 sqrt(3) = 3.46`` mutual Hill radii, the two-planet Hill-stability
criterion of Gladman (1993), https://doi.org/10.1006/icar.1993.1169. It is a two-planet
result applied to a multi-planet system, so it is indicative, not a threshold -- shown
because a large fraction of the prior is drawn interior to it. ``orbits_cross``
(``binary_planets/sim.py``) tests radial-range overlap only, so nested near-circular orbits
pass the rejection at any separation and no minimum-separation criterion is imposed.

The shaded histogram is the number of runs per bin -- i.e. the prior actually sampled, after
the crossing rejection. Read it alongside the survival curve: bins where the prior is thin
carry large binomial errors.

**The curve is not monotonic**, and the rise below ~1 mutual Hill radius is the 1:1
co-orbital island: survivors inside 2 mutual Hill radii sit at a median period ratio of 1.02,
and 78% of them below 1.05. Exactly co-orbital circular orbits are rejected by
``orbits_cross`` (identical radial ranges overlap), but ``a`` and ``a + epsilon`` at
``e ~ 0`` are not, so Trojan and horseshoe configurations enter the prior and survive.

Note these two axes are **degenerate at small separation** -- the mutual Hill radius is
~0.02 a for these masses, so ``delta < 1.5`` implies a period ratio under 1.05. The shaded
band marks the co-orbital region rather than pretending a period-ratio cut is an independent
control on it.

Usage
-----
    python plotting/hill_separation_survival.py [--mode sin_inj] [--root output_old]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tools.physics.gap_structure import collect_hill  # noqa: E402

os.makedirs(HERE / "figs", exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "font.size": 20.0,
})

# Petroff 10-colour cycle
PETROFF10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
             "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

HILL_STABILITY = 2.0 * np.sqrt(3.0)  # Gladman (1993) two-planet criterion, mutual Hill radii
CO_ORBITAL = 1.05                    # period ratio below which a run counts as near-co-orbital.
                                     # Reporting choice: survivors inside 2 mutual Hill radii have
                                     # a median period ratio of 1.02, so this brackets the 1:1
                                     # island without reaching the 5:4 resonance at 1.25.
N_BINS = 24                          # log-spaced bins in mutual Hill separation
DELTA_MIN, DELTA_MAX = 0.1, 60.0     # plotted range in mutual Hill radii


def binned_survival(delta: np.ndarray, alive: np.ndarray, edges: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Survival fraction per bin with binomial 1-sigma errors.

    Parameters
    ----------
    delta : np.ndarray
        Separation in mutual Hill radii, shape ``(n_runs,)``.
    alive : np.ndarray
        Survival flag, shape ``(n_runs,)`` of bool.
    edges : np.ndarray
        Bin edges, shape ``(n_bins + 1,)``.

    Returns
    -------
    centres, frac, err, counts : np.ndarray
        Geometric bin centres; survival fraction; its 1-sigma binomial error; and the number
        of runs per bin. ``frac`` and ``err`` are ``nan`` where a bin is empty.
    """
    idx = np.digitize(delta, edges) - 1
    frac = np.full(edges.size - 1, np.nan)
    err = np.full(edges.size - 1, np.nan)
    counts = np.zeros(edges.size - 1, dtype=int)
    for i in range(edges.size - 1):
        sel = idx == i
        n = int(sel.sum())
        counts[i] = n
        if n:
            p = alive[sel].mean()
            frac[i], err[i] = p, np.sqrt(p * (1 - p) / n)
    return np.sqrt(edges[:-1] * edges[1:]), frac, err, counts


def main() -> None:
    """Build the survival-vs-Hill-separation figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="sin_inj", choices=["sin_inj", "bin_inj"])
    parser.add_argument("--root", type=pathlib.Path, default=ROOT / "output_old",
                        help="output directory (default: output_old/, the published whfast pass)")
    args = parser.parse_args()

    d = collect_hill(args.mode, args.root)
    delta, alive = d["delta"], d["survived"]
    good = np.isfinite(delta) & (delta > 0)
    delta, alive = delta[good], alive[good]

    edges = np.logspace(np.log10(DELTA_MIN), np.log10(DELTA_MAX), N_BINS + 1)
    centres, frac, err, counts = binned_survival(delta, alive, edges)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    prior = ax.twinx()
    prior.hist(delta, bins=edges, color=PETROFF10[3], alpha=0.30,
               label="All runs (sampled prior)")
    prior.set_ylabel(r"Runs per bin")
    prior.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    ok = np.isfinite(frac) & (counts > 0)
    ax.errorbar(centres[ok], frac[ok], err[ok], fmt="o-", color=PETROFF10[0],
                capsize=2, lw=1.5, markersize=5, label="All runs")

    # Mark the 1:1 island. NOT drawn as a separate "co-orbital excluded" curve: delta and
    # period ratio are degenerate here. For these masses the mutual Hill radius is ~0.02 a,
    # so delta < 1.5 implies a period ratio below 1.05 -- cutting on period ratio would
    # simply delete the low-delta bins rather than control for anything.
    ratio = d["ratio"][good]
    co = ratio < CO_ORBITAL
    edge_co = delta[co].max() if co.any() else np.nan
    ax.axvspan(DELTA_MIN, edge_co, color=PETROFF10[4], alpha=0.10, zorder=0)
    ax.text(DELTA_MIN * 1.25, 3e-6, r"1:1 co-orbital", color=PETROFF10[4], fontsize=14)

    ax.axvline(HILL_STABILITY, color=PETROFF10[2], ls="--", lw=1.5)
    ax.text(HILL_STABILITY * 1.08, ax.get_ylim()[1] * 0.55,
            r"$2\sqrt{3}$", color=PETROFF10[2], fontsize=15)

    below = np.mean(delta < HILL_STABILITY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Separation from nearest known planet [mutual Hill radii]")
    ax.set_ylabel(r"Survival fraction")
    ax.set_xlim(DELTA_MIN, DELTA_MAX)
    ax.set_title(rf"{args.mode.replace('_', chr(92) + '_')}, "
                 rf"{100*below:.0f}\% of draws start inside $2\sqrt{{3}}$", fontsize=16)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = prior.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=14, loc="upper left")

    fig.tight_layout()
    out = HERE / "figs" / f"hill_separation_survival_{args.mode}.png"
    fig.savefig(out, dpi=500)
    print(f"Saved figure: {out.relative_to(ROOT)}")
    print(f"  n = {delta.size} runs, {int(alive.sum())} survivors")
    print(f"  median separation: all {np.median(delta):.2f}, "
          f"survivors {np.median(delta[alive]):.2f} mutual Hill radii")
    print(f"  fraction of draws inside 2sqrt(3): {below:.1%}")


if __name__ == "__main__":
    main()
