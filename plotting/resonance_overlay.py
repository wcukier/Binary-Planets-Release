# LLM-GENERATED — UNVERIFIED (tier: figure-pub)
# Remove this line only after completing the checks in Assistant/POLICY.md §2.
"""Do gap-edge survivors sit near mean-motion resonances? Revision-plan task 1.3b.

Two panels answering referee Major 2 (reviewer_report.md:44) directly.

**Top** -- survival fraction against the rescaled semi-major axis, the coordinate of
Figures 5 and 6, with the density of first- through fourth-order resonances in the same
coordinate underneath. This is the visual test of the referee's proposal: if the survivor
structure near the gap edges were driven by resonances, the survival curve would track the
resonance density. Note that the density rises toward *both* edges purely because the
period ratio approaches 1 there and the j:(j-1) sequence crowds together.

**Bottom** -- survival fraction against fractional distance to the nearest resonance, drawn
separately within quantile bins of the period ratio to the nearer bounding planet. Binning
this way is what makes the panel a test rather than a restatement of the confound:
proximity to a bounding planet sets both the survival rate and the local resonance
density, so the raw relation is steep and meaningless. A resonance effect appears as a
slope *within* a band of fixed colour.

The numbers behind both panels come from ``tools/physics/gap_structure.py``.

Usage
-----
    python plotting/resonance_overlay.py [--mode sin_inj|bin_inj]
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

from tools.physics.gap_structure import (  # noqa: E402
    MAX_ORDER,
    N_CONTROL_BINS,
    SYSTEMS,
    collect,
    rescale_a,
    resonance_distance,
    resonance_table,
    tightest_ratio,
)

os.makedirs(HERE / "figs", exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
    "font.size": 20.0,
})

# Petroff 10-colour cycle
PETROFF10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
             "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]

RESONANCE_J_CAP = 7   # matches the unsaturated cap in gap_structure.report_resonance_test;
                      # at j <= 20 most of the sampled range lies within 1% of a resonance
                      # and the coordinate carries almost no information
N_X_BINS = 25         # bins in the rescaled semi-major axis
N_DELTA_BINS = 6      # bins in distance to the nearest resonance


def survival_curve(value: np.ndarray, alive: np.ndarray, edges: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned survival fraction with binomial 1-sigma errors.

    Parameters
    ----------
    value : np.ndarray
        Quantity to bin on, shape ``(n_runs,)``.
    alive : np.ndarray
        Survival flag, shape ``(n_runs,)`` of bool.
    edges : np.ndarray
        Bin edges, shape ``(n_bins + 1,)``.

    Returns
    -------
    centres : np.ndarray
        Bin centres, shape ``(n_bins,)``.
    frac, err : np.ndarray
        Survival fraction and its 1-sigma binomial error, shape ``(n_bins,)``. Both are
        ``nan`` in empty bins.
    """
    idx = np.digitize(value, edges) - 1
    frac = np.full(edges.size - 1, np.nan)
    err = np.full(edges.size - 1, np.nan)
    for i in range(edges.size - 1):
        sel = idx == i
        n = int(sel.sum())
        if n:
            p = alive[sel].mean()
            frac[i], err[i] = p, np.sqrt(p * (1 - p) / n)
    return 0.5 * (edges[:-1] + edges[1:]), frac, err


def resonance_positions(j_cap: int = RESONANCE_J_CAP) -> np.ndarray:
    """Rescaled-axis positions of every resonance with a bounding planet, over all systems.

    Parameters
    ----------
    j_cap : int
        Largest ``j`` considered; see ``gap_structure.resonance_table``.

    Returns
    -------
    np.ndarray
        Rescaled semi-major axes, dimensionless, of every resonance falling inside a gap,
        pooled over all systems. Shape ``(n_resonances_in_gaps,)``.

    Notes
    -----
    A resonance at period ratio ``R`` with the inner bounding planet sits at
    ``a = a_in R^{2/3}``; with the outer bounding planet, at ``a = a_out R^{-2/3}``. Both
    families are included, and positions outside ``[0, 1]`` are dropped. Uses the catalog
    gap so the positions correspond to the published rescaling, not to any single run's
    redrawn axes.
    """
    ratios = resonance_table(MAX_ORDER, j_cap)[0]
    positions = []
    for system in np.load(SYSTEMS, allow_pickle=True):
        gap = system["gap"]
        for a in (gap[0] * ratios ** (2 / 3), gap[1] * ratios ** (-2 / 3)):
            x = rescale_a(a, gap)
            positions.append(x[(x >= 0.0) & (x <= 1.0)])
    return np.concatenate(positions)


def main() -> None:
    """Build the two-panel resonance figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="sin_inj", choices=["sin_inj", "bin_inj"])
    parser.add_argument("--root", type=pathlib.Path, default=None)
    args = parser.parse_args()

    d = collect(args.mode, args.root)
    alive = d["survived"]
    delta = resonance_distance(d, RESONANCE_J_CAP)
    ratio = tightest_ratio(d)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 11))

    # ---- top: survival and resonance density against position in the gap ----
    x_edges = np.linspace(0.0, 1.0, N_X_BINS + 1)
    centres, frac, err = survival_curve(d["x"], alive, x_edges)
    ax_top.errorbar(centres, frac, err, fmt="o-", color=PETROFF10[0], capsize=2, lw=1.5,
                    label="Survival fraction")
    ax_top.set_xlabel(r"Rescaled semi-major axis")
    ax_top.set_ylabel(r"Survival fraction")
    ax_top.set_xlim(0, 1)

    density = ax_top.twinx()
    density.hist(resonance_positions(), bins=x_edges, color=PETROFF10[3], alpha=0.35,
                 label=rf"Resonance density ($q \leq {MAX_ORDER}$, $j \leq {RESONANCE_J_CAP}$)")
    density.set_ylabel(r"Resonances per bin, all systems")
    density.set_zorder(ax_top.get_zorder() - 1)
    ax_top.patch.set_visible(False)

    handles = ax_top.get_legend_handles_labels()[0] + density.get_legend_handles_labels()[0]
    labels = ax_top.get_legend_handles_labels()[1] + density.get_legend_handles_labels()[1]
    ax_top.legend(handles, labels, fontsize=13, loc="lower center", framealpha=0.9)

    # ---- bottom: the controlled test ----
    good = np.isfinite(delta) & np.isfinite(ratio)
    q = np.quantile(ratio[good], np.linspace(0, 1, N_CONTROL_BINS + 1))
    q[-1] += 1e-9
    d_edges = np.quantile(delta[good], np.linspace(0, 1, N_DELTA_BINS + 1))
    d_edges[-1] += 1e-9

    for i in range(N_CONTROL_BINS):
        in_bin = good & (ratio >= q[i]) & (ratio < q[i + 1])
        centres, frac, err = survival_curve(delta[in_bin], alive[in_bin], d_edges)
        ax_bot.errorbar(centres, frac, err, fmt="o-", capsize=2, lw=1.5,
                        color=PETROFF10[i],
                        label=rf"$P$ ratio ${q[i]:.2f}$--${q[i+1]:.2f}$")

    ax_bot.set_xscale("log")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"$|\delta|$, fractional distance to the nearest resonance")
    ax_bot.set_ylabel(r"Survival fraction")
    # Drop the lower limit so the legend does not sit on the lowest-ratio curve.
    ax_bot.set_ylim(bottom=0.4 * ax_bot.get_ylim()[0])
    ax_bot.legend(fontsize=13, title=r"Ratio to nearer bounding planet", title_fontsize=13,
                  loc="lower left", framealpha=0.9)

    fig.tight_layout()
    out = HERE / "figs" / f"resonance_overlay_{args.mode}.png"
    fig.savefig(out, dpi=500)
    print(f"Saved figure: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
