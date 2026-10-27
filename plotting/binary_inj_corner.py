import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

import matplotlib as mpl

import os
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

sys.path.append(str(ROOT))

from binary_planets.sim import get_hill_radius  # noqa: E402
from binary_planets.utils import mutual_inclination  # noqa: E402
from binary_planets.constants import AU_TO_M, RADIUS_EARTH  # noqa: E402
import cmasher as cmr  # noqa: E402, F401
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.append(str(HERE))
from crossing_cap import e_max  # noqa: E402

os.makedirs(HERE / "figs", exist_ok=True)

def gen_arr(config, bin=True, n=12000, secondary=True):
    n_secondary = config["n_secondary"]
    l = bin*len(config["binary"].keys()) + secondary*n_secondary*len(config["secondary_0"].keys()) + 2
    arr = np.zeros((n, l))*np.nan
    keys = []
    keys.append("stable")
    keys.append("m_star")

    if secondary:
        for i in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                keys.append(f"{i}/{key}")
    if bin:
        for key in config["binary"].keys():
            keys.append(f"bin/{key}")
    
    
    return arr, keys




def add_to_array(arr, config, keys, i, stable, bin=True, n=12000, secondary=True):
    if not np.any(arr):
        arr, keys = gen_arr(config, bin=bin, n=n, secondary=secondary)
        
    n_secondary = config["n_secondary"]
    arr[i, 0] = stable

    arr[i, 1] = config["m_star"]
    j = 2
    
    if secondary:
        for k in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                arr[i, j] = config[f"secondary_{k}"][key]
                j += 1
    if bin:
        for key in config["binary"].keys():
            arr[i, j] = config["binary"][key]
            j += 1
    
    
    return arr, keys
    


run_systems = np.load(ROOT / "data" / "compact_systems_run_composite.npy", allow_pickle=True)
all_names = [sys["name"] for sys in run_systems if sys["name"]]

# BP_OUTPUT_ROOT lets this run against a pass other than output/ (e.g. the 2026-09 trace
# grid) without forking the script.  BP_REQUIRE_SIN=0 drops the requirement that a system
# also have single-injection output -- the binary corner plot never reads sin_inj, and the
# trace grid has no single-injection data at all.
OUTPUT_ROOT = pathlib.Path(os.environ.get("BP_OUTPUT_ROOT", ROOT / "output"))
FIG_SUFFIX = os.environ.get("BP_FIG_SUFFIX", "")

bin_planets = {d.name for d in (OUTPUT_ROOT / "bin_inj").iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
if os.environ.get("BP_REQUIRE_SIN", "1") == "1":
    sin_planets = {d.name for d in (OUTPUT_ROOT / "sin_inj").iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
    valid_planets = bin_planets & sin_planets
else:
    valid_planets = bin_planets
names = [name for name in all_names if name in valid_planets]
# names = ["Kepler-186"]

def transform(arr, keys, system_idx):
    """Convert one system's drawn parameters to the plotted ones.

    Both ``arr`` and ``keys`` are modified in place: masses become total mass and mass
    ratio, the pair inclination becomes the mutual inclination, angles go to degrees,
    d to Hill radii and the COM a is rescaled into the gap.
    """
    m_tot = arr[:, keys=="bin/m1"] + arr[:, keys=="bin/m2"]
    q = arr[:, keys=="bin/m1"] / m_tot
    arr[:, keys=="bin/m1"] = m_tot
    arr[:, keys=="bin/m2"] = q
    keys[keys=="bin/m1"] = "bin/m_tot"
    keys[keys=="bin/m2"] = "bin/q"

    # The pair's own inclination only matters relative to its COM orbit, so plot the
    # mutual inclination (the LK angle) in its place.
    arr[:, keys=="bin/inc"] = mutual_inclination(arr[:, keys=="bin/inc"],
                                                 arr[:, keys=="bin/bin_inc"],
                                                 arr[:, keys=="bin/Omega"])
    keys[keys=="bin/inc"] = "bin/i_mut"

    for _k in ("bin/i_mut", "bin/bin_inc"):
        arr[:, keys == _k] = np.degrees(arr[:, keys == _k])

    hill_radius = get_hill_radius(arr[:, keys=="bin/a"], 
                                  arr[:, keys=="bin/e_sys"], 
                                  arr[:, keys=="bin/m_tot"], 
                                  arr[:, keys=="m_star"])
    arr[:, keys=="bin/d"] /= hill_radius

    arr[:, keys=="bin/a"] = ((np.log10(arr[:, keys=="bin/a"]) - np.log10(run_systems[system_idx]["gap"][0]))
                             / (np.log10(run_systems[system_idx]["gap"][1]) - np.log10(run_systems[system_idx]["gap"][0])))


def load_prior_sample(system, keys, system_idx):
    """Weighted draws from the prior before the 5-yr rejection (plotting/prerejection_prior.py).

    ``keys`` are the raw (untransformed) column names.  Returns ``(arr, weight)`` with the columns of ``keys`` (NaN where not stored), or
    None if the system has not been sampled.  Weights sum to the grid's draw count.
    """
    path = OUTPUT_ROOT / "prerejection" / dir / f"{system}.npz"
    if not path.exists():
        return None
    smp = np.load(path)
    out = np.full((len(smp["weight"]), len(keys)), np.nan)
    for k in smp.files:
        if k in keys:
            out[:, keys==k] = smp[k][:, None]
    transform(out, keys.copy(), system_idx)
    return out, smp["weight"]


def _uniform(lo, hi):
    return lambda b: np.diff(np.clip(b, lo, hi)) / (hi - lo)


def _isotropic_deg(b):
    return -np.diff(np.cos(np.radians(np.clip(b, 0, 180)))) / 2


# Fraction of the pre-rejection prior per bin, for the parameters drawn before the
# crossing-orbit loop (draw.py at the grid revision).  The rest come from the sample.
ANALYTIC = {"bin/m_tot": _uniform(0.38, 72), "bin/q": _uniform(0, 0.5), "bin/e": _uniform(0, 1),
            "bin/e_sys_rel": _uniform(0, 1),
            "bin/i_mut": _isotropic_deg, "bin/bin_inc": _isotropic_deg}


is_binary = True

dir = "bin_inj"
arr_t = None
stable_t = None
prior_samples = []
stable_omega = []
unstable_omega = []
for system in tqdm(names):
    cached_data =  np.load(OUTPUT_ROOT / dir / system / "collated_results.npz", allow_pickle=True)
    cached_summaries = cached_data["summaries"]
    cached_cfgs = cached_data["cfgs"]
    cached_elements = cached_data["elements"]
    cached_distances = cached_data["distances"]
    n = len(cached_summaries)
    
    try:
        arr = None
        keys = None
        system_idx = np.where(np.array(all_names) == system)[0][0]


        j = 0
        stable = np.zeros(n)

        for i in range(0, n):
            try:
#                 with open(f"../output/{dir}/{system}/{i}/config.json") as f:
#                     cfg = json.load(f)
#                     elements = np.load(f"../output/{dir}/{system}/{i}/elements.npy")
#                     distances = np.load(f"../output/{dir}/{system}/{i}/distances.npy")
                cfg = cached_cfgs[i]
                elements = cached_elements[i]
                distances = cached_distances[i]
    
                a_i = elements[0, :, 0]
                a_f = elements[-1, :, 0]

                stable[i] = np.all(((a_i-a_f)/a_i) < 1e100)

                if is_binary:
                    bin_e = cfg["binary"]["e"]
                    bin_a = a_i[1]
                    max_d = 1.1 * bin_a * (1 + bin_e)
                    if np.any(distances > max_d):
                        stable[i] = False


                # stable[i] = cfg["secondary_0"]["omega"] < 0

                b = cfg["binary"]
                e_contact = 1 - (b["radius1"] + b["radius2"]) * RADIUS_EARTH / AU_TO_M / b["d"]
                cfg = dict(cfg, binary=dict(b, e_sys_rel=b["e_sys"] / e_max(cfg, 2),
                                            e_rel=b["e"] / e_contact))
                arr, keys = add_to_array(arr, cfg, keys, j, stable[i], bin=is_binary, secondary=False)


#                 print(stable[i])
                if stable[i]:
                    stable_omega.append(cfg["secondary_0"]["omega"])
                else:
                    unstable_omega.append(cfg["secondary_0"]["omega"])
                j += 1

            except Exception as e:
                print(e)
                pass
        arr = arr[:j, :]
        stable = stable[:j]
        stable = np.logical_not(np.logical_not(stable))
        keys = np.array(keys)
        arr[:, 1:] += np.random.random((arr.shape[0], arr.shape[1]-1)) * 1e-20


        smp = load_prior_sample(system, keys, system_idx)
        transform(arr, keys, system_idx)

        if not np.any(arr_t):
            arr_t = arr
            stable_t = stable
        else:
            arr_t = np.vstack([arr_t, arr])
            stable_t = np.hstack([stable_t, stable])
        prior_samples.append(smp)
    except Exception as e:
        print(e)


if any(r is None for r in prior_samples):
    print("prerejection sample missing for some systems; omitting the pre-rejection prior")
    arr_pre_t = None
else:
    arr_pre_t = np.vstack([r[0] for r in prior_samples])
    w_pre_t = np.concatenate([r[1] for r in prior_samples])

param_labels = ["Total Mass \n" r"[M$_\oplus$]", "Mass Ratio", "Binary a \n[Hill Radii]",
                "Binary\n" r"$e/e_{\rm contact}$", "COM a \n(Rescaled)", r"COM $e/e_{\rm max}$",
                "Mutual \nInclination\n [deg]", "COM \nInclination \n [deg]"]

# 37 of the 396,000 runs start with e above e_contact (all collide); keep them off the axis.
PLOT_RANGE = {"bin/e_rel": (0, 1)}

good_keys = ['bin/m_tot', 'bin/q', 'bin/d', 'bin/e_rel', 'bin/a', 'bin/e_sys_rel',  'bin/i_mut', 'bin/bin_inc']

# plt.style.use("dark_background")
mpl.rc("font", size=16)


n_vars = 7

fig, axs = plt.subplots(ncols=n_vars+1, nrows=n_vars+1, figsize=(15, 15))
gs = axs[0, 0].get_gridspec()
n = 50

levels = np.logspace(-2, 0.5, 8)
density_levels = np.linspace(0, 1, 9)

c1 = mpl.colormaps["cmr.sepia"](0.95)
c2 = mpl.colormaps["cmr.lavender"](0.95)

c1_dark = mpl.colormaps["cmr.sepia"](0.6)
c2_dark = mpl.colormaps["cmr.lavender"](0.6)


pairs = [(i, j) for i in range(n_vars+1) for j in range(i+1, n_vars+1)]

for i, j in tqdm(pairs):
        ax = axs[j,i]
        key1 = good_keys[i]
        key2 = good_keys[j]
        
        x = arr_t[arr_t[:, 0]>-1, keys==key1]
        y = arr_t[arr_t[:, 0]>-1, keys==key2]

        x_s = arr_t[arr_t[:, 0]==1, keys==key1]
        y_s = arr_t[arr_t[:, 0]==1, keys==key2]
        
        xs = np.linspace(*PLOT_RANGE.get(key1, (np.min(x), np.max(x))), n)
        ys = np.linspace(*PLOT_RANGE.get(key2, (np.min(y), np.max(y))), n)

        xx, yy = np.meshgrid(xs, ys)
        
        # --- replaced this ---
        # kde_x = gaussian_kde(x)
        # kde_y = gaussian_kde(y)
        # kde_x_s = gaussian_kde(x_s)
        # kde_y_s = gaussian_kde(y_s)
        # density = np.matmul(kde_y(ys).reshape((n,1)), kde_x(xs).reshape((1,n)))
        # density_s = np.matmul(kde_y_s(ys).reshape((n,1)), kde_x_s(xs).reshape((1,n)))

        # --- with this ---
        kde = gaussian_kde(np.vstack([x.ravel(), y.ravel()]))
        # kde_s = gaussian_kde(np.vstack([x_s.ravel(), y_s.ravel()]))

        xy = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(xy).reshape(n, n)
        density /= density.max()
        # density_s = kde_s(xy).reshape(n, n)
        # -------------------

        cs = ax.contourf(xx, yy, density, levels=density_levels, cmap="cmr.sepia", alpha=0.5)
#         ax.contour(xx, yy, density_s, linewidths=2, cmap="cmr.lavender", levels=3)
#         ax.scatter(x_s, y_s, s=0.25, c="k", alpha=0.3)
        ax.scatter(x_s, y_s, s=1.2, c="k", alpha=0.3, linewidths=0)


#         ax.set_xticks([])
#         ax.set_yticks([])
        ax.sharex(axs[n_vars, i])
        if (j != n_vars): ax.tick_params('x', labelbottom=False)
        else: ax.set_xlabel(param_labels[i])

        
        ax.sharey(axs[j, 0])
        if (i != 0): ax.tick_params('y', labelleft=False)
        else: ax.set_ylabel(param_labels[j])
        ax.set_facecolor("white")

for i in range(n_vars+1):
    for j in range(i+1, n_vars+1):
        axs[i,j].remove()
        
        
y_top = 0
for i in range(n_vars+1):
    ax = axs[i,i]
    key1 = good_keys[i]
    x = arr_t[arr_t[:, 0]>-1, keys==key1]
    x_s = arr_t[arr_t[:, 0]==1, keys==key1]

    n_bins=25
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Count")
    
    if arr_pre_t is None:
        bins = n_bins
    else:
        x_pre = arr_pre_t[:, keys==key1].ravel()
        x_pre = x_pre[np.isfinite(x_pre)]
        bins = np.histogram_bin_edges(np.concatenate([x.ravel(), x_pre]), n_bins)
    if key1 in PLOT_RANGE:
        bins = np.linspace(*PLOT_RANGE[key1], n_bins + 1)

    n_prior = ax.hist(x, bins=bins, color=c1)[0]
    y_top = max(y_top, n_prior.max())
    ax.hist(x, bins=bins, color=c1_dark, histtype="step")

    bins_s = bins if key1 in PLOT_RANGE else n_bins
    ax.hist(x_s, bins=bins_s, color=c2)
    ax.hist(x_s, bins=bins_s, color=c2_dark, histtype="step", lw=2)

    if arr_pre_t is not None:
        if key1 in ANALYTIC:
            counts = w_pre_t.sum() * ANALYTIC[key1](bins)
        else:
            counts = np.histogram(x_pre, bins, weights=w_pre_t)[0]
        ax.stairs(counts, bins, color=c1_dark, ls=":", lw=2)
        y_top = max(y_top, counts.max())

    ax.set_yscale("log")

    ax.sharey(axs[0,0])
    ax.sharex(axs[n_vars, i])
    if (i != n_vars): ax.tick_params('x', labelbottom=False)
    else: ax.set_xlabel(param_labels[i])

# plt.show()
# The diagonals share y with axs[0, 0]; size it to the tallest histogram.
axs[0, 0].set_ylim(top=2 * y_top)

for i in range(n_vars+1):
    axs[n_vars, i].tick_params(axis='x', rotation=45)
    axs[i, i].tick_params(axis='x', rotation=45)

for idx, k in enumerate(good_keys):
    if k in PLOT_RANGE:
        axs[n_vars, idx].set_xlim(PLOT_RANGE[k])
        if idx > 0:
            axs[idx, 0].set_ylim(PLOT_RANGE[k])

fig.subplots_adjust(hspace=0.05, wspace=0.05)  # a) reduce spacing

lax = fig.add_subplot(gs[0:2, n_vars-1:])
lax.axis("off")
handles = [mpl.patches.Patch(facecolor=c1, edgecolor=c1_dark, label="All Runs (Prior)"),
           mpl.patches.Patch(facecolor=c2, edgecolor=c2_dark, lw=2, label="Surviving Runs")]
if arr_pre_t is not None:
    handles.append(mpl.lines.Line2D([], [], color=c1_dark, ls=":", lw=2,
                                    label="Prior before 5 yr rejection"))
handles.append(mpl.lines.Line2D([], [], ls="", marker="o", ms=5, mfc="k", mec="none",
                                alpha=0.6, label="Surviving Run"))
leg = lax.legend(handles=handles, loc="upper right", frameon=False)
fig.canvas.draw()
bb = leg.get_window_extent().transformed(lax.transAxes.inverted())
cb_width = 0.75
cax = lax.inset_axes([(bb.x0 + bb.x1 - cb_width) / 2, 0.4, cb_width, 0.05])
cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.set_label("Prior Density")


plt.savefig(HERE / "figs" / f"figure6_corner_binary{FIG_SUFFIX}.png", dpi=150,
            bbox_inches="tight")