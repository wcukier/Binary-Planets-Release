import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

from scipy.stats import gaussian_kde

import matplotlib as mpl

import os
import pathlib

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

sys.path.append(str(ROOT))

import cmasher as cmr  # noqa: E402

sys.path.append(str(HERE))
from crossing_cap import e_max  # noqa: E402

os.makedirs(HERE / "figs", exist_ok=True)

run_name = sys.argv[1] if len(sys.argv) > 1 else "sin_inj"

# Same overrides as binary_inj_corner.py.
OUTPUT_ROOT = pathlib.Path(os.environ.get("BP_OUTPUT_ROOT", ROOT / "output"))
FIG_SUFFIX = os.environ.get("BP_FIG_SUFFIX", "")

def gen_arr(config, bin=True, n=12000, secondary=True, extra_keys = []):
    n_secondary = config["n_secondary"]
    
    if secondary == "sin_inj":
        sec = 1
    elif secondary:
        sec = n_secondary
    
    l = bin*len(config["binary"].keys()) + sec*len(config["secondary_0"].keys()) + 2 + len(extra_keys)
    arr = np.zeros((n, l))*np.nan
    keys = []
    keys.append("stable")
    keys.append("m_star")

    if secondary == "sin_inj":
        for key in config["secondary_0"].keys():
            keys.append(f"{0}/{key}")
    elif secondary:
        for i in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                keys.append(f"{i}/{key}")
    if bin:
        for key in config["binary"].keys():
            keys.append(f"bin/{key}")
    
    for key in extra_keys:
        keys.append(key)
    
    return arr, keys




def add_to_array(arr, config, keys, i, stable, bin=True, n=12000, secondary=True, extra_keys=[], extra_vars=[]):
    if not np.any(arr):
        arr, keys = gen_arr(config, bin=bin, n=n, secondary=secondary, extra_keys=extra_keys)
        
    n_secondary = config["n_secondary"]
    arr[i, 0] = stable

    arr[i, 1] = config["m_star"]
    
    inj = n_secondary-1
    
    j = 2
    if secondary == "sin_inj":
        for key in config["secondary_0"].keys():
            arr[i, j] = config[f"secondary_{inj}"][key]
            j += 1
    elif secondary:
        for k in range(0, n_secondary):
            for key in config["secondary_0"].keys():
                arr[i, j] = config[f"secondary_{k}"][key]
                j += 1
    if bin:
        for key in config["binary"].keys():
            arr[i, j] = config["binary"][key]
            j += 1
    for k in range(len(extra_keys)):
        arr[i, j] = extra_vars[k]
        j += 1
    
    return arr, keys


def rescale(arr, keys, system_idx):
    """Rescale a into the gap and inclination to degrees, in place."""
    gap = np.log10(run_systems[system_idx]["gap"])
    for k in ("0/a", "a_f"):
        arr[:, keys==k] = (np.log10(arr[:, keys==k]) - gap[0]) / (gap[1] - gap[0])

    # Degrees, to match the text and binary_inj_corner.py.
    arr[:, keys=="0/inc"] = np.degrees(arr[:, keys=="0/inc"])


def load_prior_sample(system, keys, system_idx):
    """Weighted draws from the prior before the 5-yr rejection (plotting/prerejection_prior.py).

    Returns ``(arr, weight)`` with the columns of ``keys`` (NaN where not stored), or
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
    rescale(out, keys, system_idx)
    return out, smp["weight"]


def _uniform(lo, hi):
    return lambda b: np.diff(np.clip(b, lo, hi)) / (hi - lo)


def _isotropic_deg(b):
    return -np.diff(np.cos(np.radians(np.clip(b, 0, 180)))) / 2


# Fraction of the pre-rejection prior per bin, for the parameters drawn before the
# crossing-orbit loop (draw.py at the grid revision).  The rest come from the sample.
ANALYTIC = {"0/m": _uniform(0.38, 72), "0/inc": _isotropic_deg, "0/e_rel": _uniform(0, 1)}

    
run_systems = np.load(ROOT / "data" / "compact_systems_run_composite.npy", allow_pickle=True)
all_names = [sys["name"] for sys in run_systems if sys["name"]]

bin_planets = {d.name for d in (OUTPUT_ROOT / "bin_inj").iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
sin_planets = {d.name for d in (OUTPUT_ROOT / run_name).iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
valid_planets = bin_planets & sin_planets
names = [name for name in all_names if name in valid_planets]
# names = ["Kepler-186"]


is_binary = False

dir = run_name

extra_keys = ["a_f", "max_change", "0/e_rel"]

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
    n = len(cached_summaries)
    
    arr = None
    keys = None
    system_idx = np.where(np.array(all_names) == system)[0][0]


    j = 0
    stable = np.zeros(n)

    for i in range(0, n):
        cfg = cached_cfgs[i]
        elements = cached_elements[i]

        n_inj = cfg["n_secondary"] - 1

        a_i = elements[0, :, 0]
        a_f = elements[-1, :, 0]

        stable[i] = np.all(((a_i-a_f)/a_i) < 1e100)
        


        arr, keys = add_to_array(arr, 
                                    cfg, 
                                    keys,
                                    j, 
                                    stable[i], 
                                    bin=is_binary, 
                                    secondary="sin_inj",
                                    extra_keys=extra_keys,
                                    extra_vars=[elements[-1, n_inj, 0], np.nanmax((a_i-a_f)/a_i),
                                                cfg[f"secondary_{n_inj}"]["e"] / e_max(cfg, 1)]
                                )

        if stable[i]:
            stable_omega.append(cfg["secondary_0"]["omega"])
        else:
            unstable_omega.append(cfg["secondary_0"]["omega"])
        j += 1

    arr = arr[:j, :]
    stable = stable[:j]
    stable = np.logical_not(np.logical_not(stable))
    keys = np.array(keys)
    arr[:, 1:] += np.random.random((arr.shape[0], arr.shape[1]-1)) * 1e-20

    rescale(arr, keys, system_idx)
    prior_samples.append(load_prior_sample(system, keys, system_idx))


    if not np.any(arr_t):
        arr_t = arr
        stable_t = stable
    else:
        arr_t = np.vstack([arr_t, arr])
        stable_t = np.hstack([stable_t, stable])


if any(r is None for r in prior_samples):
    print("prerejection sample missing for some systems; omitting the pre-rejection prior")
    arr_pre_t = None
else:
    arr_pre_t = np.vstack([r[0] for r in prior_samples])
    w_pre_t = np.concatenate([r[1] for r in prior_samples])

param_labels = [r"Mass [M$_\oplus$]", "a (Rescaled)", r"$e/e_{\rm max}$", "Inclination [deg]"]

mpl.rc("font", size=10)
# plt.style.use("dark_background")

n_vars = 3

fig, axs = plt.subplots(ncols=n_vars+1, nrows=n_vars+1, figsize=(7.5, 7.5))
gs = axs[0, 0].get_gridspec()
n = 50

levels = np.logspace(-2, 0.5, 8)
density_levels = np.linspace(0, 1, 9)

c1 = mpl.colormaps["cmr.sepia"](0.95)
c2 = mpl.colormaps["cmr.lavender"](0.95)

c1_dark = mpl.colormaps["cmr.sepia"](0.6)
c2_dark = mpl.colormaps["cmr.lavender"](0.6)

pairs = [(i, j) for i in range(n_vars+1) for j in range(i+1, n_vars+1)]
good_keys = ['0/m', '0/a', '0/e_rel', '0/inc', "a_f", "max_change"]



for i, j in tqdm(pairs):
        ax = axs[j,i]
        key1 = good_keys[i]
        key2 = good_keys[j]
        
        x = arr_t[arr_t[:, 0]>-1, keys==key1]
        y = arr_t[arr_t[:, 0]>-1, keys==key2]

        x_s = arr_t[arr_t[:, 0]==1, keys==key1]
        y_s = arr_t[arr_t[:, 0]==1, keys==key2]
        
        xs = np.linspace(np.min(x), np.max(x), n)
        ys = np.linspace(np.min(y), np.max(y), n)

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
        # ax.contour(xx, yy, density_s, linewidths=2, cmap="cmr.lavender", levels=3)

        ax.scatter(x_s, y_s, s=0.84, c="k", alpha=0.3, linewidths=0)
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
        ax = axs[i,j]
        ax.axis("off")
        if [i,j] == [0,n_vars]:
            handles = [mpl.patches.Patch(facecolor=c1, edgecolor=c1_dark,
                                         label="All Runs (Prior)"),
                       mpl.patches.Patch(facecolor=c2, edgecolor=c2_dark, lw=2,
                                         label="Surviving Runs")]
            if arr_pre_t is not None:
                handles.append(mpl.lines.Line2D([], [], color=c1_dark, ls=":", lw=1.5,
                                                label="Prior before 5 yr rejection"))
            handles.append(mpl.lines.Line2D([], [], ls="", marker="o", ms=4.5, mfc="k", mec="none",
                                            alpha=0.6, label="Surviving Run"))
            leg = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
            fig.canvas.draw()
            bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
            cb_width = 1.0
            cax = ax.inset_axes([(bb.x0 + bb.x1 - cb_width) / 2, 0.3, cb_width, 0.08])
            cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
            cb.set_label("Prior Density", fontsize=9)
        
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

    n_prior = ax.hist(x, bins=bins, color=c1)[0]
    y_top = max(y_top, n_prior.max())
    ax.hist(x, bins=bins, color=c1_dark, histtype="step")

    ax.hist(x_s, bins=n_bins, color=c2)
    ax.hist(x_s, bins=n_bins, color=c2_dark, histtype="step", lw=2)

    if arr_pre_t is not None:
        if key1 in ANALYTIC:
            counts = w_pre_t.sum() * ANALYTIC[key1](bins)
        else:
            counts = np.histogram(x_pre, bins, weights=w_pre_t)[0]
        ax.stairs(counts, bins, color=c1_dark, ls=":", lw=1.5)
        y_top = max(y_top, counts.max())

    ax.set_yscale("log")

    ax.sharey(axs[0,0])
    ax.sharex(axs[n_vars, i])
    if (i != n_vars): ax.tick_params('x', labelbottom=False)
    else: ax.set_xlabel(param_labels[i])

# The diagonals share y with axs[0, 0]; size it to the tallest histogram.
axs[0, 0].set_ylim(top=2 * y_top)

for i in range(n_vars+1):
    axs[n_vars, i].tick_params(axis='x', rotation=45)
    axs[i, i].tick_params(axis='x', rotation=45)



plt.savefig(HERE / "figs" / f"figure5_corner_single{FIG_SUFFIX}.png", dpi=500)
