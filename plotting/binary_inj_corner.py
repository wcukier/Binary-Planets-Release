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
import cmasher as cmr  # noqa: E402, F401
from scipy.stats import gaussian_kde  # noqa: E402

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

is_binary = True

dir = "bin_inj"
arr_t = None
stable_t = None
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


        m_tot = arr[:, keys=="bin/m1"] + arr[:, keys=="bin/m2"]
        q = arr[:, keys=="bin/m1"] / m_tot
        arr[:, keys=="bin/m1"] = m_tot
        arr[:, keys=="bin/m2"] = q
        keys[keys=="bin/m1"] = "bin/m_tot"
        keys[keys=="bin/m2"] = "bin/q"

        # Angles are stored in radians but reported in degrees everywhere in the text and
        # by the referee (the 39.2-140.8 deg LK window, Nesvorny's 50-140 deg band), so the
        # axes are converted here rather than leaving the reader to translate.
        for _k in ("bin/inc", "bin/bin_inc"):
            arr[:, keys == _k] = np.degrees(arr[:, keys == _k])

        hill_radius = get_hill_radius(arr[:, keys=="bin/a"], 
                                      arr[:, keys=="bin/e_sys"], 
                                      arr[:, keys=="bin/m_tot"], 
                                      arr[:, keys=="m_star"])
        arr[:, keys=="bin/d"] /= hill_radius

        arr[:, keys=="bin/a"] = ((np.log10(arr[:, keys=="bin/a"]) - np.log10(run_systems[system_idx]["gap"][0]))
                                 / (np.log10(run_systems[system_idx]["gap"][1]) - np.log10(run_systems[system_idx]["gap"][0])))

        if not np.any(arr_t):
            arr_t = arr
            stable_t = stable
        else:
            arr_t = np.vstack([arr_t, arr])
            stable_t = np.hstack([stable_t, stable])
    except Exception as e:
        print(e)


param_labels = ["Total Mass \n" r"[M$_\oplus$]", "Mass Ratio", "Binary a \n[Hill Radii]",
                "Binary e", "COM a \n(Rescaled)", "COM e",
                "Binary \nInclination\n [deg]", "COM \nInclination \n [deg]"]

good_keys = ['bin/m_tot', 'bin/q', 'bin/d', 'bin/e', 'bin/a', 'bin/e_sys',  'bin/inc', 'bin/bin_inc']

# plt.style.use("dark_background")
mpl.rc("font", size=16)


n_vars = 7

fig, axs = plt.subplots(ncols=n_vars+1, nrows=n_vars+1, figsize=(15, 15))
gs = axs[0, 0].get_gridspec()
n = 50

levels = np.logspace(-2, 0.5, 8)

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
        # density_s = kde_s(xy).reshape(n, n)
        # -------------------

        ax.contourf(xx, yy, density, cmap="cmr.sepia", alpha=0.5)
#         ax.contour(xx, yy, density_s, linewidths=2, cmap="cmr.lavender", levels=3)
#         ax.scatter(x_s, y_s, s=0.25, c="k", alpha=0.3)
        ax.scatter(x_s, y_s, s=0.4, c="k", alpha=0.3)


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
        
        
for i in range(n_vars+1):
    ax = axs[i,i]
    key1 = good_keys[i]
    x = arr_t[arr_t[:, 0]>-1, keys==key1]
    x_s = arr_t[arr_t[:, 0]==1, keys==key1]

    n_bins=25
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Count")
    
    ax.hist(x, bins=n_bins, color=c1)
    ax.hist(x, bins=n_bins, color=c1_dark, histtype="step")

    ax.hist(x_s, bins=n_bins, color=c2)
    ax.hist(x_s, bins=n_bins, color=c2_dark, histtype="step", lw=2)

    ax.set_yscale("log")

    ax.sharey(axs[0,0])
    ax.sharex(axs[n_vars, i])
    if (i != n_vars): ax.tick_params('x', labelbottom=False)
    else: ax.set_xlabel(param_labels[i])

# plt.show()
for i in range(n_vars+1):
    axs[n_vars, i].tick_params(axis='x', rotation=45)
    axs[i, i].tick_params(axis='x', rotation=45)

fig.subplots_adjust(hspace=0.05, wspace=0.05)  # a) reduce spacing


plt.savefig(HERE / "figs" / f"figure6_corner_binary{FIG_SUFFIX}.png", dpi=150,
            bbox_inches="tight")