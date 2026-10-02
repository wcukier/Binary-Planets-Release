import os
import sys
import warnings

import cmasher as cmr
import numpy as np
import pathlib
from matplotlib import pyplot as plt

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent

sys.path.append(str(ROOT))

from binary_planets.sim import get_hill_radius  # noqa: E402

os.makedirs(HERE / "figs", exist_ok=True)

warnings.formatwarning = lambda msg, *args, **kwargs: f"Warning: {msg}\n"

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Computer Modern Roman",
  "font.size":    20.0
})

n = 3000
run_systems = np.load(ROOT / "data" / "compact_systems_run_composite.npy", allow_pickle=True)

dirs = ["sin_inj", "bin_inj"]


bin_inj = ROOT / "output" / "bin_inj"
sin_inj = ROOT / "output" / "sin_inj"

bin_planets = {d.name for d in bin_inj.iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}
sin_planets = {d.name for d in sin_inj.iterdir() if d.is_dir() and (d / "collated_results.npz").exists()}

new_line = False

for name in bin_planets.symmetric_difference(sin_planets):
    which = "bin_inj only" if name in bin_planets else "sin_inj only"
    warnings.warn(f"{name} has collated_results.npz in {which} — excluded from systems list.")
    new_line = True

if new_line: print()

gap_ratios = {sys["name"]: sys["gap"][1]/sys["gap"][0] for sys in run_systems}

systems = sorted(bin_planets & sin_planets)

for name in systems:
    if name not in gap_ratios:
        raise KeyError(f"{name} has results but is missing from run_systems")

results = {}


# ============ Load Data ============

for dir in dirs:
    results[dir] = {}
    for system in systems:
        results[dir][system] = {}

        n_act = 0
        none_survived = False
        cached_data =  np.load(ROOT / "output" / dir / system / "collated_results.npz", allow_pickle=True)
        cached_summaries = cached_data["summaries"]
        cached_cfgs = cached_data["cfgs"]
        cached_elements = cached_data["elements"]
        cached_distances = cached_data["distances"]
        
        if not none_survived:
            for i in range(1, n):
                try:
                    summary = cached_summaries[i]
                    cfg = cached_cfgs[i]
                    elements = cached_elements[i]
                    
                    if n_act==0:
                        results[dir][system]["a_i"] = elements[0,:,0]
                        results[dir][system]["a_f"] = elements[-1,:,0]
                        if dir[:3] == "bin":
                            results[dir][system]["bin_m1"] = [cfg["binary"]["m1"]]
                            results[dir][system]["bin_m2"] = [cfg["binary"]["m2"]]
                            results[dir][system]["bin_esys"] = [cfg["binary"]["e_sys"]]
                            results[dir][system]["bin_e"] = [cfg["binary"]["e"]]
                            results[dir][system]["bin_d"] = [cfg["binary"]["d"]]
                            results[dir][system]["bin_a"] = [cfg["binary"]["a"]]
                        results[dir][system]["stable"] = []

                    else:
                        results[dir][system]["a_i"] = np.vstack(
                            (results[dir][system]["a_i"], elements[0,:,0]))
                        results[dir][system]["a_f"] = np.vstack(
                            (results[dir][system]["a_f"], elements[-1,:,0]))
                        if dir[:3] == "bin":
                            results[dir][system]["bin_m1"].append(cfg["binary"]["m1"])
                            results[dir][system]["bin_m2"].append(cfg["binary"]["m2"])
                            results[dir][system]["bin_esys"].append(cfg["binary"]["e_sys"])
                            results[dir][system]["bin_e"].append(cfg["binary"]["e"])
                            results[dir][system]["bin_d"].append(cfg["binary"]["d"])
                            results[dir][system]["bin_a"].append(cfg["binary"]["a"])

                    n_act += 1
                    a_i, a_f =  results[dir][system]["a_i"][-1],  results[dir][system]["a_f"][-1]

                    stable = np.all(((a_i-a_f)/a_i) < 1e100)

                    if "bin" in dir:
                        distances = cached_distances[i]
                        bin_e = cfg["binary"]["e"]
                        bin_a = a_i[1]
                        
                        hill_radius = get_hill_radius(bin_a, 
                              cfg["binary"]["e_sys"], 
                              cfg["binary"]["m1"] + cfg["binary"]["m2"], 
                              cfg["m_star"])
                                     
                        max_d = 1.1 * bin_a * (1 + bin_e)
                        max_d = 1.1 * hill_radius
                        if np.any(distances > max_d):
                            stable = False
                    
                    results[dir][system]["stable"].append(stable)
                    
                except Exception as e:
#                     print(e)
                    pass
            try:
                if dir[:3] == "bin":                
                    results[dir][system]["bin_m1"] = np.array(results[dir][system]["bin_m1"])
                    results[dir][system]["bin_m2"] = np.array(results[dir][system]["bin_m2"])
                    results[dir][system]["bin_esys"] = np.array(results[dir][system]["bin_esys"])
                    results[dir][system]["bin_e"] = np.array(results[dir][system]["bin_e"])
                    results[dir][system]["bin_d"] = np.array(results[dir][system]["bin_d"])
                    results[dir][system]["bin_a"] = np.array(results[dir][system]["bin_a"])
            except:
                pass
            try:
                a_i, a_f =  results[dir][system]["a_i"],  results[dir][system]["a_f"]
                
                norm = np.max(((a_i-a_f)/a_i), axis=1)
                
                results[dir][system]["norm"] = norm
                
                results[dir][system]["n_survived"] = np.sum(results[dir][system]["stable"])
                results[dir][system]["n_act"] = n_act
                results[dir][system]["survival_frac"] = results[dir][system]["n_survived"]/n_act
                print(f"{system} Survival Fraction: {results[dir][system]['survival_frac']} for {n_act} systems ({dir})")
            except Exception as e:
                print(f"Error processing {system}: {e}")
                results[dir][system]["n_survived"] = np.nan
                results[dir][system]["n_act"] = np.nan
                results[dir][system]["survival_frac"] = np.nan
                
        else:
                print(f"NONE SURVIVED in {system}")
                results[dir][system]["n_survived"] = 0
                results[dir][system]["n_act"] = int(n_act)
                results[dir][system]["survival_frac"] = 0
                
print("\n")

# ============ Small Gap vs Large Gap Histogram (Figure 3) ============
    
single_dir = dirs[0]
binary_dir = dirs[1]

single_inj = [results[single_dir][system]["survival_frac"] for system in systems]
binary_inj = [results[binary_dir][system]["survival_frac"] for system in systems]

single_inj_err = [np.sqrt(results[single_dir][system]["n_survived"])/results[single_dir][system]["n_act"] for system in systems]
binary_inj_err = [np.sqrt(results[binary_dir][system]["n_survived"])/results[binary_dir][system]["n_act"] for system in systems]


gaps = np.array([gap_ratios[system] for system in systems])
large_gap = gaps > 1.95

single_inj_hist = np.array(single_inj)
single_inj_hist[single_inj_hist < 1e-4] = 10**(-4.7)

cmap = cmr.seaweed

c1 = cmap(.9)
c2 = cmap(.1)

c1d = cmap(.75)
c2d = cmap(.25)

c1dd = cmap(.6)
c2dd = cmap(.4)

binary_inj_hist = np.array(binary_inj)
binary_inj_hist[binary_inj_hist < 1e-4] = 10**(-4.7)

colors = ["grey", "blue", "red"]

bins = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0]

plt.subplot(211)
plt.hist([ np.log10(single_inj_hist[~large_gap]), np.log10(single_inj_hist[large_gap])], histtype="barstacked", 
         label=["Small Gap Systems", "Large Gap Systems"],
        alpha=1, color=[c2, c2d], bins=bins, 
         )
plt.hist([np.log10(single_inj_hist)], histtype="step",
         color=c2dd, bins=bins, linewidth=2)

d = .05
plt.plot([-4-d/2, -4+d/2], [0,0], clip_on=False, c="white",  zorder=5)
plt.plot([-4-d, -4], [-20*d, +20*d], clip_on=False, c='k', lw=.5, zorder=6)
plt.plot([-4, -4+d], [-20*d, +20*d], clip_on=False, c='k', lw=.5, zorder=6)

plt.xticks([-4.75, -3.25, -2.25, -1.25, -0.25], [r"", r"", r"", r"", r""])
plt.xlim(-5.25, 1)
plt.ylim(0, 36)
plt.text(0.75, 29, "Single Injection", fontsize=16, ha="right")

plt.ylabel("Number of Systems", fontsize=16)

plt.legend(loc=2, fontsize=16)

plt.subplot(212)

plt.hist([np.log10(binary_inj_hist[~large_gap]), np.log10(binary_inj_hist[large_gap]),], histtype="barstacked", 
         label=["Small Gap Systems", "Large Gap Systems"],
        alpha=1, color=[c1, c1d], bins=bins)

plt.hist([np.log10(binary_inj_hist)], histtype="step",
         color=c1dd, bins=bins, linewidth=2)

plt.text(-5, 29, "Binary Injection", fontsize=16)

d = .05
plt.plot([-4-d/2, -4+d/2], [0,0], clip_on=False, c="white",  zorder=5)
plt.plot([-4-d, -4], [-20*d, +20*d], clip_on=False, c='k', lw=.5, zorder=6)
plt.plot([-4, -4+d], [-20*d, +20*d], clip_on=False, c='k', lw=.5, zorder=6)

plt.xticks([-4.75, -3.25, -2.25, -1.25, -0.25], [r"0\%",  r"0.1\%", r"1\%", r"10\%", r"100\%"])
plt.xlim(-5.25, 1)
plt.ylim(0, 36)
plt.legend(loc=1, fontsize=16)
plt.xlabel("Survival Fraction")
plt.ylabel("Number of Systems", fontsize=16)
plt.tight_layout()
plt.savefig(HERE / "figs" / "figure3_survival_comp.png")
print("Saved figure: figs/figure3_survival_comp.png")


# ============ Indv System Survival Fraction (Figure 7) ============

plt.subplots(figsize=(9,6.5))

cutoff = 0


single_inj = np.array(single_inj)
binary_inj = np.array(binary_inj)

single_inj_error = np.array([np.sqrt(results[single_dir][system]["n_survived"])
                             / results[single_dir][system]["n_act"]
                             for system in systems])

binary_inj_error = np.array([np.sqrt(results[binary_dir][system]["n_survived"])
                             / results[binary_dir][system]["n_act"]
                             for system in systems])


mask = ((binary_inj/(single_inj+1e-20) > 1) * ((single_inj > 1e-3) + (binary_inj > 1e-3))) + binary_inj > 0.01

systems=np.array(systems)
width = 0.33
xs = np.arange(len(binary_inj[mask]))
plt.plot(figsize=(30,10))


plt.bar(xs, single_inj[mask], width=width, label="Single Injection", color=c2d)
plt.errorbar(xs, single_inj[mask], single_inj_error[mask], c="black", fmt=".", lw=1, markersize=0, capsize=2)


plt.bar(xs+width, binary_inj[mask], width=width, label="Binary Injection", color=c1)

plt.errorbar(xs+width, binary_inj[mask], binary_inj_error[mask], c="black", fmt=".", lw=1, markersize=0, capsize=2)


labels = np.copy(systems)

for i in range(len(labels)):
    if large_gap[i]: labels[i] = labels[i] + "*"

plt.xticks(xs+width/2, labels[mask], rotation=(-60), ha="left", rotation_mode="anchor")
plt.ylabel("Survival Fraction")

plt.legend()
plt.text(15, 0.7e-4, "* Large Gap System")
plt.ylim(1e-3, 1)
plt.yscale("log")
plt.tight_layout()
plt.savefig(HERE / "figs" / "figure7_survival_frac.png", transparent=True, dpi=500)
print("Saved figure: figs/figure7_survival_frac.png")


# ================ Survival Frac by gap size (Fig 4) ========================

plt.subplots(figsize=(9,6.5))


for i in range(len(single_inj)):
    plt.plot([gaps[i], gaps[i]], [single_inj[i], binary_inj[i]], "k:", alpha = 0.5)

plt.scatter(gaps, single_inj, marker="s", s=16, zorder=4, label="Single Injection", color=c2d)
plt.errorbar(gaps, single_inj, single_inj_err, zorder=3, color=c2d, fmt=".", lw=1, markersize=0, capsize=2, alpha=0.9)


plt.scatter(gaps, binary_inj, marker="s", s=16, zorder=4, label="Binary Injection", color=c1)
plt.errorbar(gaps, binary_inj, binary_inj_err, zorder=3, color=c1, fmt=".", lw=1, markersize=0, capsize=2, alpha=0.9)


plt.xlabel("Gap Size")
plt.ylabel("Survival Fraction")
plt.legend()
plt.tight_layout()
plt.savefig(HERE / "figs" / "figure4_scatterplot.png", dpi=500)
print("Saved figure: figs/figure4_scatterplot.png")
