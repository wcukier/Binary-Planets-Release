# Binary Planets

Simulation and analysis code accompanying Cukier & Daylan (submitted). Injects single and binary planets into compact multi-planet systems and measures dynamical survival rates.

## Requirements

```bash
pip install numpy matplotlib scipy tqdm cmasher h5py rebound
```

The `trace` integrator requires rebound ≥ 4.4. The plotting scripts also require a LaTeX installation for rendered labels (`text.usetex = True`).

## Getting the data

There are two ways to obtain the simulation data needed for the plots.

### Option 1 — Download the data from Cukier & Daylan (submitted)

Run the download script from the repository root:

```bash
python download_data.py
```

This downloads and extracts the collated simulation results into:

```
output/bin_inj/<planet_name>/collated_results.npz
output/sin_inj/<planet_name>/collated_results.npz
```

The zip files are deleted automatically after extraction.

### Option 2 — Run the simulations yourself

The simulations in the paper were run with `run_one.py`, which runs a single simulation per invocation. The paper uses runs 0–2999 for single injection and 0–5999 for binary injection.

```bash
python run_one.py --mode <mode> --system <planet_name> --run <i>
```

| Argument | Description |
|---|---|
| `--mode` | `sin_inj` = single planet injection, `bin_inj` = binary planet injection |
| `--system` | System name as in `data/run_plan.json` (spaces may be written as `_`) |
| `--run` | Run index; sets the random seed, so each run is reproducible |
| `--outdir` | (optional) root for the run directory, default `.` |
| `--result-npz` | (optional) output file collecting the run's results, default `result.npz` |

The timestep, integrator, and simulation length come from `data/run_plan.json`: the `trace` integrator, a timestep of 5% of the period of the innermost planet, and 1,000,000 years. Run `python run_one.py --help` for the checkpointing options.

Alternatively, run `run_planets.py` once per planetary system per mode. The `output/` directory and run-name subdirectory are created automatically.

```bash
python run_planets.py <run_name> <mode> <planet_index> [dt] [t_end]
```

| Argument | Description |
|---|---|
| `run_name` | Output subdirectory under `output/` (system name is appended automatically) |
| `mode` | `0` = no injection, `1` = single planet injection, `2` = binary planet injection |
| `planet_index` | 1-based index into `data/compact_systems_run_composite.npy` |
| `dt` | (optional) integrator timestep in years, default `1e-5` (`leapfrog`); `auto` takes the timestep and integrator from `data/run_plan.json` |
| `t_end` | (optional) simulation end time in years, default `1,000,000` |

With `dt = auto`, `planet_index` is instead the 1-based position in `data/run_plan.json`, which orders the systems by estimated cost. Each invocation launches 1000 simulations in parallel using `multiprocessing.Pool`. Results are written to `output/<run_name>/<planet_name>/<i>/`.

Once all systems have been simulated, collate the per-run outputs into `collated_results.npz` files:

```bash
python collate.py <run_name>                                 # run_planets.py output
python collate.py <run_name> --source npz --root <dir>       # run_one.py output
```

For `--source npz`, the result files are read from `<dir>/<run_name>/<planet_name>/run_<i>.npz`. `collate.py` skips any planet whose `collated_results.npz` is already newer than all its runs.

#### Priors

Priors are defined in two places.

**Existing planet properties — [`binary_planets/utils.py`](binary_planets/utils.py)**

These draw orbital parameters for the known planets in each system from their catalog uncertainties. Where a catalog bound is missing, ±20% of the catalog value is used.

| Parameter | Prior |
|---|---|
| Stellar mass | Uniform over catalog uncertainty `[mid+lo, mid+hi]`; if missing, drawn from empirical distribution in `data/st_mass_dist.npy` |
| Planet semi-major axis | Uniform over catalog uncertainty `[mid+lo, mid+hi]` if published; otherwise the catalog value rescaled by `(M_star / M_star,catalog)^(1/3)` for the drawn stellar mass |
| Planet eccentricity | Uniform over catalog uncertainty `[mid+lo, mid+hi]`; fixed at **0** if unknown |
| Planet inclination | Fixed at **0** (coplanar) |
| Planet mass | Uniform over catalog uncertainty if known; otherwise forecasted from radius using the Forecaster probabilistic mass–radius relation (Chen & Kipping 2017, ApJ 834 17; [arXiv:1603.08614](https://arxiv.org/abs/1603.08614)) |
| Planet radius | The radius used for the mass forecast, if any; otherwise uniform over catalog uncertainty |

Draws in which the known planets are out of order or on crossing orbits are rejected and redrawn. To change these, edit the corresponding `get_*` functions in [`binary_planets/utils.py`](binary_planets/utils.py).

**Injected planet/binary properties — [`binary_planets/draw.py`](binary_planets/draw.py)**

These are drawn fresh for each simulation run. All angles (ω, Ω, phase) are drawn from `Uniform(-π, π)`, and inclinations are drawn isotropically (`cos i ~ Uniform(-1, 1)`).

The **orbital gap** for each system is the largest dynamical gap among adjacent planet pairs, defined as the pair with the greatest ratio of outer to inner semi-major axis. `a_in` and `a_out` are the semi-major axes of the inner and outer bounding planets respectively, as drawn for that run. Injected objects are placed log-uniformly within this range.

**Mode 1 — single planet injection**:

| Parameter | Prior |
|---|---|
| Mass | `Uniform(0.38, 72)` Earth masses |
| Semi-major axis | `LogUniform(a_in, a_out)` AU |
| Eccentricity | `Uniform(0, 1)` |

**Mode 2 — binary planet injection**:

| Parameter | Prior |
|---|---|
| Total mass | `Uniform(0.38, 72)` Earth masses |
| Mass ratio q (= m₁/m_tot) | `Uniform(0, 0.5)` |
| COM semi-major axis | `LogUniform(a_in, a_out)` AU |
| Binary separation d | `Uniform(0, r_Hill)` (within the Hill radius) |
| Binary eccentricity | `Uniform(0, 1)` |
| COM eccentricity | `Uniform(0, e_max)`, with `e_max = min(a_out/a_in - 1, 1 - a_in/a_out)` |

Injected orbits that cross a known planet's orbit are redrawn, and any run that goes unstable within the first 5 years is redrawn. To change a prior, edit the corresponding `np.random.*` call in [`binary_planets/draw.py`](binary_planets/draw.py).

## Making the plots

Plotting scripts can be run from either the repository root or the `plotting/` directory. Figures are saved to `plotting/figs/`.

```bash
python plotting/survival_rates.py      # figures 3, 4, 7
python plotting/single_inj_corner.py   # figure 5
python plotting/binary_inj_corner.py   # figure 6
```

Each script automatically uses the intersection of planets present in both `output/sin_inj/` and `output/bin_inj/`.
