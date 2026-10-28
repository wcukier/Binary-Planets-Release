"""
Downloads collated npz zips from Zenodo (doi:10.5281/zenodo.23697084), expands
them into the original folder structure, and deletes the zips when done.

Usage:
    python download_data.py

Output:
    output/bin_inj/<planet_name>/collated_results.npz
    output/sin_inj/<planet_name>/collated_results.npz
"""

import pathlib
import shutil
import urllib.request
import zipfile

# ── Configuration ─────────────────────────────────────────────────────────────
DOWNLOADS = [
    {
        "url":      "https://zenodo.org/records/23697084/files/collated_npzs_bininj.zip?download=1",
        "zip_path": pathlib.Path("collated_npzs_bininj.zip"),
        "out_root": pathlib.Path("output/bin_inj"),
    },
    {
        "url":      "https://zenodo.org/records/23697084/files/collated_npzs_sininj.zip?download=1",
        "zip_path": pathlib.Path("collated_npzs_sininj.zip"),
        "out_root": pathlib.Path("output/sin_inj"),
    },
]
# ──────────────────────────────────────────────────────────────────────────────

for dl in DOWNLOADS:
    url      = dl["url"]
    zip_path = dl["zip_path"]
    out_root = dl["out_root"]

    print(f"Downloading '{zip_path.name}'...", flush=True)
    urllib.request.urlretrieve(url, zip_path)
    print(f"Saved to '{zip_path}'.", flush=True)

    print(f"Expanding into '{out_root}/'...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".npz")]
        for name in sorted(names):
            planet_name = pathlib.Path(name).stem
            dest_dir = out_root / planet_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / "collated_results.npz"
            print(f"  [extract] {name} -> {dest_file}", flush=True)
            with zf.open(name) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)

    zip_path.unlink()
    print(f"  [deleted] '{zip_path}'", flush=True)
    print(f"Done with '{zip_path.name}'.\n", flush=True)

print("All downloads complete.")
