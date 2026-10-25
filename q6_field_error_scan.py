"""
Quad field-error study, Q6 only (2026-09-21, refined + rebased same day):
21 Q6_B_SC values from -1% to +1% in 0.1% steps, each traced with the
DEFAULT stepped_uniform beam (generateInitialDistribution's built-in
default grid) from target to DSSD, reporting:
  - overall transmission, target -> DSSD
  - FP2 resolution = |ME(1,7)|/WV at FP2 Slits
  - FP3 resolution = |ME(1,7)|/WV at FP3 Slits
  - mean X and max-min X range AT DSSD ITSELF (reverted back from an
    earlier-same-day VD1879 substitution, per user request)

Rebased 2026-09-21 per user request: Q14 and Q15 are now held at their
optimized values (-20%, -20% -- the mid-ridge point on the transmission
plateau found by q14_q15_corner_extension_scan.py, 96.83% best achievable
via those two quads) for every point in this scan, instead of nominal.
So this Q6 error study is now "how sensitive is the ALREADY-OPTIMIZED
tune to a Q6 field error", not "how sensitive is the nominal tune". This
also means every point (including Q6=0%) needs a fresh COSY compile --
Q6=0% here is Q14=-20%/Q15=-20%/Q6=nominal, which is NOT the same as the
plain nominal "pg_June2025" cache, so that shortcut no longer applies.

WV throughout = max-min range (not std), per user's 2026-09-21 choice,
matching the .fox file's own VMAX(RAY(1))-VMIN(RAY(1)) convention.

ME(1,7) is the EXACT cumulative first-order (linear) transfer-matrix
element (row=x, column=dm), obtained by composing each element's local
linear submatrix (see compute_me17_cumulative.py) -- validated against
COSY's own native ME(1,7) at FP2 Slits to ~0.3% (2026-09-21).

Transmission uses the corrected boundary convention found 2026-09-18:
transmission_indices[i] > target_index (not >=) -- a ray blocked exactly
AT an index does not count as having reached it.

DSSD mean X and range X are now ADDITIONALLY reported under a
hypothetical 50cm x 50cm DSSD aperture (2026-09-21 user request), as
dssd_mean_x_50cm_mm / dssd_range_x_50cm_mm, alongside the original
real-aperture versions. This is possible without changing the beamline
polygons themselves: process_single_ray (ray_functions.pyx) computes
x_values/y_values for EVERY element up to final_index via an unconditional
matrix_transform loop -- the aperture/polygon check that determines
transmission_indices is a completely separate second pass over that
already-computed coordinate array (see lines ~569-629 of
ray_functions.pyx) and does not truncate it. So all_x[i][DSSD_INDEX] and
all_y[i][DSSD_INDEX] are valid, physically meaningful positions for every
ray that survived every upstream aperture up to DSSD -- including ones
the REAL (small) DSSD aperture blocked -- letting us re-apply a different
DSSD cut after the fact with no new COSY run needed for that alone. The
50cm hypothesis takes the candidate set to be every ray with
transmission_indices[i] >= DSSD_INDEX (survived everything upstream of
DSSD, real aperture or not) and keeps those with |x|<=250mm AND
|y|<=250mm at DSSD_INDEX (a 50x50cm square centered on the beam axis).
n_to_dssd_50cm / transmission_pct_50cm are also recorded for reference,
though only mean/range are plotted.

Checkpointed/resumable + parallelized across N_WORKERS separate processes
(ProcessPoolExecutor, not threading -- run_cosy_and_save_fort's
os.chdir() is whole-process state, so plain threads would race each
other's compiles; see q14_q15_local_refine_scan.py for the full
rationale).

Run yourself (per project convention -- COSY/ECAT runs and multi-point
searches are user-run, checkpointed so they can be stopped/restarted):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 q6_field_error_scan.py
"""
import os
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import element_names_stripped, pkl_directory
from helper_functions import runECAT, load_optics_data
from compute_me17_cumulative import cumulative_linear_matrix

N_WORKERS = 10

Q14_BASE = -0.20
Q15_BASE = -0.20

DSSD_HALF_SIZE_MM = 250.0  # 50cm x 50cm square, centered on the beam axis

FP2_SLITS_INDEX = element_names_stripped.index('FP2 Slits')
FP3_SLITS_INDEX = element_names_stripped.index('FP3 Slits & VD1783')
DSSD_INDEX = element_names_stripped.index('DSSD')

Q6_VALUES = [round(v, 4) for v in np.arange(-0.010, 0.0101, 0.001)]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "q6_field_error_scan_results.csv")
FIELDNAMES = ["Q6_B_SC", "n_launched", "n_to_dssd", "transmission_pct",
              "fp2_ME17", "fp2_WV_mm", "fp2_resolution",
              "fp3_ME17", "fp3_WV_mm", "fp3_resolution",
              "dssd_mean_x_mm", "dssd_range_x_mm",
              "n_to_dssd_50cm", "transmission_pct_50cm",
              "dssd_mean_x_50cm_mm", "dssd_range_x_50cm_mm"]


def evaluate(q6_value):
    tuneChanges = [['Q14', 'B_SC', Q14_BASE], ['Q15', 'B_SC', Q15_BASE]]
    if q6_value != 0.0:
        tuneChanges.append(['Q6', 'B_SC', float(q6_value)])
    tag = f"pg_June2025_q1415base_q6scan_{q6_value:+.4f}".replace(".", "p")
    cosyFile, run_cosy_flag = tag[:60], True

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)

    def x_at(idx):
        return np.array([-x[idx] * 1000 for i, x in enumerate(all_x) if transmission_indices[i] > idx])

    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    transmission_pct = 100 * n_to_dssd / n_total

    x_dssd = x_at(DSSD_INDEX)
    x_fp2 = x_at(FP2_SLITS_INDEX)
    x_fp3 = x_at(FP3_SLITS_INDEX)
    fp2_wv = x_fp2.max() - x_fp2.min()
    fp3_wv = x_fp3.max() - x_fp3.min()

    _, gdf = load_optics_data(usedFoxFile, pkl_directory)
    fp2_me17 = cumulative_linear_matrix(gdf, FP2_SLITS_INDEX)[0, 6]
    fp3_me17 = cumulative_linear_matrix(gdf, FP3_SLITS_INDEX)[0, 6]

    # hypothetical 50cm x 50cm DSSD: candidates are rays that survived
    # every upstream aperture up to DSSD (real aperture or not), then
    # re-cut in both X and Y against the bigger square
    candidates = [i for i, t in enumerate(transmission_indices) if t >= DSSD_INDEX]
    x_50cm = []
    for i in candidates:
        x_mm = -all_x[i][DSSD_INDEX] * 1000
        y_mm = -all_y[i][DSSD_INDEX] * 1000
        if abs(x_mm) <= DSSD_HALF_SIZE_MM and abs(y_mm) <= DSSD_HALF_SIZE_MM:
            x_50cm.append(x_mm)
    x_50cm = np.array(x_50cm)
    n_to_dssd_50cm = len(x_50cm)

    return {
        "Q6_B_SC": q6_value, "n_launched": n_total, "n_to_dssd": n_to_dssd,
        "transmission_pct": transmission_pct,
        "fp2_ME17": fp2_me17, "fp2_WV_mm": fp2_wv, "fp2_resolution": abs(fp2_me17) / (fp2_wv / 1000),
        "fp3_ME17": fp3_me17, "fp3_WV_mm": fp3_wv, "fp3_resolution": abs(fp3_me17) / (fp3_wv / 1000),
        "dssd_mean_x_mm": x_dssd.mean(), "dssd_range_x_mm": x_dssd.max() - x_dssd.min(),
        "n_to_dssd_50cm": n_to_dssd_50cm, "transmission_pct_50cm": 100 * n_to_dssd_50cm / n_total,
        "dssd_mean_x_50cm_mm": x_50cm.mean(), "dssd_range_x_50cm_mm": x_50cm.max() - x_50cm.min(),
    }


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set(round(float(r["Q6_B_SC"]), 4) for r in csv.DictReader(f))


if __name__ == "__main__":
    ensure_results_file_exists()
    done = load_done()

    tasks = [q6 for q6 in Q6_VALUES if round(q6, 4) not in done]
    print(f"{len(done)} points already done, {len(tasks)} remaining.\n")

    with open(RESULTS_FILE, "a", newline="") as f, ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        futures = {pool.submit(evaluate, q6): q6 for q6 in tasks}
        print(f"submitted {len(futures)} points across {N_WORKERS} worker processes\n", flush=True)
        for future in as_completed(futures):
            q6 = futures[future]
            row = future.result()
            writer.writerow(row)
            f.flush()
            print(f"Q6={q6*100:+.2f}%: transmission = {row['n_to_dssd']}/{row['n_launched']} = "
                  f"{row['transmission_pct']:.2f}%  FP2 res={row['fp2_resolution']:.2f}  "
                  f"FP3 res={row['fp3_resolution']:.2f}  DSSD mean_x={row['dssd_mean_x_mm']:+.3f}mm",
                  flush=True)

    print(f"\nDone. Wrote {RESULTS_FILE}")
