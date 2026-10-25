"""
Combined quad-mistune Monte Carlo (2026-09-21): each trial independently
draws EVERY one of the 15 quads' B_SC from Uniform[-1%, +1%] AROUND ITS
OWN operating point (0% nominal for all quads except Q14/Q15, which
operate at the optimized -20% -- same operating point as
quad_field_error_scan_all.py), runs ONE COSY compile for that combined
15-quad mistune, and records all 5 tracked metrics:
  - transmission, target -> DSSD (real DSSD aperture)
  - FP2 resolution = |ME(1,7)|/WV at FP2 Slits
  - FP3 resolution = |ME(1,7)|/WV at FP3 Slits
  - DSSD mean X and range X ("dX"), both the real aperture and the
    50cm x 50cm hypothetical aperture (see q6_field_error_scan.py /
    quad_field_error_scan_all.py docstrings for why the 50cm version can
    be computed from the same run with no extra COSY needed, and why
    it's the smoother/preferred one for downstream analysis)

Unlike the earlier grid/OAT-style scans, every trial mistunes ALL 15
quads simultaneously and independently -- no two trials share a tune, so
there is nothing to seed/reuse and no fixed grid to complete. Per user
request 2026-09-21, this runs INDEFINITELY, appending one row per
completed trial, until manually stopped (Ctrl-C). Stopping is graceful:
in-flight trials are allowed to finish and get written before exit, so
the results file is never left with a half-written row, and re-running
the script later just continues appending more trials (trial_id resumes
from one past the current max in the file).

Disk usage: since this runs indefinitely, each trial deletes its own
per-run compile artifacts (Matrix_Files/<cosyFile>/, pkl_Files/<cosyFile>.pkl,
COSY10.2/<cosyFile>.fox) immediately after extracting what it needs --
otherwise an indefinite run would reproduce the ~229GB pileup that had to
be cleaned up earlier the same day (2026-09-21). Only the named baseline
caches (pg_June2025 etc.) are ever kept.

Parallelized across N_WORKERS separate processes (ProcessPoolExecutor,
not threading -- run_cosy_and_save_fort's os.chdir() is whole-process
state; see q14_q15_local_refine_scan.py for the full rationale), using a
rolling pool: as each trial finishes, a new random trial is immediately
submitted to take its place, so N_WORKERS compiles are always in flight
(until a stop is requested).

Run yourself (per project convention -- COSY/ECAT runs and multi-point
searches are user-run; stop anytime with Ctrl-C, resume anytime by
running again):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 mc_quad_mistune_scan.py
"""
import os
import csv
import shutil
import signal
import numpy as np
import concurrent.futures as cf

from config import element_names_stripped, pkl_directory, matrix_directory, COSY_directory
from helper_functions import runECAT, load_optics_data
from compute_me17_cumulative import cumulative_linear_matrix

N_WORKERS = 10

QUADS = [f"Q{i}" for i in range(1, 16)]
OPTIMIZED_BASE = {"Q14": -0.20, "Q15": -0.20}  # everything else defaults to 0.0 (nominal)
DELTA_LOW, DELTA_HIGH = -0.010, 0.010

DSSD_HALF_SIZE_MM = 250.0  # 50cm x 50cm square, centered on the beam axis

FP2_SLITS_INDEX = element_names_stripped.index('FP2 Slits')
FP3_SLITS_INDEX = element_names_stripped.index('FP3 Slits & VD1783')
DSSD_INDEX = element_names_stripped.index('DSSD')

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "mc_quad_mistune_scan_results.csv")
FIELDNAMES = ["trial_id"] + [f"{q}_delta" for q in QUADS] + [
    "n_launched", "n_to_dssd", "transmission_pct",
    "fp2_ME17", "fp2_WV_mm", "fp2_resolution",
    "fp3_ME17", "fp3_WV_mm", "fp3_resolution",
    "dssd_mean_x_mm", "dssd_range_x_mm",
    "n_to_dssd_50cm", "transmission_pct_50cm",
    "dssd_mean_x_50cm_mm", "dssd_range_x_50cm_mm",
]


def evaluate(trial_id, deltas):
    tuneChanges = [[q, 'B_SC', OPTIMIZED_BASE.get(q, 0.0) + deltas[q]] for q in QUADS]
    cosyFile = f"pg_June2025_mc_{trial_id}"[:60]

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=True, foxFile=cosyFile,
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

    candidates = [i for i, t in enumerate(transmission_indices) if t >= DSSD_INDEX]
    x_50cm = []
    for i in candidates:
        x_mm = -all_x[i][DSSD_INDEX] * 1000
        y_mm = -all_y[i][DSSD_INDEX] * 1000
        if abs(x_mm) <= DSSD_HALF_SIZE_MM and abs(y_mm) <= DSSD_HALF_SIZE_MM:
            x_50cm.append(x_mm)
    x_50cm = np.array(x_50cm)
    n_to_dssd_50cm = len(x_50cm)

    # every trial's per-run compile artifacts (Matrix_Files subdirectory,
    # pkl_Files cache, COSY10.2 .fox) are disposable once this point's
    # metrics are extracted -- delete them immediately so an indefinite
    # run doesn't reproduce the ~229GB pileup cleaned up earlier
    # (2026-09-21). Safe here: everything needed (gdf, all_x/all_y) is
    # already in memory, and each trial's cosyFile name is unique to
    # this worker's own files.
    #
    # BUG FOUND 2026-09-22: this originally deleted by `cosyFile` (the tag
    # passed IN to runECAT), but runECAT/prepareCOSYchanges silently
    # generates its OWN internal hash-based filename whenever tuneChanges
    # is non-empty (always true here, since every trial mistunes all 15
    # quads) and returns the ACTUAL name used as `usedFoxFile` -- it does
    # NOT reuse the passed-in tag. So this cleanup was deleting a
    # directory/files that never existed, while the real (hash-named)
    # artifacts were never touched -- silently piling up until the disk
    # filled completely (4331 leaked trials, 394GB, "no space left on
    # device"). Fixed by cleaning up `usedFoxFile`, the name runECAT
    # actually used.
    shutil.rmtree(os.path.join(matrix_directory, usedFoxFile), ignore_errors=True)
    pkl_path = os.path.join(pkl_directory, f"{usedFoxFile}.pkl")
    if os.path.exists(pkl_path):
        os.remove(pkl_path)
    fox_path = os.path.join(COSY_directory, f"{usedFoxFile}.fox")
    if os.path.exists(fox_path):
        os.remove(fox_path)

    row = {"trial_id": trial_id}
    row.update({f"{q}_delta": deltas[q] for q in QUADS})
    row.update({
        "n_launched": n_total, "n_to_dssd": n_to_dssd, "transmission_pct": transmission_pct,
        "fp2_ME17": fp2_me17, "fp2_WV_mm": fp2_wv, "fp2_resolution": abs(fp2_me17) / (fp2_wv / 1000),
        "fp3_ME17": fp3_me17, "fp3_WV_mm": fp3_wv, "fp3_resolution": abs(fp3_me17) / (fp3_wv / 1000),
        "dssd_mean_x_mm": x_dssd.mean(), "dssd_range_x_mm": x_dssd.max() - x_dssd.min(),
        "n_to_dssd_50cm": n_to_dssd_50cm, "transmission_pct_50cm": 100 * n_to_dssd_50cm / n_total,
        "dssd_mean_x_50cm_mm": x_50cm.mean(), "dssd_range_x_50cm_mm": x_50cm.max() - x_50cm.min(),
    })
    return row


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def next_trial_id():
    with open(RESULTS_FILE, "r", newline="") as f:
        ids = [int(r["trial_id"]) for r in csv.DictReader(f)]
    return (max(ids) + 1) if ids else 0


def random_deltas(rng):
    return {q: rng.uniform(DELTA_LOW, DELTA_HIGH) for q in QUADS}


_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    if not _stop_requested:
        print("\nStop requested -- finishing in-flight trials, then exiting cleanly "
              "(press Ctrl-C again to force-quit immediately)...", flush=True)
        _stop_requested = True
    else:
        raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    ensure_results_file_exists()
    trial_id = next_trial_id()
    print(f"Resuming from trial_id={trial_id}. Running indefinitely -- Ctrl-C to stop.\n", flush=True)

    rng = np.random.default_rng()
    n_completed = 0

    with open(RESULTS_FILE, "a", newline="") as f, cf.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        pending = {}

        for _ in range(N_WORKERS):
            deltas = random_deltas(rng)
            fut = pool.submit(evaluate, trial_id, deltas)
            pending[fut] = trial_id
            trial_id += 1

        while pending:
            done, _ = cf.wait(pending.keys(), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                tid = pending.pop(fut)
                row = fut.result()
                writer.writerow(row)
                f.flush()
                n_completed += 1
                print(f"[trial {tid}] ({n_completed} completed) "
                      f"transmission={row['transmission_pct']:.2f}%  "
                      f"FP2 res={row['fp2_resolution']:.2f}  FP3 res={row['fp3_resolution']:.2f}  "
                      f"DSSD mean_x={row['dssd_mean_x_50cm_mm']:+.2f}mm  "
                      f"DSSD range_x={row['dssd_range_x_50cm_mm']:.2f}mm", flush=True)

                if not _stop_requested:
                    deltas = random_deltas(rng)
                    newfut = pool.submit(evaluate, trial_id, deltas)
                    pending[newfut] = trial_id
                    trial_id += 1

    print(f"\nStopped. {n_completed} trials completed this run. Wrote {RESULTS_FILE}")
