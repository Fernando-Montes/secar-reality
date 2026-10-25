"""
Quantifies, per knob, how efficiently it moves the M12(z)=0 and M16(z)=0
crossing points -- the design goal being to see which knobs (if any)
could plausibly shift BOTH crossings by +0.28m (downstream, from
upstream of VD1638 to downstream of it) within a realistic +-5% range,
per the user's 2026-08-12 hypothesis: E2's measured sign flip (M12/M16
both negative vs nominal's positive) implies the true crossings sit
downstream of VD1638, opposite of the nominal design -- and if a knob
can't plausibly produce that shift within +-5%, it's evidence the E2/E3
discrepancy isn't explained by ordinary tuning of that knob alone.

Knobs: Q1-Q7, H1-H3, O1 (all B_SC), B1-B4 Exit dB, WF1 Exit dB -- 16 total.
For each, sweeps +-5%/+-2.5% (4 points, nominal shared/reused across all
knobs), and at each point:
  - runs the SAME two tiny deterministic ray sets as z_evolution_test.py
    (M16 test: dE sweep +-3.1%, angle=0; M12 test: angle sweep +-25mrad,
    dE=0), reusing ONE COSY compile for both (run_cosy_flag=True for the
    first, False for the second, same pattern as fp2_metrics.py),
  - for each z-step, fits the per-ray X vs dE (or vs angle) slope across
    the 11 rays, then finds where that slope crosses zero (linear
    interpolation between the two bracketing z-samples) -- that's
    M16(z)=0 or M12(z)=0 respectively.

Search window restricted to FP1 Slits..FP3 Slits (z ~7.5-32m) to keep
the run cheap and avoid picking up an unrelated crossing further
downstream; final_index = FP3 Slits (no need to trace to DSSD for this).

Checkpointed/resumable: each (knob_id, value) point appended to
RESULTS_FILE as soon as it completes. A separate summary step (run at
the end, or via --summarize-only) fits d(z_crossing)/d(knob) per knob
and solves for the %/knob needed to shift each crossing by +0.28m.

Run:
    cd AgentSearch
    python3 z_crossing_sensitivity_scan.py
"""
import os
import sys
import csv
import time
import warnings
import numpy as np
from multiprocessing import Pool

from config import z_coordinates, element_names_stripped
from helper_functions import runECAT

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_STEPS = 11
FP1_IDX = element_names_stripped.index('FP1 Slits')
FP3_IDX = element_names_stripped.index('FP3 Slits & VD1783')
SEARCH_Z_MIN = z_coordinates[FP1_IDX]
SEARCH_Z_MAX = z_coordinates[FP3_IDX]
SHIFT_TARGET = 0.28  # meters, downstream

KNOB_ELEMENTS_BSC = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'H1', 'H2', 'H3', 'O1']
KNOB_ELEMENTS_DB = ['B1', 'B2', 'B3', 'B4', 'WF1']
KNOB_VALUES = [-0.05, -0.025, 0.025, 0.05]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "z_crossing_sensitivity_results.csv")
FIELDNAMES = ["knob_id", "value", "m16_crossing_z", "m12_crossing_z", "elapsed_s"]


def build_tasks():
    tasks = [("NOMINAL", 0.0, [])]
    for elem in KNOB_ELEMENTS_BSC:
        for v in KNOB_VALUES:
            tasks.append((f"{elem}_B_SC", v, [[elem, 'B_SC', v]]))
    for elem in KNOB_ELEMENTS_DB:
        for v in KNOB_VALUES:
            label = "WF1_Exit_dB" if elem == "WF1" else f"{elem}_Exit_dB"
            elem_name = "WF1 Exit" if elem == "WF1" else f"{elem} Exit"
            tasks.append((label, v, [[elem_name, 'dB', v]]))
    return tasks


def find_crossing_z(x_by_ray, param_values, z_min, z_max):
    """x_by_ray: list of per-ray X arrays (already truncated to common
    length). param_values: per-ray dE or angle value. Returns the
    interpolated z where the X-vs-param slope crosses zero within
    [z_min, z_max], or None if no sign change found there."""
    n_z = min(len(x) for x in x_by_ray)
    param_values = np.asarray(param_values)
    slopes = []
    zs = []
    for zi in range(n_z):
        z = z_coordinates[zi]
        if z < z_min or z > z_max:
            continue
        x_at_z = np.array([x_by_ray[r][zi] for r in range(len(x_by_ray))])
        slope = np.polyfit(param_values, x_at_z, 1)[0]
        slopes.append(slope)
        zs.append(z)
    slopes = np.array(slopes)
    zs = np.array(zs)
    sign_changes = np.where(np.diff(np.sign(slopes)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]  # first crossing in the window
    z1, z2 = zs[i], zs[i + 1]
    s1, s2 = slopes[i], slopes[i + 1]
    return z1 - s1 * (z2 - z1) / (s2 - s1)


def is_db_only(tuneChanges):
    """dB-type changes never touch the .fox file (see prepareTuneChanges
    in helper_functions.py -- they're applied as per-ray field
    perturbations, not a lattice edit), so tuneChanges that are empty or
    dB-only never produce a real cosyChanges edit. Calling runECAT
    directly (not through fp2_metrics.py's wrapper) means we must
    replicate that fallback ourselves: use the real base fox file,
    read-only (run_cosy_flag=False, after nominal has been computed
    once, sequentially, up front) -- otherwise (a) a made-up cosyFile
    name doesn't exist on disk (the exact bug fixed 2026-08-09 in
    fp2_metrics.py) and (b) multiple dB-only tasks running concurrently
    would all race on the SAME shared "pg_June2025" COSY output
    directory (the exact bug fixed 2026-08-10 in
    fp2_offset_jacobian_scan.py). Both fixes are re-applied here since
    this script calls runECAT directly."""
    return not tuneChanges or all(c[1] == 'dB' for c in tuneChanges)


def evaluate_point(task):
    knob_id, value, tuneChanges = task
    t0 = time.time()
    db_only = is_db_only(tuneChanges)
    if knob_id == "NOMINAL":
        # The one task that must actually compile "pg_June2025" -- run
        # sequentially, before the pool opens, so every other db_only
        # task can safely read it (never both compile the same shared
        # file concurrently).
        cosyFile = "pg_June2025"
        run_cosy_flag_1 = True
    elif db_only:
        cosyFile = "pg_June2025"
        run_cosy_flag_1 = False
    else:
        tag = f"{knob_id}_{value:+.5f}".replace(".", "p")
        cosyFile = f"pg_June2025_zxing_{tag}"[:60]
        run_cosy_flag_1 = True
    try:
        dist_m16 = {
            'type': 'stepped_uniform',
            'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': 0, 'max': 0, 'steps': 1},
            'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
            'dE': {'min': -3.1 / 100, 'max': 3.1 / 100, 'steps': N_STEPS},
            'dZ': {'min': 0, 'max': 0, 'steps': 1},
        }
        all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
            chamber_names, beampipes, usedFoxFile = runECAT(
                initialDistribution=dist_m16, run_cosy_flag=run_cosy_flag_1, tuneChanges=tuneChanges,
                save_rays_flag=False, final_index=FP3_IDX, foxFile=cosyFile)
        dE_vals = [dE[0] for dE in all_dE]
        m16_z = find_crossing_z(transmitted_x, dE_vals, SEARCH_Z_MIN, SEARCH_Z_MAX)

        dist_m12 = {
            'type': 'stepped_uniform',
            'x': {'min': 0, 'max': 0, 'steps': 1}, 'ax': {'min': -25 / 1000, 'max': 25 / 1000, 'steps': N_STEPS},
            'y': {'min': 0, 'max': 0, 'steps': 1}, 'ay': {'min': 0, 'max': 0, 'steps': 1},
            'dE': {'min': 0, 'max': 0, 'steps': 1}, 'dZ': {'min': 0, 'max': 0, 'steps': 1},
        }
        all_x2, all_ax2, all_y2, all_ay2, all_dE2, transmitted_x2, transmitted_y2, transmission_indices2, \
            chamber_names2, beampipes2, usedFoxFile2 = runECAT(
                initialDistribution=dist_m12, run_cosy_flag=False, tuneChanges=tuneChanges,
                save_rays_flag=False, final_index=FP3_IDX, foxFile=usedFoxFile)
        ax_vals = [ax[0] for ax in all_ax2]
        m12_z = find_crossing_z(transmitted_x2, ax_vals, SEARCH_Z_MIN, SEARCH_Z_MAX)

        return {"knob_id": knob_id, "value": value,
                "m16_crossing_z": m16_z if m16_z is not None else "",
                "m12_crossing_z": m12_z if m12_z is not None else "",
                "elapsed_s": time.time() - t0}
    except Exception as e:
        return ("FAILED", f"{knob_id}={value}: {e}")


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_completed():
    completed = set()
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            completed.add((row["knob_id"], round(float(row["value"]), 8)))
    return completed


def append_result(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


if __name__ == "__main__":
    ensure_results_file_exists()
    completed = load_completed()

    if ("NOMINAL", 0.0) not in completed:
        print("Computing NOMINAL sequentially (bootstraps the shared pg_June2025 matrix)...")
        nominal_row = evaluate_point(("NOMINAL", 0.0, []))
        if isinstance(nominal_row, tuple) and nominal_row[0] == "FAILED":
            raise RuntimeError(f"NOMINAL evaluation failed: {nominal_row[1]}")
        append_result(nominal_row)
        completed.add(("NOMINAL", 0.0))
        print(f"  NOMINAL done ({nominal_row['elapsed_s']:.0f}s)  "
              f"M16z={nominal_row['m16_crossing_z']}  M12z={nominal_row['m12_crossing_z']}\n")

    tasks = [t for t in build_tasks() if (t[0], round(t[1], 8)) not in completed]
    print(f"{len(completed)} points already done, {len(tasks)} remaining.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for msg in pool.imap_unordered(evaluate_point, tasks):
                if isinstance(msg, tuple) and msg[0] == "FAILED":
                    print(f"  FAILED: {msg[1]}", flush=True)
                else:
                    append_result(msg)
                    print(f"  {msg['knob_id']}={msg['value']:+.5f}  M16z={msg['m16_crossing_z']}  "
                          f"M12z={msg['m12_crossing_z']}  ({msg['elapsed_s']:.0f}s)", flush=True)

    print("\nDone. Run z_crossing_sensitivity_summary.py to build the ranking table.")
