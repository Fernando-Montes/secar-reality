"""
Fixes a real bug found 2026-08-14 in z_crossing_sensitivity_scan.py:
find_crossing_z returned only the FIRST sign-change of the M12(z) or
M16(z) slope inside a wide fixed window (FP1 Slits..FP3 Slits,
z~7.5-32m). For knobs with real leverage (Q1-Q6, B1, B2 -- confirmed by
inspecting the raw crossing_z values per knob, see chat 2026-08-14), a
+-2.5%/+-5% perturbation routinely pushed the near-VD1638 crossing far
enough to either leave the window entirely, or be overtaken by a
completely different, unrelated root elsewhere in the window (typically
near z~7.6-7.7m, right at the FP1 edge) -- which the old summary script
then silently treated as if it were the SAME crossing continuing to
move, producing nonsense "required %" numbers. This is what caused the
apparent (and wrong) conclusion that Q3 could flip both the M12 and M16
crossings past VD1638 within a tiny +-0.15-0.6%, contradicting the
directly-measured dM12/dQ3 (independently confirmed consistent, both
sign and magnitude, by TWO different methods: the group/FP1-slit OAT
scan, and a clean aperture-free ray fan -- see q3_raw_fan_check.py).

Fix, this script (raw data only -- the actual fix is applied in the
CONTINUATION step, z_crossing_sensitivity_continuation.py): records
EVERY sign-change/crossing found in the window, not just the first, as
a semicolon-joined list. Also uses a much finer knob-value grid
(8 magnitudes x 2 signs, starting at 0.25%) so the continuation step
downstream can reliably track a SINGLE physical crossing step-by-step
without it having room to jump between unrelated roots.

All 16 knobs re-scanned (not just the 8 previously flagged as
compromised) for a single, internally-consistent dataset.

Checkpointed/resumable: each (knob_id, value) point appended to
RESULTS_FILE as soon as it completes.

Run:
    cd AgentSearch
    python3 z_crossing_sensitivity_scan_v2.py
"""
import os
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

KNOB_ELEMENTS_BSC = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'H1', 'H2', 'H3', 'O1']
KNOB_ELEMENTS_DB = ['B1', 'B2', 'B3', 'B4', 'WF1']
MAGNITUDES = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
KNOB_VALUES = [-m for m in reversed(MAGNITUDES)] + [m for m in MAGNITUDES]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "z_crossing_sensitivity_results_v2.csv")
FIELDNAMES = ["knob_id", "value", "m16_crossings_z", "m12_crossings_z", "elapsed_s"]


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


def find_all_crossings_z(x_by_ray, param_values, z_min, z_max):
    """Returns EVERY interpolated z where the X-vs-param slope crosses
    zero within [z_min, z_max] (not just the first), so downstream
    continuation tracking can pick the physically-correct one."""
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
    crossings = []
    for i in sign_changes:
        z1, z2 = zs[i], zs[i + 1]
        s1, s2 = slopes[i], slopes[i + 1]
        crossings.append(z1 - s1 * (z2 - z1) / (s2 - s1))
    return crossings


def is_db_only(tuneChanges):
    """See z_crossing_sensitivity_scan.py's docstring for the full
    explanation -- dB-only tuneChanges never touch the .fox file, so
    they must reuse the real base fox file, read-only."""
    return not tuneChanges or all(c[1] == 'dB' for c in tuneChanges)


def evaluate_point(task):
    knob_id, value, tuneChanges = task
    t0 = time.time()
    db_only = is_db_only(tuneChanges)
    if knob_id == "NOMINAL":
        cosyFile = "pg_June2025"
        run_cosy_flag_1 = True
    elif db_only:
        cosyFile = "pg_June2025"
        run_cosy_flag_1 = False
    else:
        tag = f"{knob_id}_{value:+.5f}".replace(".", "p")
        cosyFile = f"pg_June2025_zxing2_{tag}"[:60]
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
        m16_crossings = find_all_crossings_z(transmitted_x, dE_vals, SEARCH_Z_MIN, SEARCH_Z_MAX)

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
        m12_crossings = find_all_crossings_z(transmitted_x2, ax_vals, SEARCH_Z_MIN, SEARCH_Z_MAX)

        return {"knob_id": knob_id, "value": value,
                "m16_crossings_z": ";".join(f"{z:.5f}" for z in m16_crossings),
                "m12_crossings_z": ";".join(f"{z:.5f}" for z in m12_crossings),
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
              f"M16z={nominal_row['m16_crossings_z']}  M12z={nominal_row['m12_crossings_z']}\n")

    tasks = [t for t in build_tasks() if (t[0], round(t[1], 8)) not in completed]
    print(f"{len(completed)} points already done, {len(tasks)} remaining.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for msg in pool.imap_unordered(evaluate_point, tasks):
                if isinstance(msg, tuple) and msg[0] == "FAILED":
                    print(f"  FAILED: {msg[1]}", flush=True)
                else:
                    append_result(msg)
                    print(f"  {msg['knob_id']}={msg['value']:+.5f}  M16z=[{msg['m16_crossings_z']}]  "
                          f"M12z=[{msg['m12_crossings_z']}]  ({msg['elapsed_s']:.0f}s)", flush=True)

    print("\nDone. Run z_crossing_sensitivity_continuation.py next to track crossings and build the ranking.")
