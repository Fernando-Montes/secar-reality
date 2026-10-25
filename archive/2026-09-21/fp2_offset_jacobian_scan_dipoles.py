"""
Extends the offset-Jacobian scan (fp2_offset_jacobian_scan.py) to cover
B1, B2, B3, B4 Exit dB -- per the user's request 2026-08-12, after the
derivative-based ranking (plot_oat_derivative_ranking.py) showed these
have real local leverage on the E2 loss that the family search's
Jacobian doesn't yet capture (it only had per-group/offset data for the
10 B_SC elements + WF1 Exit dB).

All 4 knobs are dB-type, same as WF1 -- never touch the .fox file (see
_needs_base_fox_fallback in fp2_metrics.py), so every point here is a
pure READ of the already-cached "pg_June2025" pkl (which every script so
far has explicitly protected from cleanup: `if usedFoxFile !=
"pg_June2025": rm ...`). No compile needed at all (unlike the original
WF1 case, which had to bootstrap nominal sequentially first) -- purely
read-only access, so all 40 points are safe to evaluate fully in
parallel with no race-condition risk.

Appends to the SAME Results/fp2_offset_jacobian_scan_results.csv used by
fp2_family_mc.py's build_jacobian(), same column format, same +-1%
DB_RANGE/10-point grid as WF1's original entries.

Run:
    cd AgentSearch
    python3 fp2_offset_jacobian_scan_dipoles.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp2_metrics import compute_M12_M16, ALL_GROUPS

DB_RANGE = 0.01
N_POINTS = 10
NRAYS_GROUPS = 1000

OUTPUT_DIR = "Results"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_offset_jacobian_scan_results.csv")
FIELDNAMES = ["knob_id", "value"] + [f"group_{g}_mean_x" for g in ALL_GROUPS] + ["elapsed_s"]

DB_ELEMENTS = ['B1', 'B2', 'B3', 'B4']
KNOBS = [
    (f"{elem}_Exit_dB", (lambda v, e=elem: [[f"{e} Exit", 'dB', v]]),
     np.round(np.linspace(-DB_RANGE, DB_RANGE, N_POINTS), 6))
    for elem in DB_ELEMENTS
]


def run_one_point(task):
    knob_id, value, tuneChanges = task
    t0 = time.time()
    try:
        M12, M16, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile="pg_June2025",
            ta_y_offset=0.0, verbose=False, run_cosy_flag=False)
        row = {"knob_id": knob_id, "value": value, "elapsed_s": time.time() - t0}
        for g in ALL_GROUPS:
            row[f"group_{g}_mean_x"] = per_group[g][0]
        return row
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

    tasks = []
    for knob_id, builder, values in KNOBS:
        for v in values:
            v = float(v)
            if (knob_id, round(v, 8)) in completed:
                continue
            tasks.append((knob_id, v, builder(v)))

    print(f"{len(completed)} points already done, {len(tasks)} remaining.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for msg in pool.imap_unordered(run_one_point, tasks):
                if isinstance(msg, tuple) and msg[0] == "FAILED":
                    print(f"  FAILED: {msg[1]}", flush=True)
                else:
                    append_result(msg)
                    print(f"  {msg['knob_id']}={msg['value']:+.5f}  done ({msg['elapsed_s']:.0f}s)", flush=True)

    print("\nDone. Re-run to pick up any points that failed/timed out.")
