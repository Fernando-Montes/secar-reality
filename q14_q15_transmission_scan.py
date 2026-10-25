"""
Q14/Q15 transmission-maximization scan (2026-09-21): coarse 5x5 grid over
Q14_B_SC x Q15_B_SC (+-10% each), default stepped_uniform beam, target ->
DSSD, to find which combination maximizes transmission before refining
locally around the best point.

Transmission uses the corrected boundary convention (transmission_indices
> DSSD_INDEX, not >=, found 2026-09-18).

Only the (0,0) point reuses the existing nominal "pg_June2025" cache; the
other 24 combinations each require a fresh COSY compile.

Checkpointed/resumable: each (Q14,Q15) point appended to RESULTS_FILE as
soon as it completes.

Run:
    cd AgentSearch
    python3 q14_q15_transmission_scan.py
"""
import os
import csv
import numpy as np

from config import element_names_stripped
from helper_functions import runECAT

DSSD_INDEX = element_names_stripped.index('DSSD')

GRID_VALUES = [-0.10, -0.05, 0.0, 0.05, 0.10]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "q14_q15_transmission_scan_results.csv")
FIELDNAMES = ["Q14_B_SC", "Q15_B_SC", "n_launched", "n_to_dssd", "transmission_pct"]


def evaluate(q14, q15):
    tuneChanges = []
    if q14 != 0.0:
        tuneChanges.append(['Q14', 'B_SC', float(q14)])
    if q15 != 0.0:
        tuneChanges.append(['Q15', 'B_SC', float(q15)])

    if not tuneChanges:
        cosyFile, run_cosy_flag = "pg_June2025", False
    else:
        tag = f"pg_June2025_q1415_{q14:+.3f}_{q15:+.3f}".replace(".", "p")
        cosyFile, run_cosy_flag = tag[:60], True

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)
    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    return {"Q14_B_SC": q14, "Q15_B_SC": q15, "n_launched": n_total,
            "n_to_dssd": n_to_dssd, "transmission_pct": 100 * n_to_dssd / n_total}


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set((round(float(r["Q14_B_SC"]), 6), round(float(r["Q15_B_SC"]), 6))
                   for r in csv.DictReader(f))


if __name__ == "__main__":
    ensure_results_file_exists()
    done = load_done()

    tasks = [(q14, q15) for q14 in GRID_VALUES for q15 in GRID_VALUES
             if (round(q14, 6), round(q15, 6)) not in done]
    print(f"{len(done)} points already done, {len(tasks)} remaining.\n")

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for q14, q15 in tasks:
            print(f"Q14={q14*100:+.1f}%  Q15={q15*100:+.1f}% ...", flush=True)
            row = evaluate(q14, q15)
            writer.writerow(row)
            f.flush()
            print(f"  transmission = {row['n_to_dssd']}/{row['n_launched']} = {row['transmission_pct']:.2f}%\n",
                  flush=True)

    print(f"Done. Wrote {RESULTS_FILE}")
