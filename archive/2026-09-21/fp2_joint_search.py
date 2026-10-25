"""
FP2 joint search: Q6 B_SC (outer, expensive — one COSY compute per value)
x WF1 Exit dB (inner, cheap — no COSY recompute, applied at ray-propagation
time). For each (Q6, WF1_dB) grid point, computes:
  - M12 (dX/dalphaX) and M16 (dX/dE) simulated slopes, fit against the
    11 confirmed E2 groups (measured targets: M12=-0.2472, M16=-0.3842)
  - a diagnostic-only true-FP2-Slits-vs-VD1638 X width (std), matching
    the E3 methodology, NOT used in the loss itself
  - a combined loss: sigma-normalized (M12, M16, Q6-vs-E3-prior) terms,
    equal explicit weight (1:1:1), per user agreement 2026-08-07:
        loss = ((M12_sim - M12_TARGET)/SIGMA_M12)^2
             + ((M16_sim - M16_TARGET)/SIGMA_M16)^2
             + ((Q6      - Q6_E3_PRIOR)/SIGMA_Q6)^2

Run this directly (not imported) — designed to be started in a terminal,
left running, and safely killed/restarted:
    cd AgentSearch
    python3 fp2_joint_search.py
Results are appended to fp2_joint_search_results.csv as each grid point
finishes; on restart, already-completed (Q6, WF1_dB) pairs (identified by
their rounded grid values) are skipped by reading that file first.
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from config import matrix_directory, pkl_directory
from fp2_metrics import compute_M12_M16, compute_fp2_width

# ---------------------------------------------------------------- targets
M12_TARGET = -0.2472
M16_TARGET = -0.3842
Q6_E3_PRIOR = -0.005  # sim-direction equivalent of E3's "+0.5% hardware" finding

SIGMA_M12 = 0.02
SIGMA_M16 = 0.03
SIGMA_Q6 = 0.0015

# ------------------------------------------------------------------ grid
# First-pass exploratory settings (2026-08-07): thinner grid + fewer rays
# than a final run, specifically to get real per-point timing before
# committing to a longer, denser run. Refine (finer grid, more rays)
# around whatever region looks promising once this pass is done.
Q6_VALUES = np.round(np.linspace(-0.05, 0.05, 21), 6)       # -5% to +5%, 0.5% steps
WF1_DB_VALUES = np.round(np.linspace(-0.01, 0.01, 11), 6)   # -1% to +1%, 0.2% steps

NRAYS_GROUPS = 1000     # per E2 group, per (Q6, WF1_dB) point
NRAYS_WIDTH = 1000      # for the generic-distribution FP2-slits/VD1638 diagnostic
TA_Y_OFFSET = 0.1/1000  # fixed at the E2 "best overall" value for this round

RESULTS_FILE = "Results/fp2_joint_search_results.csv"  # moved here 2026-08-08, run complete
FIELDNAMES = ["Q6", "WF1_dB", "M12_sim", "M16_sim", "loss",
              "fp2_slits_std", "vd1638_std", "fp2_slits_mean", "vd1638_mean",
              "elapsed_s"]


def loss_fn(M12_sim, M16_sim, Q6):
    t12 = ((M12_sim - M12_TARGET) / SIGMA_M12) ** 2
    t16 = ((M16_sim - M16_TARGET) / SIGMA_M16) ** 2
    tQ6 = ((Q6 - Q6_E3_PRIOR) / SIGMA_Q6) ** 2
    return t12 + t16 + tQ6


def load_completed():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                completed.add((round(float(row["Q6"]), 6), round(float(row["WF1_dB"]), 6)))
    return completed


def ensure_results_file_exists():
    # Must be called once, before the worker Pool starts, to avoid a race
    # where multiple workers each see "file missing" and all write a
    # header row (corrupting the CSV).
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_result(row):
    # File is guaranteed to exist (see ensure_results_file_exists) by the
    # time any worker calls this, so every worker just appends a data row.
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def run_one_q6(q6_idx):
    """Runs in its own process: one COSY compute for this Q6, reused for
    every WF1_dB value and every E2 group and the width diagnostic."""
    Q6 = float(Q6_VALUES[q6_idx])
    completed = load_completed()  # re-read per worker in case of a prior partial run
    pending_wf1 = [w for w in WF1_DB_VALUES if (round(Q6, 6), round(float(w), 6)) not in completed]
    if not pending_wf1:
        return f"[Q6={Q6:+.4f}] all WF1 points already done, skipping"

    cosyFile = f"pg_June2025_Q6search_{q6_idx}"
    results_written = 0
    try:
        for j, WF1_dB in enumerate(pending_wf1):
            WF1_dB = float(WF1_dB)
            t0 = time.time()
            tuneChanges = [['Q6', 'B_SC', Q6], ['WF1 Exit', 'dB', WF1_dB]]

            first_call = (j == 0)
            M12_sim, M16_sim, per_group, usedFoxFile = compute_M12_M16(
                tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile=cosyFile,
                ta_y_offset=TA_Y_OFFSET, verbose=False,
                run_cosy_flag=(True if first_call else False))
            cosyFile = usedFoxFile  # lock in actual name after first (possibly COSY-run) call

            width = compute_fp2_width(
                tuneChanges=tuneChanges, nRays=NRAYS_WIDTH, cosyFile=cosyFile,
                run_cosy_flag=False, verbose=False)

            loss = loss_fn(M12_sim, M16_sim, Q6)
            elapsed = time.time() - t0

            append_result({
                "Q6": Q6, "WF1_dB": WF1_dB, "M12_sim": M12_sim, "M16_sim": M16_sim,
                "loss": loss, "fp2_slits_std": width["fp2_std"], "vd1638_std": width["vd1638_std"],
                "fp2_slits_mean": width["fp2_mean"], "vd1638_mean": width["vd1638_mean"],
                "elapsed_s": elapsed,
            })
            results_written += 1

        # Clean up this Q6's temporary matrix/pkl files now that all its WF1 points are done.
        os.system(f"rm -rf {matrix_directory}/{cosyFile}")
        os.system(f"rm -rf {pkl_directory}/{cosyFile}.pkl")

        return f"[Q6={Q6:+.4f}] done, {results_written} new points written"
    except Exception as e:
        return f"[Q6={Q6:+.4f}] FAILED after {results_written} points: {e}"


if __name__ == "__main__":
    ensure_results_file_exists()
    completed = load_completed()
    total_points = len(Q6_VALUES) * len(WF1_DB_VALUES)
    print(f"Grid: {len(Q6_VALUES)} Q6 values x {len(WF1_DB_VALUES)} WF1_dB values = {total_points} points")
    print(f"Already completed (from {RESULTS_FILE} if present): {len(completed)}")
    print(f"Results file: {os.path.abspath(RESULTS_FILE)}\n")

    q6_indices = list(range(len(Q6_VALUES)))
    with Pool(processes=10) as pool:
        for msg in pool.imap_unordered(run_one_q6, q6_indices):
            print(msg, flush=True)

    print("\nAll Q6 values processed (or already complete). Check the results file for the full grid.")
