"""
Extends the OAT-slope Jacobian with offset (intercept) information, per
the user's 2026-08-10 suggestion: we've only ever fit the SLOPE of the
X-vs-dalphaX and X-vs-dE lines (M12, M16) against the measured slopes,
throwing away the measured OFFSETS (intercepts) even though we have them
too (DALPHAX_GROUPS_MEASURED_X / DE_GROUPS_MEASURED_X in fp2_metrics.py).
Adding those as 2 more constraints may reduce or eliminate the 2D
degenerate family found from the M12/M16-only Jacobian -- though the user
correctly cautioned this should NOT be restricted to just the Q6/Q2/Q5/WF1
knobs kept after the earlier degeneracy reduction: that reduction was
based on redundancy in (M12,M16) space only, so a knob pair that looked
redundant there could easily be independent once offsets are added, and
pre-selecting away from it would hide exactly the degeneracy we're
looking for. So this scan covers the FULL original B_SC-type candidate
pool from the OAT screen (H1, H3, O1, Q1-Q7 -- 10 knobs) plus WF1 Exit
dB, not just the 4 previously kept -- the resulting Jacobian will be 4
rows (M12,M16,Offset12,Offset16) x 11 columns, and its null space (via
SVD, no pre-selection) reveals whatever real degeneracy exists across the
whole pool.

compute_M12_M16 already computes per-group means internally (needed to
fit M12/M16 in the first place) but only ever returned the fitted SLOPE,
discarding the per-group means themselves -- so we have no saved data to
fit the offsets from the existing OAT screen or BO runs. This script
reruns the same knob x value grid as the OAT screen (B_SC +-10%, dB +-1%,
10 points each, plus nominal), but this time saves the per-group means so
both slope AND intercept can be fit vs. each knob.

Checkpointed (resumable): each (knob_id, value) point is appended to
RESULTS_FILE as soon as it completes; already-completed points are
skipped on restart.

Run:
    cd AgentSearch
    python3 fp2_offset_jacobian_scan.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp2_metrics import compute_M12_M16, ALL_GROUPS

BSC_RANGE = 0.10
DB_RANGE = 0.01
N_POINTS = 10
NRAYS_GROUPS = 1000

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_offset_jacobian_scan_results.csv")
FIELDNAMES = ["knob_id", "value"] + [f"group_{g}_mean_x" for g in ALL_GROUPS] + ["elapsed_s"]

BSC_ELEMENTS = ['H1', 'H3', 'O1', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7']

KNOBS = [
    (f"{elem}_B_SC", (lambda v, e=elem: [[e, 'B_SC', v]]), np.round(np.linspace(-BSC_RANGE, BSC_RANGE, N_POINTS), 6))
    for elem in BSC_ELEMENTS
] + [
    ("WF1_Exit_dB", lambda v: [['WF1 Exit', 'dB', v]], np.round(np.linspace(-DB_RANGE, DB_RANGE, N_POINTS), 6)),
]


def run_one_point(task):
    knob_id, value, tuneChanges, is_shared = task
    t0 = time.time()
    if is_shared:
        # dB-only (or empty) tuneChanges never triggers a .fox edit (see
        # _needs_base_fox_fallback in fp2_metrics.py), so it ALWAYS
        # resolves to the same shared "pg_June2025" COSY output
        # directory. Running multiple such tasks in the same parallel
        # Pool batch is a race condition -- confirmed 2026-08-10 via
        # check_wf1_stability.py: sequential reruns of the same WF1
        # point were stable (group_3 mean_x=-14.95/-14.968mm, twice),
        # but the parallel scan's saved value for that exact point was
        # -69.75mm -- far beyond what n~100-ray sampling noise (~0.05mm,
        # from std/sqrt(n)) could produce. helper_functions.py's
        # run_cosy_and_save_fort always writes to
        # {matrix_directory}/pg_June2025/ regardless of which worker is
        # running, so concurrent WF1 tasks were overwriting each other's
        # fort.* files mid-read. Fix: read-only here (run_cosy_flag=
        # False), reusing the nominal matrix computed once, sequentially,
        # before the pool opens (see __main__ below) -- never writes
        # during the parallel phase, so no race is possible.
        cosyFile = "pg_June2025"
        run_cosy_flag = False
    else:
        # B_SC-type changes DO produce a real .fox edit -> a unique
        # per-value cosyFile/directory, so these are already safe to
        # compile in parallel (run_cosy_flag=None keeps the normal
        # compile-once-on-first-group-then-reuse behavior).
        tag = f"{value:+.6f}".replace(".", "p")
        cosyFile = f"pg_June2025_offjac_{knob_id}_{tag}"
        run_cosy_flag = None
    try:
        M12, M16, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile=cosyFile,
            ta_y_offset=0.0, verbose=False, run_cosy_flag=run_cosy_flag)
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

    if ("NOMINAL", 0.0) not in completed:
        # Compute nominal sequentially, OUTSIDE the pool, so its COSY
        # compile can't race with anything -- this also guarantees a
        # fresh, correct "pg_June2025" pkl exists on disk before any
        # parallel dB-only (WF1) task tries to read-only from it below.
        print("Computing NOMINAL sequentially (bootstraps the shared pg_June2025 matrix)...")
        nominal_row = run_one_point(("NOMINAL", 0.0, [], False))
        if isinstance(nominal_row, tuple) and nominal_row[0] == "FAILED":
            raise RuntimeError(f"NOMINAL evaluation failed: {nominal_row[1]}")
        append_result(nominal_row)
        completed.add(("NOMINAL", 0.0))
        print(f"  NOMINAL done ({nominal_row['elapsed_s']:.0f}s)\n")

    tasks = []
    for knob_id, builder, values in KNOBS:
        is_shared = (knob_id == "WF1_Exit_dB")
        for v in values:
            v = float(v)
            if (knob_id, round(v, 8)) in completed:
                continue
            tasks.append((knob_id, v, builder(v), is_shared))

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
