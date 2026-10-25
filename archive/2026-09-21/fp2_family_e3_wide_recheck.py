"""
v7 (2026-08-14): now against fp2_family_mc_v7_results.csv -- the 12-knob
family search (Q1-Q7 B_SC + WF1/B1/B2/B3/B4 Exit dB) using the weighted
+ ridge-regularized solve (fixed the M12-sign-flipping bug and the dB
knobs' outsized leverage that stalled v3-v6). 46 E2-passing candidates,
best loss=0.81 (vs. v1's 1.45).

Redoes the E3 test properly for all E2-passing candidates, using a WIDE
Q6 scan (same design validated in fp2_e3_check_mc_candidate.py: +-4%,
enough points to actually bracket the true minimum) instead of the
narrow 7-point window embedded in the MC run's own in-loop check, which
was confirmed to extrapolate wildly (see
e3_fitted_shift_narrow7pt_UNRELIABLE column).

For each candidate: holds Q1/Q2/Q4/Q5/Q7/WF1 fixed at that candidate's
own values, scans Q6 over a wide range, fits a parabola, and reports
whether the fitted vertex actually falls INSIDE the scanned range
(trustworthy) or still extrapolates outside it (would need an even wider
scan -- flagged, not silently trusted).

Two output files:
  - fp2_family_e3_wide_raw.csv: every (iteration, Q6) point evaluated,
    checkpointed/resumable the same way as fp2_e3_validation.py.
  - fp2_family_e3_wide_summary.csv: one row per candidate once all its
    points are done -- candidate_Q6, q6_at_true_min, true_shift_wide_scan
    (the number to trust), in_scanned_range, passes_e3.

Run:
    cd AgentSearch
    python3 fp2_family_e3_wide_recheck.py
"""
import os
import csv
import numpy as np
from multiprocessing import Pool

from fp2_metrics import compute_fp2_width

MC_RESULTS_FILE = "Results/fp2_family_mc_v7_results.csv"
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
E3_SHIFT_CENTER = -0.0088
E3_TOLERANCE = 0.004

KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
         'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
NRAYS = 5000
N_POINTS = 21
Q6_WIDE_RANGE = 0.04  # +-4%, same as the validated check scan

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_FILE = os.path.join(OUTPUT_DIR, "fp2_family_e3_wide_raw_v7.csv")
RAW_FIELDNAMES = ["iteration", "Q6_B_SC", "vd1638_mean", "vd1638_std", "fp2_slits_mean", "fp2_slits_std"]
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "fp2_family_e3_wide_summary_v7.csv")
SUMMARY_FIELDNAMES = ["iteration", "candidate_Q6", "q6_at_true_min", "true_shift_wide_scan",
                       "in_scanned_range", "passes_e3"]


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def load_e2_passing_candidates():
    rows = list(csv.DictReader(open(MC_RESULTS_FILE)))
    candidates = {}
    for r in rows:
        if r["loss"] in (None, "") or float(r["loss"]) > CHI2_THRESHOLD_2SIGMA_DF4:
            continue
        iteration = int(r["iteration"])
        candidates[iteration] = {k: float(r[k]) for k in KNOBS}
    return candidates


def evaluate_point(task):
    iteration, other_knobs, q6 = task
    tuneChanges = [knob_to_tunechange(k, v) for k, v in other_knobs.items() if k != 'Q6_B_SC']
    tuneChanges.append(['Q6', 'B_SC', float(q6)])
    tag = f"fam{iteration}_e3w_{q6:+.5f}".replace(".", "p")
    cosyFile = tag[:60]
    try:
        width = compute_fp2_width(tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=cosyFile,
                                    run_cosy_flag=True, verbose=False)
        return {"iteration": iteration, "Q6_B_SC": q6, "vd1638_mean": width["vd1638_mean"],
                "vd1638_std": width["vd1638_std"], "fp2_slits_mean": width["fp2_mean"],
                "fp2_slits_std": width["fp2_std"]}
    except Exception as e:
        return ("FAILED", iteration, q6, str(e))


def ensure_files_exist():
    if not os.path.exists(RAW_FILE):
        with open(RAW_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=RAW_FIELDNAMES).writeheader()
    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES).writeheader()


def load_raw_completed():
    completed = set()
    with open(RAW_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            completed.add((int(row["iteration"]), round(float(row["Q6_B_SC"]), 8)))
    return completed


def load_summary_done():
    with open(SUMMARY_FILE, "r", newline="") as f:
        return set(int(row["iteration"]) for row in csv.DictReader(f))


def append_row(path, fieldnames, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        f.flush()


def load_all_raw():
    rows = []
    with open(RAW_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k != "iteration"} | {"iteration": int(row["iteration"])})
    return rows


if __name__ == "__main__":
    ensure_files_exist()
    candidates = load_e2_passing_candidates()
    raw_completed = load_raw_completed()
    summary_done = load_summary_done()
    print(f"{len(candidates)} E2-passing candidates. {len(summary_done)} already summarized.\n")

    tasks = []
    for iteration, knobs in candidates.items():
        if iteration in summary_done:
            continue
        q6_center = knobs['Q6_B_SC']
        q6_scan = np.linspace(q6_center - Q6_WIDE_RANGE, q6_center + Q6_WIDE_RANGE, N_POINTS)
        for q6v in q6_scan:
            if (iteration, round(float(q6v), 8)) in raw_completed:
                continue
            tasks.append((iteration, knobs, float(q6v)))

    print(f"{len(tasks)} points remaining to evaluate.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for i, result in enumerate(pool.imap_unordered(evaluate_point, tasks)):
                if isinstance(result, tuple) and result[0] == "FAILED":
                    print(f"  FAILED iter={result[1]} Q6={result[2]}: {result[3]}", flush=True)
                    continue
                append_row(RAW_FILE, RAW_FIELDNAMES, result)
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(tasks)} points done...", flush=True)

    # ---- summarize any candidate whose points are all now present ----
    all_raw = load_all_raw()
    by_iter = {}
    for r in all_raw:
        by_iter.setdefault(r["iteration"], []).append(r)

    summary_done = load_summary_done()
    for iteration, knobs in candidates.items():
        if iteration in summary_done:
            continue
        pts = by_iter.get(iteration, [])
        q6_center = knobs['Q6_B_SC']
        expected_n = N_POINTS
        if len(pts) < expected_n:
            continue  # not all points in yet (some may have failed) -- leave for a future run
        pts.sort(key=lambda r: r["Q6_B_SC"])
        q6s = np.array([r["Q6_B_SC"] for r in pts])
        stds = np.array([r["vd1638_std"] for r in pts])
        a, b, c = np.polyfit(q6s, stds, 2)
        if a <= 0:
            q6_at_min, shift, in_range = None, None, False
        else:
            q6_at_min = -b / (2 * a)
            shift = q6_at_min - q6_center
            in_range = q6s.min() <= q6_at_min <= q6s.max()
        passes = (shift is not None) and abs(shift - E3_SHIFT_CENTER) <= E3_TOLERANCE
        append_row(SUMMARY_FILE, SUMMARY_FIELDNAMES, {
            "iteration": iteration, "candidate_Q6": q6_center,
            "q6_at_true_min": q6_at_min if q6_at_min is not None else "",
            "true_shift_wide_scan": shift if shift is not None else "",
            "in_scanned_range": in_range, "passes_e3": passes,
        })
        shift_str = f"{shift*100:+.3f}%" if shift is not None else "no interior min"
        print(f"  iter {iteration}: true_shift={shift_str}  in_range={in_range}  passes_e3={passes}", flush=True)

    print(f"\nDone. See {SUMMARY_FILE} for the per-candidate results.")
