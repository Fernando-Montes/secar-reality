"""
Sanity check on the family MC's E3 test (2026-08-11): picks the
lowest-loss E2-passed/E3-failed candidate from fp2_family_mc_results.csv
(iteration 852, loss=1.446, fitted shift=+12.35% -- wildly outside the
narrow 7-point window the MC actually scanned) and does a WIDE Q6 scan
around it (+-4%, 26 points, same style as fp2_e3_validation.py), holding
Q1/Q2/Q4/Q5/Q7/WF1 fixed at the candidate's own values. Overlays the 7
actual narrow-window points the MC used, so we can see exactly how they
sit relative to the true curve and whether a parabola fit to just those
7 could plausibly produce that +12.35% extrapolation.

Checkpointed the same way as fp2_e3_validation.py.

Run:
    cd AgentSearch
    python3 fp2_e3_check_mc_candidate.py
"""
import os
import csv
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fp2_metrics import compute_fp2_width

CANDIDATE = {
    "Q1_B_SC": 0.018528554508350302, "Q2_B_SC": -0.04857068567701506,
    "Q4_B_SC": -0.019222471579555214, "Q5_B_SC": 0.003522874235785041,
    "Q6_B_SC": -0.0074535085342535175, "Q7_B_SC": -0.0008618780307185561,
    "WF1_Exit_dB": 0.0025524283094756958,
}
MC_FITTED_SHIFT = 0.12354460396677927
E3_SHIFT_CENTER = -0.0088
E3_SCAN_HALFWIDTH = 0.01  # the MC's own narrow window, for overlay reference
E3_N_POINTS_MC = 7

NRAYS = 5000
N_POINTS = 26
Q6_WIDE_RANGE = 0.04  # +-4%, wide scan for this check

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_e3_check_mc_candidate_results.csv")
FIELDNAMES = ["Q6_B_SC", "vd1638_mean", "vd1638_std", "fp2_slits_mean", "fp2_slits_std"]

OTHER_KNOBS = {k: v for k, v in CANDIDATE.items() if k != "Q6_B_SC"}


def knob_to_tunechange(knob_id, value):
    if knob_id == 'WF1_Exit_dB':
        return ['WF1 Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def evaluate_point(q6):
    tuneChanges = [knob_to_tunechange(k, v) for k, v in OTHER_KNOBS.items()]
    tuneChanges.append(['Q6', 'B_SC', float(q6)])
    tag = f"{q6:+.5f}".replace(".", "p")
    cosyFile = f"pg_June2025_e3check_{tag}"
    width = compute_fp2_width(tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=cosyFile,
                                run_cosy_flag=True, verbose=False)
    return {"Q6_B_SC": q6, "vd1638_mean": width["vd1638_mean"], "vd1638_std": width["vd1638_std"],
            "fp2_slits_mean": width["fp2_mean"], "fp2_slits_std": width["fp2_std"]}


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_completed():
    completed = set()
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            completed.add(round(float(row["Q6_B_SC"]), 8))
    return completed


def append_result(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def load_all_results():
    rows = []
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


if __name__ == "__main__":
    ensure_results_file_exists()
    completed = load_completed()

    q6_center = CANDIDATE["Q6_B_SC"]
    all_q6 = np.linspace(q6_center - Q6_WIDE_RANGE, q6_center + Q6_WIDE_RANGE, N_POINTS)
    remaining = [float(v) for v in all_q6 if round(float(v), 8) not in completed]
    print(f"{len(completed)}/{N_POINTS} points already done, {len(remaining)} remaining.\n")

    if remaining:
        with Pool(processes=10) as pool:
            for row in pool.imap_unordered(evaluate_point, remaining):
                append_result(row)
                print(f"  Q6={row['Q6_B_SC']*100:+.3f}%  vd1638_std={row['vd1638_std']:.3f} mm", flush=True)

    rows = sorted(load_all_results(), key=lambda r: r["Q6_B_SC"])
    q6s = np.array([r["Q6_B_SC"] for r in rows])
    vd_std = np.array([r["vd1638_std"] for r in rows])
    fp2_std = np.array([r["fp2_slits_std"] for r in rows])

    e3_predicted_q6 = q6_center + E3_SHIFT_CENTER
    mc_fitted_q6 = q6_center + MC_FITTED_SHIFT
    narrow_center = q6_center + E3_SHIFT_CENTER
    narrow_q6s = np.linspace(narrow_center - E3_SCAN_HALFWIDTH, narrow_center + E3_SCAN_HALFWIDTH, E3_N_POINTS_MC)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(q6s * 100, vd_std, "o-", color="tab:blue", label="std(X) at VD1638 (wide check scan)")
    ax.plot(q6s * 100, fp2_std, "s--", color="tab:gray", alpha=0.5, label="std(X) at true FP2 Slits (reference)")

    # mark the MC's own 7 narrow-window points by interpolating onto the wide curve for display
    narrow_vd_interp = np.interp(narrow_q6s, q6s, vd_std)
    ax.plot(narrow_q6s * 100, narrow_vd_interp, "D", color="orange", markersize=10, zorder=5,
            label="MC's actual 7 narrow-window points")

    ax.axvline(q6_center * 100, color="black", linestyle=":", label=f"candidate Q6 ({q6_center*100:+.2f}%)")
    ax.axvline(e3_predicted_q6 * 100, color="tab:red", linestyle="--",
               label=f"E3-predicted min ({e3_predicted_q6*100:+.2f}%)")
    ax.axvline(mc_fitted_q6 * 100, color="tab:purple", linestyle="-.",
               label=f"MC's fitted min ({mc_fitted_q6*100:+.2f}%, shift={MC_FITTED_SHIFT*100:+.2f}%)")
    if len(q6s) > 0:
        true_min_idx = int(np.argmin(vd_std))
        ax.axvline(q6s[true_min_idx] * 100, color="tab:green", linestyle="-",
                   label=f"true min in wide scan ({q6s[true_min_idx]*100:+.2f}%)")

    ax.set_xlabel("Q6_B_SC [%]  (Q1/Q2/Q4/Q5/Q7/WF1 held fixed at candidate values)")
    ax.set_ylabel("std(X) [mm]")
    ax.set_title(f"MC candidate (iter 852, loss=1.45): wide check vs. the narrow 7-point E3 fit")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_e3_check_mc_candidate.png")
    fig.savefig(outfile, dpi=200)
    print(f"\nwrote {outfile}")
