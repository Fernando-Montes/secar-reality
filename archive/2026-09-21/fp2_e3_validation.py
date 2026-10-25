"""
The E3 validation check this whole search has been building toward:
E3 (real hardware, 14N6+ pilot beamtime) found that decreasing Q6 by
~0.88% from whatever the true (unknown) tune was minimized the beam spot
width AT VD1638 (the viewer actually used -- not the true FP2 Slits
position, 0.28m upstream, which was never directly observed in E3). This
is a statement about a LOCAL DERIVATIVE at the true tune, not an
absolute-value constraint on Q6 (see fp2_bo_search_v2.py's docstring for
the full reasoning on why the original BO's Q6-prior loss term was wrong).

CANDIDATE is the best E2-matching solution found so far (2026-08-10,
Gauss-Newton-refined near-nominal minimum-norm solution):
    Q6=-2.437%, Q2=-0.819%, Q5=+0.136%, WF1 Exit dB=+0.449%
    -> M12=-0.2467, M16=-0.3833 (targets -0.2472, -0.3842), loss=0.001

This scans Q6 around the candidate's own value (Q2/Q5/WF1 held fixed at
the candidate), and checks whether vd1638_std is minimized near
candidate_Q6 - 0.0088 (i.e. Q6 ~ -3.317%), independently verifying E3's
finding against this candidate rather than assuming it.

Checkpointed: each Q6 point is written to RESULTS_FILE as soon as it
completes, and already-completed points are skipped on restart -- safe to
Ctrl+C and rerun (2026-08-10, added after the first version collected
everything in memory and only wrote the CSV at the very end).

Run:
    cd AgentSearch
    python3 fp2_e3_validation.py
"""
import os
import csv
import numpy as np
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fp2_metrics import compute_fp2_width

CANDIDATE = {"Q6_B_SC": -0.02437, "Q2_B_SC": -0.00819, "Q5_B_SC": 0.00136, "WF1_Exit_dB": 0.00449}
E3_PREDICTED_SHIFT = -0.0088
NRAYS = 5000
N_POINTS = 26
Q6_RANGE = 0.025  # scan candidate_Q6 +- this, in fractional B_SC

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_e3_validation_results.csv")
FIELDNAMES = ["Q6_B_SC", "vd1638_mean", "vd1638_std", "fp2_slits_mean", "fp2_slits_std"]


def evaluate_point(q6):
    tuneChanges = [['Q6', 'B_SC', float(q6)], ['Q2', 'B_SC', CANDIDATE["Q2_B_SC"]],
                    ['Q5', 'B_SC', CANDIDATE["Q5_B_SC"]], ['WF1 Exit', 'dB', CANDIDATE["WF1_Exit_dB"]]]
    tag = f"{q6:+.5f}".replace(".", "p")
    cosyFile = f"pg_June2025_e3val_{tag}"
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


def make_plot(rows):
    rows = sorted(rows, key=lambda r: r["Q6_B_SC"])
    q6s = np.array([r["Q6_B_SC"] for r in rows])
    vd_std = np.array([r["vd1638_std"] for r in rows])
    fp2_std = np.array([r["fp2_slits_std"] for r in rows])

    q6_center = CANDIDATE["Q6_B_SC"]
    predicted_q6 = q6_center + E3_PREDICTED_SHIFT
    best_idx = int(np.argmin(vd_std))
    q6_at_min = q6s[best_idx]

    print(f"\nCandidate Q6 = {q6_center*100:+.3f}%")
    print(f"VD1638 std minimized at Q6 = {q6_at_min*100:+.3f}%  (shift = {(q6_at_min-q6_center)*100:+.3f}%)")
    print(f"E3-predicted minimum at Q6 = {predicted_q6*100:+.3f}%  (shift = {E3_PREDICTED_SHIFT*100:+.3f}%)")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(q6s * 100, vd_std, "o-", color="tab:blue", label="std(X) at VD1638 (measured quantity in E3)")
    ax.plot(q6s * 100, fp2_std, "s--", color="tab:gray", alpha=0.6, label="std(X) at true FP2 Slits (for reference)")
    ax.axvline(q6_center * 100, color="black", linestyle=":", label=f"candidate Q6 ({q6_center*100:+.2f}%)")
    ax.axvline(predicted_q6 * 100, color="tab:red", linestyle="--",
               label=f"E3-predicted min ({predicted_q6*100:+.2f}%)")
    ax.axvline(q6_at_min * 100, color="tab:green", linestyle="-",
               label=f"actual min found ({q6_at_min*100:+.2f}%)")
    ax.set_xlabel("Q6_B_SC [%]  (Q2/Q5/WF1 held fixed at candidate values)")
    ax.set_ylabel("std(X) [mm]")
    ax.set_title(f"E3 validation ({len(rows)}/{N_POINTS} points): "
                 f"does -0.88% Q6 from the candidate minimize the VD1638 spot?")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, "fp2_e3_validation.png")
    fig.savefig(outfile, dpi=200)
    print(f"wrote {outfile}")


if __name__ == "__main__":
    ensure_results_file_exists()
    completed = load_completed()

    q6_center = CANDIDATE["Q6_B_SC"]
    all_q6_values = np.linspace(q6_center - Q6_RANGE, q6_center + Q6_RANGE, N_POINTS)
    remaining = [float(v) for v in all_q6_values if round(float(v), 8) not in completed]

    print(f"{len(completed)}/{N_POINTS} points already done, {len(remaining)} remaining.\n")

    if remaining:
        with Pool(processes=10) as pool:
            for row in pool.imap_unordered(evaluate_point, remaining):
                append_result(row)
                print(f"  Q6={row['Q6_B_SC']*100:+.3f}%  vd1638_std={row['vd1638_std']:.3f} mm", flush=True)

    make_plot(load_all_results())
