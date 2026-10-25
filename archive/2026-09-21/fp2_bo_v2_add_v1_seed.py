"""
One-time merge: v1's 381 historical evaluations (fp2_bo_search_results.csv,
collected under the flawed 3-term loss with the Q6-absolute-value prior)
already tried many JOINT (Q6,Q2,Q5) combinations and recorded M12_sim/M16_sim
for each -- unlike the OAT warm start (1D lines through the origin only).

v2's own from-scratch run (fp2_bo_search_v2_results.csv, 184 points: 30 OAT
+ 154 BO) got stuck at loss=598 after its first batch and never improved,
while v1's data, re-scored under v2's correct 2-term loss (no new COSY runs
needed -- just recompute the formula on existing M12_sim/M16_sim), already
contains a loss=4.6 point. Merging it in gives the next v2 BO run a real
joint-combination seed to build a GP surrogate from, instead of only 1D
OAT lines that gave it no information about cross-knob interactions.

Idempotent: skips if v1 rows have already been merged in (checks the
source column).

Run:
    cd AgentSearch
    python3 fp2_bo_v2_add_v1_seed.py
"""
import csv
import os

from fp2_bo_search_v2 import loss_fn, RESULTS_FILE, FIELDNAMES

V1_RESULTS_FILE = "Results/fp2_bo_search_results.csv"


def load_v1_rescored():
    rows = []
    with open(V1_RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            m12 = float(row["M12_sim"])
            m16 = float(row["M16_sim"])
            rows.append({
                "Q6_B_SC": float(row["Q6_B_SC"]), "Q2_B_SC": float(row["Q2_B_SC"]),
                "Q5_B_SC": float(row["Q5_B_SC"]), "M12_sim": m12, "M16_sim": m16,
                "loss": loss_fn(m12, m16),
                "fp2_slits_std": float(row["fp2_slits_std"]), "vd1638_std": float(row["vd1638_std"]),
                "fp2_slits_mean": float(row["fp2_slits_mean"]), "vd1638_mean": float(row["vd1638_mean"]),
                "elapsed_s": float(row["elapsed_s"]), "source": "V1_rescored_seed",
            })
    return rows


if __name__ == "__main__":
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", newline="") as f:
            already_merged = any(row["source"] == "V1_rescored_seed" for row in csv.DictReader(f))
        if already_merged:
            print(f"{RESULTS_FILE} already contains V1_rescored_seed rows -- nothing to do.")
            raise SystemExit(0)
    else:
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    v1_rows = load_v1_rescored()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for r in v1_rows:
            writer.writerow(r)

    best = min(v1_rows, key=lambda r: r["loss"])
    print(f"Merged {len(v1_rows)} rescored v1 points into {RESULTS_FILE}.")
    print(f"Best among merged: loss={best['loss']:.2f} at Q6={best['Q6_B_SC']*100:+.2f}% "
          f"Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")
