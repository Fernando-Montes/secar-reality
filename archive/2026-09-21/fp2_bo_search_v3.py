"""
v3: adds WF1 Exit dB as a 4th BO knob alongside Q6/Q2/Q5 B_SC, motivated
by the 2026-08-10 decomposition finding: Q6 alone already gets loss=6.9
(vs. nominal's 930.7), and Q2+Q5 stacked on top only improve that to 4.6
(the current v2 best) -- a modest ~1.5x gain. But the user's own hand-tuned
Q6+WF1 Exit dB+TA_Y combo reaches loss=0.524, a ~13x improvement over Q6
alone, even though WF1 Exit dB and TA_Y are each individually
indistinguishable from nominal on their own (loss~840-890). That means
WF1 Exit dB is a much more effective FINE-TUNING knob once Q6 has done
the coarse correction than Q2/Q5 are -- worth adding to the BO's search
space before trying to characterize the Q6/Q2/Q5-only family's valley,
since that family may have a hard ceiling around loss~4-5.

TA_Y is deliberately NOT added here -- it's a beam/target-alignment offset
(an experimental setup parameter), a different category from the magnet
knobs the BO has searched so far, and we don't yet have clean data
isolating its effect from WF1's without TA_Y baked in. Can be added later
if WF1 alone doesn't close the gap.

Loss function, bounds for Q6/Q2/Q5, and all checkpointing/warm-start/pool
machinery reused unchanged from fp2_bo_search_v2.py. WF1 Exit dB bounds
set to +-2% -- double the +-1% range the OAT screen tested, and 8x the
user's own +-0.25% found value, for margin.

Run:
    cd AgentSearch
    python3 fp2_bo_search_v3.py
"""
import os
import csv
import time
import numpy as np

from skopt import Optimizer
from skopt.space import Real

from config import matrix_directory, pkl_directory
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn, TimedRestartingPool

# ------------------------------------------------------------------ search space
BOUNDS_BSC = 0.25
BOUNDS_DB = 0.02
DIMENSIONS = [Real(-BOUNDS_BSC, BOUNDS_BSC, name="Q6_B_SC"),
              Real(-BOUNDS_BSC, BOUNDS_BSC, name="Q2_B_SC"),
              Real(-BOUNDS_BSC, BOUNDS_BSC, name="Q5_B_SC"),
              Real(-BOUNDS_DB, BOUNDS_DB, name="WF1_Exit_dB")]

BATCH_SIZE = 10
NRAYS_GROUPS = 1000
NRAYS_WIDTH = 1000

RESULTS_FILE = "Results/fp2_bo_search_v3_results.csv"
FIELDNAMES = ["Q6_B_SC", "Q2_B_SC", "Q5_B_SC", "WF1_Exit_dB", "M12_sim", "M16_sim", "loss",
              "fp2_slits_std", "vd1638_std", "fp2_slits_mean", "vd1638_mean",
              "elapsed_s", "source"]

V2_RESULTS_FILE = "Results/fp2_bo_search_v2_results.csv"
OAT_RESULTS_FILE = "Results/fp2_oat_screen_results.csv"

# 743 (v2 history, WF1=0) + 10 (OAT WF1-only points, Q6=Q2=Q5=0) ~= 753
# seed points; budget ~200 fresh joint evaluations on top.
TARGET_TOTAL_EVALS = 953


# ----------------------------------------------------------------------- eval
def evaluate_point(x):
    Q6, Q2, Q5, WF1 = [float(v) for v in x]
    t0 = time.time()
    tag = f"{Q6:+.5f}_{Q2:+.5f}_{Q5:+.5f}_{WF1:+.5f}".replace(".", "p")
    cosyFile = f"pg_June2025_bo3_{tag}"
    try:
        tuneChanges = [['Q6', 'B_SC', Q6], ['Q2', 'B_SC', Q2], ['Q5', 'B_SC', Q5],
                        ['WF1 Exit', 'dB', WF1]]
        M12_sim, M16_sim, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile=cosyFile,
            ta_y_offset=0.0, verbose=False, run_cosy_flag=True)

        width = compute_fp2_width(
            tuneChanges=tuneChanges, nRays=NRAYS_WIDTH, cosyFile=usedFoxFile,
            run_cosy_flag=False, verbose=False)

        loss = loss_fn(M12_sim, M16_sim)
        elapsed = time.time() - t0

        if usedFoxFile != "pg_June2025":
            os.system(f"rm -rf {matrix_directory}/{usedFoxFile}")
            os.system(f"rm -rf {pkl_directory}/{usedFoxFile}.pkl")

        return {
            "Q6_B_SC": Q6, "Q2_B_SC": Q2, "Q5_B_SC": Q5, "WF1_Exit_dB": WF1,
            "M12_sim": M12_sim, "M16_sim": M16_sim, "loss": loss,
            "fp2_slits_std": width["fp2_std"], "vd1638_std": width["vd1638_std"],
            "fp2_slits_mean": width["fp2_mean"], "vd1638_mean": width["vd1638_mean"],
            "elapsed_s": elapsed, "source": "BO_v3",
        }
    except Exception as e:
        return ("FAILED", f"Q6={Q6},Q2={Q2},Q5={Q5},WF1={WF1}: {e}")


# ------------------------------------------------------------------ checkpoint
def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_history():
    rows = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                rows.append({k: (float(v) if k != "source" else v) for k, v in row.items()})
    return rows


def append_results(rows):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for r in rows:
            writer.writerow(r)
        f.flush()


def load_warm_start():
    """Two free sources, no new COSY runs needed:
    1. v2's full history (743 points, Q6/Q2/Q5 explored) with WF1_Exit_dB=0
       (none of those runs touched WF1) -- loss unchanged since it doesn't
       depend on WF1 directly, only through M12_sim/M16_sim already computed.
    2. OAT screen's WF1_Exit_dB one-knob-at-a-time rows (Q6=Q2=Q5=0),
       loss recomputed with loss_fn (OAT's own loss included a 4th
       tQ6-style term in general -- recompute to be safe/consistent).
    Neither source has explored the Q6+WF1 joint interaction yet -- that's
    what the fresh BO evaluations are for.
    """
    warm = []
    if os.path.exists(V2_RESULTS_FILE):
        with open(V2_RESULTS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                warm.append({
                    "Q6_B_SC": float(row["Q6_B_SC"]), "Q2_B_SC": float(row["Q2_B_SC"]),
                    "Q5_B_SC": float(row["Q5_B_SC"]), "WF1_Exit_dB": 0.0,
                    "M12_sim": float(row["M12_sim"]), "M16_sim": float(row["M16_sim"]),
                    "loss": float(row["loss"]),
                    "fp2_slits_std": float(row["fp2_slits_std"]), "vd1638_std": float(row["vd1638_std"]),
                    "fp2_slits_mean": float(row["fp2_slits_mean"]), "vd1638_mean": float(row["vd1638_mean"]),
                    "elapsed_s": float(row["elapsed_s"]), "source": "V2_history_seed",
                })
    else:
        print(f"Warning: {V2_RESULTS_FILE} not found, skipping that seed source.")

    if os.path.exists(OAT_RESULTS_FILE):
        with open(OAT_RESULTS_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["knob_id"] != "WF1_Exit_dB":
                    continue
                m12 = float(row["M12_sim"]); m16 = float(row["M16_sim"])
                warm.append({
                    "Q6_B_SC": 0.0, "Q2_B_SC": 0.0, "Q5_B_SC": 0.0,
                    "WF1_Exit_dB": float(row["value"]),
                    "M12_sim": m12, "M16_sim": m16, "loss": loss_fn(m12, m16),
                    "fp2_slits_std": float(row["fp2_slits_std"]), "vd1638_std": float(row["vd1638_std"]),
                    "fp2_slits_mean": float(row["fp2_slits_mean"]), "vd1638_mean": float(row["vd1638_mean"]),
                    "elapsed_s": float(row["elapsed_s"]), "source": "OAT_WF1_warm_start",
                })
    else:
        print(f"Warning: {OAT_RESULTS_FILE} not found, skipping that seed source.")

    return warm


if __name__ == "__main__":
    ensure_results_file_exists()
    history = load_history()

    if not history:
        warm = load_warm_start()
        if warm:
            append_results(warm)
            history = warm
            print(f"Warm-started with {len(warm)} points (v2 history @ WF1=0, OAT WF1-only scan).")

    opt = Optimizer(dimensions=DIMENSIONS, base_estimator="GP", acq_func="EI",
                     n_initial_points=10, random_state=42)

    if history:
        X_hist = [[r["Q6_B_SC"], r["Q2_B_SC"], r["Q5_B_SC"], r["WF1_Exit_dB"]] for r in history]
        y_hist = [r["loss"] for r in history]
        opt.tell(X_hist, y_hist)
        best = min(history, key=lambda r: r["loss"])
        print(f"Replayed {len(history)} historical points into the optimizer.")
        print(f"Current best: loss={best['loss']:.2f} at "
              f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% "
              f"Q5={best['Q5_B_SC']*100:+.2f}% WF1={best['WF1_Exit_dB']*100:+.3f}%\n")

    n_remaining = TARGET_TOTAL_EVALS - len(history)
    if n_remaining <= 0:
        print(f"Already have {len(history)} >= TARGET_TOTAL_EVALS={TARGET_TOTAL_EVALS}. "
              f"Raise TARGET_TOTAL_EVALS to keep going.")
    else:
        n_batches = int(np.ceil(n_remaining / BATCH_SIZE))
        print(f"{n_remaining} evaluations remaining -> {n_batches} batches of {BATCH_SIZE}.\n")

        with TimedRestartingPool(processes=10, timeout=600, maxtasksperchild=50) as pool:
            for b in range(n_batches):
                t_batch = time.time()
                x_batch = opt.ask(n_points=BATCH_SIZE, strategy="cl_min")
                results = pool.map(evaluate_point, x_batch)

                x_ok, y_ok, rows_ok = [], [], []
                for x, r in zip(x_batch, results):
                    if r is None:
                        print(f"  point {x} TIMED OUT, telling optimizer a penalty value")
                        x_ok.append(x); y_ok.append(1e6)
                    elif isinstance(r, tuple) and r[0] == "FAILED":
                        print(f"  point {x} FAILED ({r[1]}), telling optimizer a penalty value")
                        x_ok.append(x); y_ok.append(1e6)
                    elif not np.isfinite(r["loss"]):
                        print(f"  point {x} produced non-finite loss ({r['loss']}), treating as failed")
                        x_ok.append(x); y_ok.append(1e6)
                    else:
                        x_ok.append(x); y_ok.append(r["loss"]); rows_ok.append(r)

                if rows_ok:
                    append_results(rows_ok)
                opt.tell(x_ok, y_ok)

                best_idx = int(np.argmin(y_ok))
                print(f"Batch {b+1}/{n_batches} done in {time.time()-t_batch:.0f}s. "
                      f"Batch best loss={y_ok[best_idx]:.2f} at Q6={x_ok[best_idx][0]*100:+.2f}% "
                      f"Q2={x_ok[best_idx][1]*100:+.2f}% Q5={x_ok[best_idx][2]*100:+.2f}% "
                      f"WF1={x_ok[best_idx][3]*100:+.3f}%", flush=True)

        all_history = load_history()
        best = min(all_history, key=lambda r: r["loss"])
        print(f"\nDone. Overall best of {len(all_history)} points: loss={best['loss']:.2f} at "
              f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% "
              f"Q5={best['Q5_B_SC']*100:+.2f}% WF1={best['WF1_Exit_dB']*100:+.3f}%")
