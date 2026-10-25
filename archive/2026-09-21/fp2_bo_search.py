"""
Batch Gaussian-process Bayesian optimization over the 3 knobs identified
as non-redundant by the OAT-screen degeneracy check (2026-08-08):
    Q6_B_SC, Q2_B_SC, Q5_B_SC   -- each in [-0.25, +0.25] (symmetric,
    user-confirmed, generous margin around each knob's individually-found
    minimum: Q6 ~-3.3%, Q2 ~-6.7%, Q5 still trending past +10%).

Uses scikit-optimize's ask/tell Optimizer interface with batch proposals
(strategy="cl_min", the "constant liar" batch heuristic) so all 10
workers stay busy each round, rather than one point at a time.

Warm-started for free from fp2_oat_screen_results.csv: the OAT screen
already evaluated each of these 3 knobs individually (others at zero),
all within the new bounds. Re-uses the stored M12_sim/M16_sim (physics
doesn't change) but RECOMPUTES the loss with the correct Q6 value in the
prior term (the OAT screen used Q6=0.0 in that term for every knob,
which was fine for single-knob ranking but wrong for the joint loss).

Robustness borrowed from ViewerAnalysis.py's optimizeMCMC: the
TimedRestartingPool class (per-task timeout + periodic worker restart).

Checkpointing: every evaluated point (warm-start and BO-proposed alike)
is appended to fp2_bo_search_results.csv. On restart, ALL history is
replayed into a fresh Optimizer via bulk tell() -- simpler and more
transparent than pickling optimizer state, at the cost of a fast one-time
replay at startup.

Run:
    cd AgentSearch
    python3 fp2_bo_search.py
"""
import os
import csv
import time
import numpy as np
import multiprocessing as mp

from skopt import Optimizer
from skopt.space import Real

from config import matrix_directory, pkl_directory
from fp2_metrics import compute_M12_M16, compute_fp2_width

# ---------------------------------------------------------------- targets
M12_TARGET = -0.2472
M16_TARGET = -0.3842
Q6_E3_PRIOR = -0.005
SIGMA_M12 = 0.02
SIGMA_M16 = 0.03
SIGMA_Q6 = 0.0015


def loss_fn(M12_sim, M16_sim, Q6):
    t12 = ((M12_sim - M12_TARGET) / SIGMA_M12) ** 2
    t16 = ((M16_sim - M16_TARGET) / SIGMA_M16) ** 2
    tQ6 = ((Q6 - Q6_E3_PRIOR) / SIGMA_Q6) ** 2
    return t12 + t16 + tQ6


# ------------------------------------------------------------------ search space
BOUNDS = 0.25
DIMENSIONS = [Real(-BOUNDS, BOUNDS, name="Q6_B_SC"),
              Real(-BOUNDS, BOUNDS, name="Q2_B_SC"),
              Real(-BOUNDS, BOUNDS, name="Q5_B_SC")]

BATCH_SIZE = 10
TARGET_TOTAL_EVALS = 400   # includes warm-start points; raised 2026-08-08 -- first 200 (192 actual)
                            # were still improving near the end, not plateaued. Extend and re-run
                            # again later to keep going if this batch is still improving too.
NRAYS_GROUPS = 1000
NRAYS_WIDTH = 1000

RESULTS_FILE = "Results/fp2_bo_search_results.csv"
FIELDNAMES = ["Q6_B_SC", "Q2_B_SC", "Q5_B_SC", "M12_sim", "M16_sim", "loss",
              "fp2_slits_std", "vd1638_std", "fp2_slits_mean", "vd1638_mean",
              "elapsed_s", "source"]
OAT_RESULTS_FILE = "Results/fp2_oat_screen_results.csv"  # moved here 2026-08-08, OAT screen complete


# -------------------------------------------------- robust pool (adapted from ViewerAnalysis.py)
class TimedRestartingPool:
    """
    Drop-in multiprocessing.Pool wrapper: per-task timeout, and periodic
    worker restarts (maxtasksperchild) to avoid memory creep over a long
    run. Adapted from ViewerAnalysis.py's optimizeMCMC infrastructure.
    """
    def __init__(self, processes=None, timeout=600, maxtasksperchild=50, start_method="spawn"):
        self.timeout = timeout
        self._ctx = mp.get_context(start_method)
        self._pool = self._ctx.Pool(processes=processes, maxtasksperchild=maxtasksperchild)

    def map(self, func, iterable):
        asyncs = [self._pool.apply_async(func, (x,)) for x in iterable]
        out = []
        for a in asyncs:
            try:
                out.append(a.get(timeout=self.timeout))
            except mp.TimeoutError:
                out.append(None)
            except Exception as e:
                out.append(("FAILED", str(e)))
        return out

    def close(self):
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ----------------------------------------------------------------------- eval
def evaluate_point(x):
    """x = [Q6_B_SC, Q2_B_SC, Q5_B_SC]. Returns a result dict or a
    ('FAILED', reason) tuple -- never raises, so a bad point can't kill
    the batch."""
    Q6, Q2, Q5 = [float(v) for v in x]
    t0 = time.time()
    tag = f"{Q6:+.5f}_{Q2:+.5f}_{Q5:+.5f}".replace(".", "p")
    cosyFile = f"pg_June2025_bo_{tag}"
    try:
        tuneChanges = [['Q6', 'B_SC', Q6], ['Q2', 'B_SC', Q2], ['Q5', 'B_SC', Q5]]
        M12_sim, M16_sim, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS_GROUPS, cosyFile=cosyFile,
            ta_y_offset=0.0, verbose=False, run_cosy_flag=True)

        width = compute_fp2_width(
            tuneChanges=tuneChanges, nRays=NRAYS_WIDTH, cosyFile=usedFoxFile,
            run_cosy_flag=False, verbose=False)

        loss = loss_fn(M12_sim, M16_sim, Q6=Q6)
        elapsed = time.time() - t0

        if usedFoxFile != "pg_June2025":
            os.system(f"rm -rf {matrix_directory}/{usedFoxFile}")
            os.system(f"rm -rf {pkl_directory}/{usedFoxFile}.pkl")

        return {
            "Q6_B_SC": Q6, "Q2_B_SC": Q2, "Q5_B_SC": Q5,
            "M12_sim": M12_sim, "M16_sim": M16_sim, "loss": loss,
            "fp2_slits_std": width["fp2_std"], "vd1638_std": width["vd1638_std"],
            "fp2_slits_mean": width["fp2_mean"], "vd1638_mean": width["vd1638_mean"],
            "elapsed_s": elapsed, "source": "BO",
        }
    except Exception as e:
        return ("FAILED", f"Q6={Q6},Q2={Q2},Q5={Q5}: {e}")


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
    """Pull Q6_B_SC/Q2_B_SC/Q5_B_SC single-knob rows from the OAT screen,
    recompute the loss correctly (real Q6 value in the prior term, not 0)."""
    if not os.path.exists(OAT_RESULTS_FILE):
        print(f"Warning: {OAT_RESULTS_FILE} not found, skipping warm start.")
        return []
    warm = []
    knob_to_col = {"Q6_B_SC": "Q6_B_SC", "Q2_B_SC": "Q2_B_SC", "Q5_B_SC": "Q5_B_SC"}
    with open(OAT_RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["knob_id"] not in knob_to_col:
                continue
            value = float(row["value"])
            M12_sim = float(row["M12_sim"])
            M16_sim = float(row["M16_sim"])
            Q6 = value if row["knob_id"] == "Q6_B_SC" else 0.0
            Q2 = value if row["knob_id"] == "Q2_B_SC" else 0.0
            Q5 = value if row["knob_id"] == "Q5_B_SC" else 0.0
            loss = loss_fn(M12_sim, M16_sim, Q6=Q6)
            warm.append({
                "Q6_B_SC": Q6, "Q2_B_SC": Q2, "Q5_B_SC": Q5,
                "M12_sim": M12_sim, "M16_sim": M16_sim, "loss": loss,
                "fp2_slits_std": float(row["fp2_slits_std"]), "vd1638_std": float(row["vd1638_std"]),
                "fp2_slits_mean": float(row["fp2_slits_mean"]), "vd1638_mean": float(row["vd1638_mean"]),
                "elapsed_s": float(row["elapsed_s"]), "source": "OAT_warm_start",
            })
    return warm


if __name__ == "__main__":
    ensure_results_file_exists()
    history = load_history()

    if not history:
        warm = load_warm_start()
        if warm:
            append_results(warm)
            history = warm
            print(f"Warm-started with {len(warm)} points from {OAT_RESULTS_FILE} (loss recomputed).")

    opt = Optimizer(dimensions=DIMENSIONS, base_estimator="GP", acq_func="EI",
                     n_initial_points=10, random_state=42)

    if history:
        X_hist = [[r["Q6_B_SC"], r["Q2_B_SC"], r["Q5_B_SC"]] for r in history]
        y_hist = [r["loss"] for r in history]
        opt.tell(X_hist, y_hist)
        best = min(history, key=lambda r: r["loss"])
        print(f"Replayed {len(history)} historical points into the optimizer.")
        print(f"Current best: loss={best['loss']:.2f} at "
              f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%\n")

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
                        # Defensive second layer: fp2_metrics.py's empty-ray
                        # checks should prevent NaN reaching here at all
                        # (2026-08-08 fix), but if a NaN/inf ever slips
                        # through some other path, treat it the same way
                        # rather than crashing opt.tell() again.
                        print(f"  point {x} produced non-finite loss ({r['loss']}), treating as failed")
                        x_ok.append(x); y_ok.append(1e6)
                    else:
                        x_ok.append(x); y_ok.append(r["loss"]); rows_ok.append(r)

                # Write to disk BEFORE telling the optimizer -- if tell()
                # ever throws again, this batch's real (expensive) results
                # aren't lost, only the optimizer's internal state update.
                if rows_ok:
                    append_results(rows_ok)
                opt.tell(x_ok, y_ok)

                best_idx = int(np.argmin(y_ok))
                print(f"Batch {b+1}/{n_batches} done in {time.time()-t_batch:.0f}s. "
                      f"Batch best loss={y_ok[best_idx]:.2f} at Q6={x_ok[best_idx][0]*100:+.2f}% "
                      f"Q2={x_ok[best_idx][1]*100:+.2f}% Q5={x_ok[best_idx][2]*100:+.2f}%", flush=True)

        all_history = load_history()
        best = min(all_history, key=lambda r: r["loss"])
        print(f"\nDone. Overall best of {len(all_history)} points: loss={best['loss']:.2f} at "
              f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")
