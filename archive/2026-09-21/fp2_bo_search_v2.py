"""
v2 of fp2_bo_search.py, corrected 2026-08-09 (user-identified physics
error in v1): the loss no longer has a term anchoring Q6's ABSOLUTE
value to an E3-derived number. E3's finding (real Q6 lowered by ~0.88%
minimizes the VD1638 viewer spot) is a statement about a LOCAL
DERIVATIVE at whatever the true (unknown) tune actually is -- not a
constraint that Q6's correct value equals "nominal - 0.88%", since the
true tune could differ from our simulated nominal in other quads too
(Q2, Q5, ...), not just Q6. Baking it into the loss as an absolute-value
prior was mixing up a gradient condition with a value constraint.

Loss is now just the two real E2-measured targets:
    loss = ((M12_sim - M12_TARGET)/SIGMA_M12)^2 + ((M16_sim - M16_TARGET)/SIGMA_M16)^2

E3 is instead used as an independent, out-of-sample VALIDATION check
after the fact (see fp2_e3_validation.py, built once this search has a
best candidate): hold Q2/Q5 fixed at the winning candidate's values
(matching E3's real procedure of varying only Q6 from "whatever the
current tune is"), scan Q6 around the candidate's own value, and check
whether the VD1638 (not "true FP2 Slits" -- VD1638 is what was actually
observed in the real E3 measurement) spot width minimizes at roughly
candidate_Q6 - 0.88%.

Same 3 knobs (Q6/Q2/Q5) and bounds as v1 -- the degeneracy analysis that
picked them was about M12/M16 response directions only, independent of
this Q6-prior question. Same batch-BO/warm-start/checkpointing
machinery as v1, writing to a SEPARATE results file (v1's loss
definition is incompatible, shouldn't be mixed with v2's).

Run:
    cd AgentSearch
    python3 fp2_bo_search_v2.py
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
SIGMA_M12 = 0.02
SIGMA_M16 = 0.03


def loss_fn(M12_sim, M16_sim):
    t12 = ((M12_sim - M12_TARGET) / SIGMA_M12) ** 2
    t16 = ((M16_sim - M16_TARGET) / SIGMA_M16) ** 2
    return t12 + t16


# ------------------------------------------------------------------ search space
BOUNDS = 0.25
DIMENSIONS = [Real(-BOUNDS, BOUNDS, name="Q6_B_SC"),
              Real(-BOUNDS, BOUNDS, name="Q2_B_SC"),
              Real(-BOUNDS, BOUNDS, name="Q5_B_SC")]

BATCH_SIZE = 10
# History now includes 381 rescored points from v1 (real joint-combination
# data, unlike the OAT warm start's 1D lines) merged in via
# fp2_bo_v2_add_v1_seed.py -- 2026-08-10, after the from-scratch OAT-only
# warm start got stuck at loss=598 for its entire 154-eval run and never
# found the loss=4.6 region that was sitting in v1's own history the whole
# time. Budget raised to explore ~200 fresh points around that better seed.
TARGET_TOTAL_EVALS = 765
NRAYS_GROUPS = 1000
NRAYS_WIDTH = 1000

RESULTS_FILE = "Results/fp2_bo_search_v2_results.csv"
FIELDNAMES = ["Q6_B_SC", "Q2_B_SC", "Q5_B_SC", "M12_sim", "M16_sim", "loss",
              "fp2_slits_std", "vd1638_std", "fp2_slits_mean", "vd1638_mean",
              "elapsed_s", "source"]
OAT_RESULTS_FILE = "Results/fp2_oat_screen_results.csv"


# -------------------------------------------------- robust pool (adapted from ViewerAnalysis.py)
class TimedRestartingPool:
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
    Q6, Q2, Q5 = [float(v) for v in x]
    t0 = time.time()
    tag = f"{Q6:+.5f}_{Q2:+.5f}_{Q5:+.5f}".replace(".", "p")
    cosyFile = f"pg_June2025_bo2_{tag}"
    try:
        tuneChanges = [['Q6', 'B_SC', Q6], ['Q2', 'B_SC', Q2], ['Q5', 'B_SC', Q5]]
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
            "Q6_B_SC": Q6, "Q2_B_SC": Q2, "Q5_B_SC": Q5,
            "M12_sim": M12_sim, "M16_sim": M16_sim, "loss": loss,
            "fp2_slits_std": width["fp2_std"], "vd1638_std": width["vd1638_std"],
            "fp2_slits_mean": width["fp2_mean"], "vd1638_mean": width["vd1638_mean"],
            "elapsed_s": elapsed, "source": "BO_v2",
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
    """Same free warm start as v1 (OAT screen's individual Q6/Q2/Q5
    rows), loss recomputed with the v2 (2-term) formula this time."""
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
            loss = loss_fn(M12_sim, M16_sim)
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
            print(f"Warm-started with {len(warm)} points from {OAT_RESULTS_FILE} (v2 loss, no Q6 prior).")

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
                      f"Q2={x_ok[best_idx][1]*100:+.2f}% Q5={x_ok[best_idx][2]*100:+.2f}%", flush=True)

        all_history = load_history()
        best = min(all_history, key=lambda r: r["loss"])
        print(f"\nDone. Overall best of {len(all_history)} points: loss={best['loss']:.2f} at "
              f"Q6={best['Q6_B_SC']*100:+.2f}% Q2={best['Q2_B_SC']*100:+.2f}% Q5={best['Q5_B_SC']*100:+.2f}%")
