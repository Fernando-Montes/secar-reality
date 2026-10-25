"""
Restricted-knob family search (2026-08-14), per user request: same
Monte Carlo methodology as fp2_family_mc.py (v7) -- sample plausible
"true" target residuals from the E2 fit uncertainty, solve a weighted +
ridge-regularized correction, verify with real COSY, accept if loss <=
the chi-square(df=4, 2-sigma) threshold (9.716) -- but restricted to
ONLY 3 knobs: Q3_B_SC, Q4_B_SC, WF1_Exit_dB.

With 3 free knobs and 4 targets (M12, M16, Offset12, Offset16) this is
now an OVERDETERMINED system -- there is no guarantee an exact solution
exists, let alone one within the acceptance region. The same weighted
ridge normal-equations solve (Jw^T Jw + Gamma)^-1 Jw^T rw is still valid
here (it's ordinary regularized least squares regardless of whether the
system is under- or over-determined); ridge/trust-range regularization
is kept for consistency with v7 (and because Q3's leverage on M16 is
still very steep -- same reason it was excluded from v1 in the first
place, see fp2_family_mc.py's v2 docstring).

The Jacobian is NOT recomputed from scratch -- it reuses the exact same
cached OAT/offset-Jacobian data (Results/fp2_offset_jacobian_scan_results.csv)
that v7's build_jacobian() reads, just sliced down to these 3 knobs'
columns. No new sensitivity-scan COSY runs needed.

Run:
    cd AgentSearch
    python3 fp2_family_mc_q3q4wf1.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp2_metrics import compute_M12_M16, DALPHAX_GROUPS, DE_GROUPS

# ---------------------------------------------------------------- targets / acceptance sigmas
M12_TARGET, M16_TARGET = -0.2472, -0.3842
OFFSET12_TARGET, OFFSET16_TARGET = 1.543, 0.537
SIGMA_M12, SIGMA_M16 = 0.02, 0.03
SIGMA_OFFSET12, SIGMA_OFFSET16 = 0.2, 1.5
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

# nominal -> target residual (same 4 terms, order M12/M16/Offset12/Offset16)
BASE_RESIDUAL = np.array([-0.4562, -0.5995, 3.2057, 2.0769])

# sampling (experimental-fit) uncertainties -- same as v7
SAMPLE_SIGMA_M12 = 0.0269
SAMPLE_SIGMA_M16 = 0.2961
SAMPLE_SIGMA_OFFSET12 = 0.3566
SAMPLE_SIGMA_OFFSET16 = 0.925
SAMPLE_SIGMAS = np.array([SAMPLE_SIGMA_M12, SAMPLE_SIGMA_M16, SAMPLE_SIGMA_OFFSET12, SAMPLE_SIGMA_OFFSET16])

NORM_CAP = 0.10
BASE_SEED = 20260814

TRUST_RANGE_BSC = 0.10
TRUST_RANGE_WF1 = 0.02

KNOBS = ['Q3_B_SC', 'Q4_B_SC', 'WF1_Exit_dB']
OAT_RESULTS_FILE = "Results/fp2_offset_jacobian_scan_results.csv"

NRAYS_E2 = 3000
TARGET_E2_PASSED = 45
BATCH_SIZE = 20

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_family_mc_q3q4wf1_results.csv")
FIELDNAMES = (["iteration", "status"] + KNOBS +
              ["combined_norm", "M12_sim", "M16_sim", "Offset12_sim", "Offset16_sim",
               "loss", "elapsed_s"])


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def build_jacobian():
    rows = list(csv.DictReader(open(OAT_RESULTS_FILE)))

    def fit_slope_offset(row, groups_dict):
        xs = np.array([groups_dict[g] for g in groups_dict])
        ys = np.array([float(row[f'group_{g}_mean_x']) for g in groups_dict])
        return np.polyfit(xs, ys, 1)  # slope, intercept

    nominal = [r for r in rows if r['knob_id'] == 'NOMINAL'][0]
    M12_0, Off12_0 = fit_slope_offset(nominal, DALPHAX_GROUPS)
    M16_0, Off16_0 = fit_slope_offset(nominal, DE_GROUPS)

    J_cols = []
    for k in KNOBS:
        pts = [r for r in rows if r['knob_id'] == k]
        vals = np.array([float(p['value']) for p in pts])
        s_M12, s_Off12 = np.polyfit(vals, [fit_slope_offset(p, DALPHAX_GROUPS)[0] for p in pts], 1)[0], \
                          np.polyfit(vals, [fit_slope_offset(p, DALPHAX_GROUPS)[1] for p in pts], 1)[0]
        s_M16, s_Off16 = np.polyfit(vals, [fit_slope_offset(p, DE_GROUPS)[0] for p in pts], 1)[0], \
                          np.polyfit(vals, [fit_slope_offset(p, DE_GROUPS)[1] for p in pts], 1)[0]
        J_cols.append([s_M12, s_M16, s_Off12, s_Off16])
    J = np.array(J_cols).T  # 4 x 3
    return J, np.array([M12_0, M16_0, Off12_0, Off16_0])


_TRUST_RANGE = np.array([TRUST_RANGE_BSC, TRUST_RANGE_BSC, TRUST_RANGE_WF1])
_GAMMA = np.diag(1.0 / _TRUST_RANGE ** 2)
DB_KNOB_BACKSTOP = 0.05

_ACCEPTANCE_W = np.diag([1 / SIGMA_M12, 1 / SIGMA_M16, 1 / SIGMA_OFFSET12, 1 / SIGMA_OFFSET16])


def sample_delta(iteration, J):
    rng = np.random.default_rng(BASE_SEED + iteration)
    sampled_residual = BASE_RESIDUAL + rng.normal(0, SAMPLE_SIGMAS)
    Jw = _ACCEPTANCE_W @ J
    rw = _ACCEPTANCE_W @ sampled_residual
    A = Jw.T @ Jw + _GAMMA
    b = Jw.T @ rw
    delta = np.linalg.solve(A, b)
    return delta


def exceeds_caps(delta):
    if np.linalg.norm(delta) > NORM_CAP:
        return True
    if abs(delta[KNOBS.index('WF1_Exit_dB')]) > DB_KNOB_BACKSTOP:
        return True
    return False


def fit_slope_offset_pg(per_group, groups_dict):
    xs = np.array([groups_dict[g] for g in groups_dict])
    ys = np.array([per_group[g][0] for g in groups_dict])
    return np.polyfit(xs, ys, 1)  # slope, intercept


def evaluate_e2(task):
    iteration, delta = task
    t0 = time.time()
    tag = f"mcq3q4wf1_{iteration}_" + "_".join(f"{d:+.5f}" for d in delta)
    cosyFile = tag.replace(".", "p")[:60]
    tuneChanges = [knob_to_tunechange(k, d) for k, d in zip(KNOBS, delta)]
    try:
        M12, M16, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS_E2, cosyFile=cosyFile,
            ta_y_offset=0.0, verbose=False)
        Off12 = fit_slope_offset_pg(per_group, DALPHAX_GROUPS)[1]
        Off16 = fit_slope_offset_pg(per_group, DE_GROUPS)[1]
        t12 = ((M12 - M12_TARGET) / SIGMA_M12) ** 2
        t16 = ((M16 - M16_TARGET) / SIGMA_M16) ** 2
        tOff12 = ((Off12 - OFFSET12_TARGET) / SIGMA_OFFSET12) ** 2
        tOff16 = ((Off16 - OFFSET16_TARGET) / SIGMA_OFFSET16) ** 2
        loss = t12 + t16 + tOff12 + tOff16
        return {"iteration": iteration, "delta": delta, "M12_sim": M12, "M16_sim": M16,
                "Offset12_sim": Off12, "Offset16_sim": Off16, "loss": loss,
                "elapsed_s": time.time() - t0}
    except Exception as e:
        return ("FAILED", iteration, str(e))


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_all_rows():
    with open(RESULTS_FILE, "r", newline="") as f:
        return list(csv.DictReader(f))


def append_result(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def make_row(iteration, status, delta=None, e2=None, elapsed_s=None):
    row = {"iteration": iteration, "status": status, "elapsed_s": elapsed_s}
    for k in KNOBS:
        row[k] = None
    row.update({"combined_norm": None, "M12_sim": None, "M16_sim": None,
                "Offset12_sim": None, "Offset16_sim": None, "loss": None})
    if delta is not None:
        row.update(dict(zip(KNOBS, delta)))
        row["combined_norm"] = float(np.linalg.norm(delta))
    if e2 is not None:
        row.update({"M12_sim": e2["M12_sim"], "M16_sim": e2["M16_sim"],
                     "Offset12_sim": e2["Offset12_sim"], "Offset16_sim": e2["Offset16_sim"],
                     "loss": e2["loss"]})
    return row


if __name__ == "__main__":
    ensure_results_file_exists()
    J, nominal_vals = build_jacobian()
    print(f"Jacobian built: {J.shape}, nominal={nominal_vals}\n{J}\n")

    existing = load_all_rows()
    done_iters = set(int(r["iteration"]) for r in existing)
    n_e2_passed = sum(1 for r in existing if r["status"] == "e2_passed")
    n_rerolled = sum(1 for r in existing if r["status"] == "norm_rejected")
    n_e2_failed = sum(1 for r in existing if r["status"] == "e2_failed")
    best_loss = min((float(r["loss"]) for r in existing if r["loss"] not in (None, "")), default=None)
    print(f"Resuming: {len(done_iters)} iterations done so far "
          f"({n_e2_passed} e2_passed, {n_rerolled} norm-rejected, {n_e2_failed} e2-failed, "
          f"best_loss={best_loss})\n")

    next_iter = (max(done_iters) + 1) if done_iters else 0

    with Pool(processes=10) as pool:
        while n_e2_passed < TARGET_E2_PASSED:
            batch_e2_tasks = []
            iter_cursor = next_iter
            while len(batch_e2_tasks) < BATCH_SIZE:
                if iter_cursor in done_iters:
                    iter_cursor += 1
                    continue
                delta = sample_delta(iter_cursor, J)
                if exceeds_caps(delta):
                    append_result(make_row(iter_cursor, "norm_rejected", delta=delta))
                    n_rerolled += 1
                    done_iters.add(iter_cursor)
                else:
                    batch_e2_tasks.append((iter_cursor, delta))
                    done_iters.add(iter_cursor)
                iter_cursor += 1
            next_iter = iter_cursor

            print(f"Evaluating E2 for {len(batch_e2_tasks)} candidates...", flush=True)
            e2_results = pool.map(evaluate_e2, batch_e2_tasks)

            for (iteration, delta), r in zip(batch_e2_tasks, e2_results):
                if isinstance(r, tuple) and r[0] == "FAILED":
                    print(f"  iter {iteration} E2 FAILED: {r[2]}", flush=True)
                    append_result(make_row(iteration, "e2_error", delta=delta))
                    continue
                if best_loss is None or r["loss"] < best_loss:
                    best_loss = r["loss"]
                if r["loss"] > CHI2_THRESHOLD_2SIGMA_DF4:
                    append_result(make_row(iteration, "e2_failed", delta=delta, e2=r, elapsed_s=r["elapsed_s"]))
                    n_e2_failed += 1
                    print(f"  iter {iteration} E2 FAILED loss={r['loss']:.2f}", flush=True)
                else:
                    n_e2_passed += 1
                    append_result(make_row(iteration, "e2_passed", delta=delta, e2=r, elapsed_s=r["elapsed_s"]))
                    print(f"  iter {iteration} E2 PASSED loss={r['loss']:.2f} ({n_e2_passed}/{TARGET_E2_PASSED})",
                          flush=True)

            print(f"\nProgress: {n_e2_passed}/{TARGET_E2_PASSED} e2_passed "
                  f"({n_rerolled} norm-rejected, {n_e2_failed} e2-failed so far, best_loss={best_loss:.3f})\n",
                  flush=True)

    print(f"\nDone. {n_e2_passed} E2-passing solutions in {RESULTS_FILE}.")
