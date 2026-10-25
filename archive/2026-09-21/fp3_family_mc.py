"""
FP3 family Monte Carlo search (2026-08-18), mirroring fp2_family_mc.py's
methodology but starting from 46 DIFFERENT base points (the FP2
1-sigma-accepted candidates) instead of a single nominal, each with its
own candidate-specific Jacobian and base residual -- per the user's
explicit design decision 2026-08-16/17: rather than assume one shared
Jacobian is representative, sample across all 46 candidate-specific
ones.

Per iteration:
  1. Pick one of the 46 FP2 1-sigma candidates uniformly at random
     (seeded, deterministic).
  2. Sample a plausible "true" FP3 residual around THAT candidate's own
     base residual (target - that candidate's own FP3 M12/M16/Offset12/
     Offset16, from fp3_family_1sigma_results.csv).

     LOSS/SAMPLING STRUCTURE (revised 2026-08-18, per user request):
     M12+Offset12 use a proper Mahalanobis distance against the FULL 2x2
     covariance matrix of the experimental M12 fit (groups 10,14,15,
     n=3, dof=1) -- slope and intercept from a 3-point OLS fit are
     strongly correlated (corr=0.76, see
     plot_m12_m16_slope_intercept_contour.py), and treating them as
     independent with hand-picked sigmas (0.04/0.5) was found to put the
     46 candidates' own FP3 M12 response ~13 sigma from target -- almost
     entirely an artifact of ignoring that correlation and using a
     narrower-than-warranted marginal sigma. The Mahalanobis distance
     Delta^T Sigma^-1 Delta uses the fit's OWN statistical covariance
     directly (no separate hand-picked sigma needed), and residual
     sampling for M12/Offset12 draws CORRELATED noise from that same
     covariance (Sigma^1/2 @ standard_normal), consistent with reusing
     the acceptance uncertainty for sampling (the same principle applied
     in the M16 case below).

     M16+Offset16 keep the original independent chi-square form, per
     user request, with sigma_M16=0.2 (loosened from 0.1) and
     sigma_Offset16=1.5 (unchanged) -- M16's own experimental fit
     (8 groups, dof=6) has a much weaker slope/intercept correlation
     (-0.12) so the independence assumption is far less consequential
     there.

     Total is still a 4-dof chi-square (2 from the Mahalanobis M12 pair
     + 2 from the independent M16 pair), so the SAME acceptance
     thresholds (1-sigma=4.72, 2-sigma=9.72) apply unchanged.
  3. Solve for the FP3 knob delta using THAT candidate's own 4x7
     Jacobian (fp3_full_jacobian_raw_results.csv, central-difference at
     +-2% for Q8-11, +-0.5% for B5/B6/WF2), via the same weighted +
     ridge-regularized normal-equations solve validated in fp2_family_mc
     v7 (delta = (Jw^T Jw + Gamma)^-1 Jw^T rw).
  4. TRUST_RANGE (the regularization prior / effective calibration
     range) is set to the SAME step used to build the Jacobian (0.02 for
     Q8-11, 0.005 for B5/B6/WF2) rather than a wider assumed range,
     since that's the only range over which the linear model has
     actually been verified here -- unlike FP2's OAT screen, which
     covered a wider +-10%/+-1% grid. NORM_CAP and the dB backstop are
     scaled down proportionally from FP2's convention (same ratio to
     their respective trust ranges).
  5. Reject before running COSY if the combined FP3-knob delta exceeds
     NORM_CAP or any individual dB knob exceeds its backstop (same
     cheap pre-filter FP2 used).
  6. Verify with real COSY: tuneChanges = that candidate's OWN FP2
     knobs (fixed) + the solved FP3 knob deltas, compute M12/M16/
     Offset12/Offset16 at FP3 via fp3_metrics.compute_M12_M16, and
     accept if the chi-square loss against the FP3 targets/sigmas is
     <= the chi-square(df=4, 2-sigma) threshold (9.716) -- exact same
     threshold convention as FP2 (4 terms, same acceptance region).

No FP3 analogue of FP2's E3 cross-check exists yet, so none is run here.

Checkpointed/resumable/parallel: every iteration (norm-rejected or
E2-failed or E2-passed) appended to RESULTS_FILE as soon as it's
decided; already-evaluated iterations are skipped on restart. Runs in
batches until TARGET_PASSED candidates pass.

Run:
    cd AgentSearch
    python3 fp3_family_mc.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp3_metrics import (compute_M12_M16, DALPHAX_GROUPS, DE_GROUPS,
                          DALPHAX_GROUPS_MEASURED_X)

# ---------------------------------------------------------------- targets / acceptance sigmas
# Target/covariance order used EVERYWHERE below: [M12, Offset12, M16, Offset16]
# (grouped by fit-pair, not by quantity type, so the M12 pair's 2x2
# covariance block sits together -- see the module docstring).


def _m12_fit_covariance():
    """OLS fit of the FP3 M12 experimental data (groups 10,14,15) plus
    its full parameter covariance matrix, reordered to [slope, intercept]
    i.e. [M12, Offset12]. Same computation as
    plot_m12_m16_slope_intercept_contour.py's fit_covariance(), kept
    self-contained here rather than importing a plotting script as
    library code."""
    groups = list(DALPHAX_GROUPS.keys())
    x = np.array([DALPHAX_GROUPS[g] for g in groups])
    y = np.array([DALPHAX_GROUPS_MEASURED_X[g] for g in groups])
    n = len(x)
    X = np.column_stack([np.ones(n), x])  # beta = [intercept, slope]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    dof = n - 2
    cov_full = (rss / dof) * np.linalg.inv(X.T @ X)  # order [intercept, slope]
    order = [1, 0]  # -> [slope, intercept] = [M12, Offset12]
    return beta[1], beta[0], cov_full[np.ix_(order, order)]


M12_TARGET, OFFSET12_TARGET, _COV_M12_RAW = _m12_fit_covariance()
# 2026-08-18, per user request: tighten the M12 acceptance region by
# using the fit's covariance divided by 4 (checked /2 first, found still
# comfortably feasible, then moved to /4) -- shrinks the confidence
# ellipse's linear dimensions by 1/2 and QUADRUPLES the Mahalanobis
# penalty for any given (M12,Offset12) mismatch, since
# chi2 = Delta^T Sigma^-1 Delta and (Sigma/4)^-1 = 4*Sigma^-1.
COV_M12 = _COV_M12_RAW / 4
M16_TARGET, OFFSET16_TARGET = -2.0956, 3.0608
SIGMA_M16, SIGMA_OFFSET16 = 0.2, 1.5
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578
TARGETS = np.array([M12_TARGET, OFFSET12_TARGET, M16_TARGET, OFFSET16_TARGET])

_COV_M12_INV = np.linalg.inv(COV_M12)
_eigvals_M12, _eigvecs_M12 = np.linalg.eigh(COV_M12)
_M12_WHITEN = _eigvecs_M12 @ np.diag(1.0 / np.sqrt(_eigvals_M12)) @ _eigvecs_M12.T  # Sigma^-1/2
_M12_SAMPLE_SQRT = _eigvecs_M12 @ np.diag(np.sqrt(_eigvals_M12)) @ _eigvecs_M12.T   # Sigma^1/2 (for correlated sampling)

FP3_KNOBS = ['Q8_B_SC', 'Q9_B_SC', 'Q10_B_SC', 'Q11_B_SC', 'B5_Exit_dB', 'B6_Exit_dB', 'WF2_Exit_dB']
_DB_KNOB_MASK = np.array([k.endswith('_Exit_dB') for k in FP3_KNOBS])
TRUST_RANGE_BSC = 0.02   # matches the Jacobian's own tested step (Q8-11)
TRUST_RANGE_DB = 0.005   # matches the Jacobian's own tested step (B5/B6/WF2)
_TRUST_RANGE = np.where(_DB_KNOB_MASK, TRUST_RANGE_DB, TRUST_RANGE_BSC)
_GAMMA = np.diag(1.0 / _TRUST_RANGE ** 2)
# 2026-08-18: an initial NORM_CAP=0.03 guess ("~ one quad's trust range,
# same ratio FP2 used") was checked against the actual Jacobians/residuals
# BEFORE committing to a COSY run (same kind of cheap linear-model
# feasibility check that caught the FP2 Q3+Q4+WF1 problem) and turned out
# to reject EVERY one of the 46 base candidates -- the real minimum
# combined norm needed to reach the acceptance threshold is 0.059-0.094
# (median 0.077) across candidates, ~2-3x that guess. Raised to 0.12 to
# comfortably cover that with margin, matching the same scale FP2's own
# successful v7 search used (NORM_CAP=0.10, with more knobs to spread
# the correction over).
NORM_CAP = 0.12
DB_KNOB_BACKSTOP = 0.0125  # 2.5x the dB trust range, same ratio FP2 used (0.05 / 0.01)

# Block-diagonal whitening matrix, order matches TARGETS = [M12, Offset12, M16, Offset16]:
# top-left 2x2 = Sigma_M12^-1/2 (whitens the Mahalanobis M12 pair), bottom-right
# 2x2 = diag(1/sigma) (the independent M16 pair). No cross term between the two
# fit-pairs (treated as independent, ignoring the shared group-10 data point).
_ACCEPTANCE_W = np.zeros((4, 4))
_ACCEPTANCE_W[0:2, 0:2] = _M12_WHITEN
_ACCEPTANCE_W[2, 2] = 1 / SIGMA_M16
_ACCEPTANCE_W[3, 3] = 1 / SIGMA_OFFSET16

V1_FILE = "Results/fp2_family_mc_results.csv"
V1_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']
V7_FILE = "Results/fp2_family_mc_v7_results.csv"
V7_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
            'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
CHI2_1SIGMA_DF4 = 4.7196
FP3_FAMILY_FILE = "Results/fp3_family_1sigma_results.csv"
JACOBIAN_FILE = "Results/fp3_full_jacobian_raw_results.csv"
FP3_KNOB_STEP = {
    'Q8_B_SC': 0.02, 'Q9_B_SC': 0.02, 'Q10_B_SC': 0.02, 'Q11_B_SC': 0.02,
    'B5_Exit_dB': 0.005, 'B6_Exit_dB': 0.005, 'WF2_Exit_dB': 0.005,
}

BASE_SEED = 20260818
NRAYS = 3000
TARGET_PASSED = 45
BATCH_SIZE = 20

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp3_family_mc_results.csv")
FIELDNAMES = (["iteration", "status", "base_source", "base_iteration"] + FP3_KNOBS +
              ["combined_norm", "M12_sim", "M16_sim", "Offset12_sim", "Offset16_sim", "loss", "elapsed_s"])


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def load_base_candidates():
    """Returns {(source, iteration): {'fp2_tuneChanges': [...], 'residual': np.array([M12,Off12,M16,Off16])}}."""
    fp2_knobs = {}
    for path, knobs, source in [(V1_FILE, V1_KNOBS, "v1"), (V7_FILE, V7_KNOBS, "v7")]:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            if r["loss"] in (None, "") or float(r["loss"]) > CHI2_1SIGMA_DF4:
                continue
            fp2_knobs[(source, int(r["iteration"]))] = {k: float(r[k]) for k in knobs}

    fp3_rows = list(csv.DictReader(open(FP3_FAMILY_FILE)))
    candidates = {}
    for r in fp3_rows:
        key = (r["source"], int(r["iteration"]))
        if key not in fp2_knobs:
            continue
        sim = np.array([float(r["M12_sim"]), float(r["Offset12_sim"]),
                         float(r["M16_sim"]), float(r["Offset16_sim"])])
        candidates[key] = {
            "fp2_tuneChanges": [knob_to_tunechange(k, v) for k, v in fp2_knobs[key].items()],
            "residual": TARGETS - sim,
        }
    return candidates


def load_jacobians():
    """Returns {(source, iteration): 4x7 np.array}, rows=[M12,Offset12,M16,Offset16], cols=FP3_KNOBS."""
    rows = list(csv.DictReader(open(JACOBIAN_FILE)))
    by_key = {}
    for r in rows:
        key = (r["source"], int(r["iteration"]), r["fp3_knob"], int(r["sign"]))
        by_key[key] = {t: float(r[t]) for t in ["M12", "M16", "Offset12", "Offset16"]}
    cands = sorted(set((r["source"], int(r["iteration"])) for r in rows))
    jacobians = {}
    for cand in cands:
        cols = []
        ok = True
        for knob in FP3_KNOBS:
            plus = by_key.get((cand[0], cand[1], knob, 1))
            minus = by_key.get((cand[0], cand[1], knob, -1))
            if plus is None or minus is None:
                ok = False
                break
            step = FP3_KNOB_STEP[knob]
            cols.append([(plus[t] - minus[t]) / (2 * step) for t in ["M12", "Offset12", "M16", "Offset16"]])
        if ok:
            jacobians[cand] = np.array(cols).T  # 4 x 7
    return jacobians


def sample_delta(iteration, base_key, J, residual):
    rng = np.random.default_rng(BASE_SEED + iteration)
    # M12/Offset12: CORRELATED noise from the fit's own covariance (Sigma^1/2 @ z).
    # M16/Offset16: independent noise from sigma_M16/sigma_Offset16, same as before.
    noise_m12 = _M12_SAMPLE_SQRT @ rng.standard_normal(2)
    noise_m16 = rng.normal(0, [SIGMA_M16, SIGMA_OFFSET16])
    sampled_residual = residual + np.concatenate([noise_m12, noise_m16])
    Jw = _ACCEPTANCE_W @ J
    rw = _ACCEPTANCE_W @ sampled_residual
    A = Jw.T @ Jw + _GAMMA
    b = Jw.T @ rw
    return np.linalg.solve(A, b)


def exceeds_caps(delta):
    if np.linalg.norm(delta) > NORM_CAP:
        return True
    if np.any(np.abs(delta[_DB_KNOB_MASK]) > DB_KNOB_BACKSTOP):
        return True
    return False


def fit_slope_offset_pg(per_group, groups_dict):
    xs = np.array([groups_dict[g] for g in groups_dict])
    ys = np.array([per_group[g][0] for g in groups_dict])
    return np.polyfit(xs, ys, 1)


def evaluate_e2(task):
    iteration, base_key, delta, fp2_tuneChanges = task
    t0 = time.time()
    tag = f"fp3mc{iteration}_" + "_".join(f"{d:+.5f}" for d in delta)
    cosyFile = tag.replace(".", "p")[:60]
    tuneChanges = fp2_tuneChanges + [knob_to_tunechange(k, d) for k, d in zip(FP3_KNOBS, delta)]
    try:
        M12, M16, per_group, _ = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=cosyFile, verbose=False)
        Off12 = fit_slope_offset_pg(per_group, DALPHAX_GROUPS)[1]
        Off16 = fit_slope_offset_pg(per_group, DE_GROUPS)[1]
        delta_m12 = np.array([M12 - M12_TARGET, Off12 - OFFSET12_TARGET])
        loss_m12 = float(delta_m12 @ _COV_M12_INV @ delta_m12)  # Mahalanobis, 2 dof
        t16 = ((M16 - M16_TARGET) / SIGMA_M16) ** 2
        tOff16 = ((Off16 - OFFSET16_TARGET) / SIGMA_OFFSET16) ** 2
        loss = loss_m12 + t16 + tOff16
        return {"iteration": iteration, "base_key": base_key, "delta": delta,
                "M12_sim": M12, "M16_sim": M16, "Offset12_sim": Off12, "Offset16_sim": Off16,
                "loss": loss, "elapsed_s": time.time() - t0}
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


def make_row(iteration, status, base_key=None, delta=None, e2=None, elapsed_s=None):
    row = {"iteration": iteration, "status": status, "elapsed_s": elapsed_s,
           "base_source": base_key[0] if base_key else None,
           "base_iteration": base_key[1] if base_key else None}
    for k in FP3_KNOBS:
        row[k] = None
    row.update({"combined_norm": None, "M12_sim": None, "M16_sim": None,
                "Offset12_sim": None, "Offset16_sim": None, "loss": None})
    if delta is not None:
        row.update(dict(zip(FP3_KNOBS, delta)))
        row["combined_norm"] = float(np.linalg.norm(delta))
    if e2 is not None:
        row.update({"M12_sim": e2["M12_sim"], "M16_sim": e2["M16_sim"],
                     "Offset12_sim": e2["Offset12_sim"], "Offset16_sim": e2["Offset16_sim"],
                     "loss": e2["loss"]})
    return row


if __name__ == "__main__":
    ensure_results_file_exists()
    base_candidates = load_base_candidates()
    jacobians = load_jacobians()
    usable_keys = sorted(set(base_candidates.keys()) & set(jacobians.keys()))
    print(f"{len(usable_keys)} base candidates usable (have both a base residual and a full Jacobian).\n")

    existing = load_all_rows()
    done_iters = set(int(r["iteration"]) for r in existing)
    n_passed = sum(1 for r in existing if r["status"] == "e2_passed")
    n_rerolled = sum(1 for r in existing if r["status"] == "norm_rejected")
    n_failed = sum(1 for r in existing if r["status"] == "e2_failed")
    best_loss = min((float(r["loss"]) for r in existing if r["loss"] not in (None, "")), default=None)
    print(f"Resuming: {len(done_iters)} iterations done so far "
          f"({n_passed} passed, {n_rerolled} norm-rejected, {n_failed} failed, best_loss={best_loss})\n")

    next_iter = (max(done_iters) + 1) if done_iters else 0
    key_rng = np.random.default_rng(BASE_SEED - 1)  # separate stream, just for picking WHICH candidate

    with Pool(processes=10) as pool:
        while n_passed < TARGET_PASSED:
            batch_tasks = []
            iter_cursor = next_iter
            while len(batch_tasks) < BATCH_SIZE:
                if iter_cursor in done_iters:
                    iter_cursor += 1
                    continue
                base_key = usable_keys[key_rng.integers(len(usable_keys))]
                J = jacobians[base_key]
                residual = base_candidates[base_key]["residual"]
                delta = sample_delta(iter_cursor, base_key, J, residual)
                if exceeds_caps(delta):
                    append_result(make_row(iter_cursor, "norm_rejected", base_key=base_key, delta=delta))
                    n_rerolled += 1
                    done_iters.add(iter_cursor)
                else:
                    batch_tasks.append((iter_cursor, base_key, delta, base_candidates[base_key]["fp2_tuneChanges"]))
                    done_iters.add(iter_cursor)
                iter_cursor += 1
            next_iter = iter_cursor

            print(f"Evaluating {len(batch_tasks)} candidates...", flush=True)
            results = pool.map(evaluate_e2, batch_tasks)

            for (iteration, base_key, delta, _), r in zip(batch_tasks, results):
                if isinstance(r, tuple) and r[0] == "FAILED":
                    print(f"  iter {iteration} FAILED: {r[2]}", flush=True)
                    append_result(make_row(iteration, "e2_error", base_key=base_key, delta=delta))
                    continue
                if best_loss is None or r["loss"] < best_loss:
                    best_loss = r["loss"]
                if r["loss"] > CHI2_THRESHOLD_2SIGMA_DF4:
                    append_result(make_row(iteration, "e2_failed", base_key=base_key, delta=delta,
                                            e2=r, elapsed_s=r["elapsed_s"]))
                    n_failed += 1
                    print(f"  iter {iteration} (base={base_key}) FAILED loss={r['loss']:.2f}", flush=True)
                else:
                    n_passed += 1
                    append_result(make_row(iteration, "e2_passed", base_key=base_key, delta=delta,
                                            e2=r, elapsed_s=r["elapsed_s"]))
                    print(f"  iter {iteration} (base={base_key}) PASSED loss={r['loss']:.2f} "
                          f"({n_passed}/{TARGET_PASSED})", flush=True)

            print(f"\nProgress: {n_passed}/{TARGET_PASSED} passed "
                  f"({n_rerolled} norm-rejected, {n_failed} failed so far, best_loss={best_loss:.3f})\n",
                  flush=True)

    print(f"\nDone. {n_passed} FP3-passing solutions in {RESULTS_FILE}.")
