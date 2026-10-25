"""
Monte Carlo family search v5 (2026-08-13): replaces v4's hard per-knob
cap with proper ridge/Tikhonov regularization baked into the solve
itself (see sample_delta and the TRUST_RANGE_*/_GAMMA constants). v4's
hard 1.5% dB cap barely moved the achievable loss versus v3's uncapped
run (25.14 -> 25.08, essentially unchanged after several more hours) --
diagnosis: an UNCONSTRAINED minimum-norm solve, filtered by post-hoc
rejection, doesn't explore "the best correction that respects the cap";
it only samples the rare cases where the unconstrained solution already
happened to be small in the capped directions, and those leftover
quad-only corrections don't necessarily fit the specific sampled
residual well (dominant failure mode: M12 landing well short of target
even though nothing looked numerically pathological). Regularization
fixes this at the source: each knob's own calibration range (+-10% for
B_SC quads, +-1% for dB knobs, matching the OAT/offset-Jacobian scans)
is treated as a Gaussian prior standard deviation, so dB knobs are
penalized ~100x more per unit movement DURING the solve, rather than
being discovered-and-discarded afterward. NORM_CAP and a generous
DB_KNOB_BACKSTOP (5%) remain only as a safety net for pathological
sampled residuals, not the primary mechanism. v3/v4's evaluations (all
failed) are not reused -- fresh RESULTS_FILE.

v4 (2026-08-13, superseded): added a per-knob hard cap (DB_KNOB_CAP) for
dB-type knobs, after v3 (12 knobs, running for several hours) failed to
find a single E2-passing candidate out of 420 tried (best loss ~25, well
above the 9.72 threshold). Root cause: B1/B2/B4 Exit dB turned out to be
~35x steeper per percentage-point than even Q3 (see their raw
offset-Jacobian scan data -- ~30mm swing in group means over just their
own +-1% tested range), so the minimum-norm solve exploited that
leverage and routinely placed them at +-2-7%, 2-7x past where their
linear response has ever been verified.

v3 (2026-08-12): added B1/B2/B3/B4 Exit dB on top of v2's Q3 addition,
after the derivative-based ranking (plot_oat_derivative_ranking.py -- a
fix for the SAME grid-resolution artifact that caused v2's Q3 addition:
ranking by best-loss-on-grid undercounts knobs whose true minimum is
narrower than their tested grid spacing) showed these four dipole dB
knobs have real local leverage on the E2 loss at nominal. Their
Offset12/Offset16 sensitivity was computed separately in
fp2_offset_jacobian_scan_dipoles.py (pure read-only reuse of the cached
nominal matrix, no compile needed, no race-condition risk -- see that
script's docstring).

v2's docstring (Q3 addition) for reference: Q3 was excluded from v1
because the ORIGINAL OAT screen's ranking of Q3 as unimportant was
itself a grid-resolution artifact -- Q3's sensitivity (dM16/dQ3 ~ -41.5,
the steepest of any single B_SC knob) is so much larger than the OAT
screen's +-10%/10-point grid spacing that every sampled point overshot
past its true (much narrower) minimum, making it look uniformly bad. The
z-crossing sensitivity scan independently confirmed Q3 is actually one
of the most efficient knobs. See plot_q3_oat_loss.py for the diagnostic
that caught this.

Treats the E2 targets (M12, M16, Offset12, Offset16) as themselves
experimentally uncertain: samples plausible "true" target residuals from
their own fit uncertainty (statistical stderr from the measured-data
linear regressions, plus a systematic added LINEARLY -- the user's
choice, more conservative than quadrature -- to Offset16's), solves the
minimum-norm 8-knob correction for each sampled residual, and only keeps
a candidate if:
  (a) its combined knob magnitude is <= NORM_CAP (rejected BEFORE running
      COSY -- we've twice confirmed the linear model breaks down well
      before 24%, so there's no point spending COSY time on candidates
      that large),
  (b) verified with real COSY, its loss against the ORIGINAL (tight,
      calibrated) target/sigma is <= the chi-square(df=4, 2-sigma)
      threshold (9.716) -- NOT the inflated sampling sigma; sampling
      broad and accepting narrow is intentional, not a double-count.

E3 is still computed for every E2-passing candidate (a narrow 7-point Q6
scan + parabola fit, same as v1) and logged as
e3_fitted_shift_narrow7pt_UNRELIABLE -- cheap extra information, kept in
case Q3's inclusion changes the picture in some way not yet understood,
per the user's explicit request 2026-08-12. But this fit is KNOWN to
extrapolate wildly (see fp2_e3_check_mc_candidate.py) and should not be
trusted at face value -- real E3 checking is the separate, validated
fp2_family_e3_wide_recheck.py, run afterward. The STOPPING condition is
reaching TARGET_E2_PASSED candidates that pass E2, regardless of the
in-loop E3 result (v1's mistake was gating stopping on the narrow E3
test, which -- we now know -- essentially never passes, so it would loop
forever).

12 knobs: Q1, Q2, Q3, Q4, Q5, Q6, Q7 B_SC + WF1/B1/B2/B3/B4 Exit dB.
Still excluding H1/H3/O1 (hexapole/octupole, historically minor and
confirmed inefficient by the z-crossing sensitivity scan).

Each sampled candidate (iteration i) is deterministic given i (seeded
via BASE_SEED+i), so this is fully checkpointable/resumable/parallel:
every iteration -- whether norm-rejected, E2-failed, or E2-passed -- is
appended to RESULTS_FILE as soon as it's decided, and already-evaluated
iterations are skipped on restart. Runs in batches until
>= TARGET_E2_PASSED candidates pass E2.

Run:
    cd AgentSearch
    python3 fp2_family_mc.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp2_metrics import compute_M12_M16, compute_fp2_width, DALPHAX_GROUPS, DE_GROUPS

# ---------------------------------------------------------------- targets / acceptance sigmas
M12_TARGET, M16_TARGET = -0.2472, -0.3842
OFFSET12_TARGET, OFFSET16_TARGET = 1.543, 0.537
SIGMA_M12, SIGMA_M16 = 0.02, 0.03
SIGMA_OFFSET12, SIGMA_OFFSET16 = 0.2, 1.5
CHI2_THRESHOLD_2SIGMA_DF4 = 9.715641142541578

# nominal -> target residual (same 4 terms, order M12/M16/Offset12/Offset16)
BASE_RESIDUAL = np.array([-0.4562, -0.5995, 3.2057, 2.0769])

# sampling (experimental-fit) uncertainties -- distinct from the acceptance sigmas above
SAMPLE_SIGMA_M12 = 0.0269
SAMPLE_SIGMA_M16 = 0.2961
SAMPLE_SIGMA_OFFSET12 = 0.3566
SAMPLE_SIGMA_OFFSET16 = 0.925  # 0.425 statistical + 0.5 systematic, added linearly per user's choice
SAMPLE_SIGMAS = np.array([SAMPLE_SIGMA_M12, SAMPLE_SIGMA_M16, SAMPLE_SIGMA_OFFSET12, SAMPLE_SIGMA_OFFSET16])

NORM_CAP = 0.10
BASE_SEED = 20260812

# Regularization (2026-08-13, replaces the hard per-knob cap): a hard
# cap on top of an UNCONSTRAINED minimum-norm solve doesn't work well --
# capping to 1.5% (from an original 10%) barely moved the achievable
# loss (v3 uncapped best=25.14 vs v4 1.5%-capped best=25.08), because
# the solve still WANTS to lean on the ultra-sensitive dB knobs (that's
# what minimizes the norm) and gets rejected/discarded rather than
# finding a genuinely good alternative -- so we were mostly sampling the
# rare leftover cases where the unconstrained solution happened to be
# small in those directions, not actually constrained solutions.
#
# Fix: generalized Tikhonov/ridge regularization baked into the solve
# itself (see sample_delta) -- each knob's own calibration range is
# treated as a Gaussian prior standard deviation, Gamma_ii = 1/trust_i^2.
# Quads (B_SC) trust their +-10% OAT calibration range; dB knobs trust
# their +-1% range, so dB knobs are penalized ~100x more heavily per
# unit movement, naturally discouraging their use without a hard
# cutoff. NORM_CAP is kept only as a loose safety backstop.
#
# WF1 gets its OWN, more generous trust range (2%, not the shared 1%),
# per the user's diagnosis 2026-08-13: even hours in, v5 (uniform 1% dB
# trust) still found zero E2-passing candidates. Verified empirically
# (not just asserted): excluding B1-B4 from the solve entirely roughly
# DOUBLES WF1's typical value (0.09% -> 0.19%) -- a real cross-term
# effect of ridge regression with correlated design columns (adding more
# regularized-but-nonzero-leverage directions shrinks everyone's
# coefficient, not just the newly-added ones), not simply "WF1's own
# weight was wrong" in isolation. WF1 has real track record the other dB
# knobs don't (hundreds of validated v1 7-knob candidates used
# WF1~0.24-0.28%, unlike B1-B4 which have never been validated at any
# nonzero value) -- 2% was found to reproduce that same range (mean
# 0.22%, [0.08%,0.35%]) even with B1-B4 present in the joint solve.
TRUST_RANGE_BSC = 0.10
TRUST_RANGE_DB = 0.01
TRUST_RANGE_WF1 = 0.02

E3_SHIFT_CENTER = -0.0088
E3_TOLERANCE = 0.004
E3_N_POINTS = 7
E3_SCAN_HALFWIDTH = 0.01  # +- around the expected shift point (candidate_Q6 + E3_SHIFT_CENTER)

KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
         'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
OAT_RESULTS_FILE = "Results/fp2_offset_jacobian_scan_results.csv"

NRAYS_E2 = 3000
NRAYS_E3 = 3000
TARGET_E2_PASSED = 45
BATCH_SIZE = 20

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp2_family_mc_v7_results.csv")
FIELDNAMES = (["iteration", "status"] + KNOBS +
              ["combined_norm", "M12_sim", "M16_sim", "Offset12_sim", "Offset16_sim",
               "loss", "e3_fitted_shift_narrow7pt_UNRELIABLE", "elapsed_s"])


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
    J = np.array(J_cols).T  # 4 x 7
    return J, np.array([M12_0, M16_0, Off12_0, Off16_0])


_DB_KNOB_MASK = np.array([k.endswith('_Exit_dB') for k in KNOBS])
_TRUST_RANGE = np.where(_DB_KNOB_MASK, TRUST_RANGE_DB, TRUST_RANGE_BSC)
_TRUST_RANGE[KNOBS.index('WF1_Exit_dB')] = TRUST_RANGE_WF1
_GAMMA = np.diag(1.0 / _TRUST_RANGE ** 2)  # ridge penalty matrix, Gamma_ii = 1/trust_i^2
# safety backstop only -- regularization is the primary mechanism now
DB_KNOB_BACKSTOP = 0.05

# Residual weighting (2026-08-13, found alongside the WF1 diagnosis): the
# solve was minimizing the RAW (unweighted) residual norm, treating a
# 1-unit miss in Offset16 (sigma=1.5, barely notable) as equally bad as a
# 1-unit miss in M12 (sigma=0.02, catastrophic -- 75x tighter). Verified
# directly: across 5 sampled residuals, the unweighted solve got M12's
# SIGN wrong 3/5 times (predicting +0.11 to +0.20 against a target of
# -0.2472), while weighting by the same 1/sigma the acceptance loss
# itself uses landed M12 within ~0.02 of target every time. This -- not
# the WF1 trust range -- looks like the real reason the achievable loss
# has been stuck. Now weight both J and the sampled residual by
# diag(1/sigma) before the ridge solve, so the solve targets the SAME
# objective the chi-square acceptance test checks.
_ACCEPTANCE_W = np.diag([1 / SIGMA_M12, 1 / SIGMA_M16, 1 / SIGMA_OFFSET12, 1 / SIGMA_OFFSET16])


def sample_delta(iteration, J):
    rng = np.random.default_rng(BASE_SEED + iteration)
    sampled_residual = BASE_RESIDUAL + rng.normal(0, SAMPLE_SIGMAS)
    Jw = _ACCEPTANCE_W @ J
    rw = _ACCEPTANCE_W @ sampled_residual
    # generalized Tikhonov/ridge solve on the WEIGHTED system: minimize
    # ||W(J@delta - r)||^2 + ||Gamma^(1/2) @ delta||^2
    # => delta = (Jw^T Jw + Gamma)^-1 Jw^T rw
    A = Jw.T @ Jw + _GAMMA
    b = Jw.T @ rw
    delta = np.linalg.solve(A, b)
    return delta


def exceeds_caps(delta):
    if np.linalg.norm(delta) > NORM_CAP:
        return True
    if np.any(np.abs(delta[_DB_KNOB_MASK]) > DB_KNOB_BACKSTOP):
        return True
    return False


def fit_slope_offset_pg(per_group, groups_dict):
    xs = np.array([groups_dict[g] for g in groups_dict])
    ys = np.array([per_group[g][0] for g in groups_dict])
    return np.polyfit(xs, ys, 1)  # slope, intercept


def evaluate_e2(task):
    iteration, delta = task
    t0 = time.time()
    tag = f"mc{iteration}_" + "_".join(f"{d:+.5f}" for d in delta)
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


def evaluate_e3_point(task):
    iteration, delta, q6_val = task
    tuneChanges = [knob_to_tunechange(k, d) for k, d in zip(KNOBS, delta) if k != 'Q6_B_SC']
    tuneChanges.append(['Q6', 'B_SC', float(q6_val)])
    tag = f"mc{iteration}_e3_{q6_val:+.5f}".replace(".", "p")
    cosyFile = tag[:60]
    try:
        width = compute_fp2_width(tuneChanges=tuneChanges, nRays=NRAYS_E3, cosyFile=cosyFile,
                                    run_cosy_flag=True, verbose=False)
        return (iteration, q6_val, width["vd1638_std"])
    except Exception as e:
        return (iteration, q6_val, None)


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


def make_row(iteration, status, delta=None, e2=None, e3_shift=None, elapsed_s=None):
    row = {"iteration": iteration, "status": status, "elapsed_s": elapsed_s}
    for k in KNOBS:
        row[k] = None
    row.update({"combined_norm": None, "M12_sim": None, "M16_sim": None,
                "Offset12_sim": None, "Offset16_sim": None, "loss": None,
                "e3_fitted_shift_narrow7pt_UNRELIABLE": e3_shift})
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
    print(f"Jacobian built: {J.shape}, nominal={nominal_vals}\n")

    existing = load_all_rows()
    done_iters = set(int(r["iteration"]) for r in existing)
    n_e2_passed = sum(1 for r in existing if r["status"] in ("e2_passed", "accepted", "e3_failed"))
    n_rerolled = sum(1 for r in existing if r["status"] == "norm_rejected")
    n_e2_failed = sum(1 for r in existing if r["status"] == "e2_failed")
    n_narrow_e3_passed = sum(1 for r in existing if r["status"] == "accepted")
    print(f"Resuming: {len(done_iters)} iterations done so far "
          f"({n_e2_passed} e2_passed [{n_narrow_e3_passed} also passed narrow E3], "
          f"{n_rerolled} norm-rejected, {n_e2_failed} e2-failed)\n")

    next_iter = (max(done_iters) + 1) if done_iters else 0

    with Pool(processes=10) as pool:
        while n_e2_passed < TARGET_E2_PASSED:
            # ---- draw a batch, filtering norm-cap rejects for free ----
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

            # ---- E2 stage, parallel ----
            print(f"Evaluating E2 for {len(batch_e2_tasks)} candidates...", flush=True)
            e2_results = pool.map(evaluate_e2, batch_e2_tasks)

            e3_pending = []  # (iteration, delta, e2_result)
            for (iteration, delta), r in zip(batch_e2_tasks, e2_results):
                if isinstance(r, tuple) and r[0] == "FAILED":
                    print(f"  iter {iteration} E2 FAILED: {r[2]}", flush=True)
                    append_result(make_row(iteration, "e2_error", delta=delta))
                    continue
                if r["loss"] > CHI2_THRESHOLD_2SIGMA_DF4:
                    append_result(make_row(iteration, "e2_failed", delta=delta, e2=r, elapsed_s=r["elapsed_s"]))
                    n_e2_failed += 1
                    print(f"  iter {iteration} E2 FAILED loss={r['loss']:.2f}", flush=True)
                else:
                    n_e2_passed += 1
                    print(f"  iter {iteration} E2 PASSED loss={r['loss']:.2f} "
                          f"({n_e2_passed}/{TARGET_E2_PASSED}), queued for (informational) E3 check", flush=True)
                    e3_pending.append((iteration, delta, r))

            # ---- E3 stage (informational only -- does not gate stopping) ----
            if e3_pending:
                e3_tasks = []
                for iteration, delta, r in e3_pending:
                    q6_candidate = delta[KNOBS.index('Q6_B_SC')]
                    center = q6_candidate + E3_SHIFT_CENTER
                    q6_scan = np.linspace(center - E3_SCAN_HALFWIDTH, center + E3_SCAN_HALFWIDTH, E3_N_POINTS)
                    for q6v in q6_scan:
                        e3_tasks.append((iteration, delta, float(q6v)))

                print(f"Evaluating (informational) E3 ({len(e3_tasks)} sub-points across "
                      f"{len(e3_pending)} candidates)...", flush=True)
                e3_results = pool.map(evaluate_e3_point, e3_tasks)

                by_iter = {}
                for iteration, q6v, std in e3_results:
                    by_iter.setdefault(iteration, []).append((q6v, std))

                for iteration, delta, r in e3_pending:
                    pts = [(q6v, std) for q6v, std in by_iter.get(iteration, []) if std is not None]
                    q6_candidate = delta[KNOBS.index('Q6_B_SC')]
                    if len(pts) < 5:
                        append_result(make_row(iteration, "e3_error", delta=delta, e2=r))
                        continue
                    q6s = np.array([p[0] for p in pts])
                    stds = np.array([p[1] for p in pts])
                    a, b, c = np.polyfit(q6s, stds, 2)
                    if a <= 0:
                        fitted_min_shift = None
                        passed = False
                    else:
                        q6_at_min = -b / (2 * a)
                        fitted_min_shift = q6_at_min - q6_candidate
                        passed = abs(fitted_min_shift - E3_SHIFT_CENTER) <= E3_TOLERANCE
                    status = "accepted" if passed else "e3_failed"
                    append_result(make_row(iteration, status, delta=delta, e2=r, e3_shift=fitted_min_shift))
                    if passed:
                        n_narrow_e3_passed += 1
                        shift_pct = fitted_min_shift * 100 if fitted_min_shift is not None else float('nan')
                        print(f"  iter {iteration} narrow E3 PASSED too (shift={shift_pct:.3f}%) -- interesting!",
                              flush=True)

            print(f"\nProgress: {n_e2_passed}/{TARGET_E2_PASSED} e2_passed "
                  f"({n_narrow_e3_passed} also passed narrow E3, {n_rerolled} norm-rejected, "
                  f"{n_e2_failed} e2-failed so far)\n")

    print(f"\nDone. {n_e2_passed} E2-passing solutions in {RESULTS_FILE} "
          f"({n_narrow_e3_passed} also passed the narrow (unreliable) E3 test).")
