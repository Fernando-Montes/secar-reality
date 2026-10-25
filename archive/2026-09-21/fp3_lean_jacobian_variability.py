"""
Lean-Jacobian variability check (2026-08-16), per user request: before
committing to building 44 separate FP3 Jacobians (one per FP2 1-sigma
candidate) or assuming a single shared one is good enough, test how much
the FP3-knob sensitivities actually vary across a handful of different
FP2 base tunes.

Picks 5 of the 44 FP2 1-sigma-accepted candidates (v1+v7 combined,
chi2<=4.7196) via a fixed-seed random sample. For EACH of the 7 FP3
knobs available downstream of FP2 (all knobs used, per user request --
"better to have this answer once for all", not just the 3 the manual
best-fit used):
    Q8_B_SC, Q9_B_SC, Q10_B_SC, Q11_B_SC   (step +-2%)
    B5_Exit_dB, B6_Exit_dB, WF2_Exit_dB     (step +-0.5%)
computes a LEAN (2-point central-difference, not full OAT grid) local
Jacobian entry against all 4 FP3 loss targets (M12 slope, M16 slope,
Offset12, Offset16) at each of the 5 candidates' own FP2 base tune.

Each candidate's own FP2 knobs (Q1-Q7/WF1/B1-4, whichever subset that
candidate's search actually used) are held fixed at their found values;
only the FP3 knob is perturbed on top.

RACE-CONDITION FIX (2026-08-16): dB-type FP3 knobs (B5/B6/WF2 Exit dB)
don't touch the compiled .fox lattice at all -- only the B_SC-type
portion of tuneChanges feeds the hash-based matrix filename (see
helper_functions.py's prepareCOSYchanges fix). That means, for a given
candidate, all 6 of its dB-knob sub-tasks (B5+-, B6+-, WF2+-) share the
SAME target matrix directory (that candidate's own B_SC lattice) --
running several of them concurrently in the Pool caused them to race on
the same compile, confirmed by candidate v7/426 returning BIT-IDENTICAL
M12/M16/Offset12/Offset16 across 4 different dB-knob/sign combinations
(impossible physically), and is the likely real cause of several of the
"no rays survived" failures too, not genuine zero transmission.
Fixed by sequentially bootstrap-compiling each candidate's own B_SC
lattice ONCE up front (before the pool opens), then running all of that
candidate's dB-knob sub-tasks read-only (run_cosy_flag=False) against
the known, already-written matrix -- safe to parallelize since they
never write. B_SC-type FP3 knob tasks (Q8-11) are unaffected (each adds
a different value to the hash, so each gets its own unique, race-free
directory) and are left as before.

Checkpointed/resumable: each (candidate, knob, sign) point appended to
RESULTS_FILE as soon as it completes. A separate summary pass computes
the central-difference Jacobian entries and their spread across the 5
candidates.

Run:
    cd AgentSearch
    python3 fp3_lean_jacobian_variability.py
"""
import os
import csv
import time
import numpy as np
from multiprocessing import Pool

from fp3_metrics import compute_M12_M16, mean_x_at_vd1783, DALPHAX_GROUPS, DE_GROUPS

V1_FILE = "Results/fp2_family_mc_results.csv"
V1_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']
V7_FILE = "Results/fp2_family_mc_v7_results.csv"
V7_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
            'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']
CHI2_1SIGMA_DF4 = 4.7196

FP3_KNOBS = ['Q8_B_SC', 'Q9_B_SC', 'Q10_B_SC', 'Q11_B_SC', 'B5_Exit_dB', 'B6_Exit_dB', 'WF2_Exit_dB']
FP3_KNOB_STEP = {
    'Q8_B_SC': 0.02, 'Q9_B_SC': 0.02, 'Q10_B_SC': 0.02, 'Q11_B_SC': 0.02,
    'B5_Exit_dB': 0.005, 'B6_Exit_dB': 0.005, 'WF2_Exit_dB': 0.005,
}

N_SAMPLE = 5
SAMPLE_SEED = 20260816
# 2026-08-16, per user request: the original 5-candidate seeded sample
# came out 4x v1 / 1x v7 -- an unlucky draw relative to the underlying
# 24 v1 / 20 v7 split (roughly even). Add one more, drawn ONLY from v7
# and excluding whatever's already sampled, via a separate deterministic
# seed so the original 5 are untouched.
EXTRA_V7_SEED = SAMPLE_SEED + 1
NRAYS = 3000

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp3_lean_jacobian_variability_results.csv")
FIELDNAMES = ["source", "iteration", "fp3_knob", "sign", "M12", "M16", "Offset12", "Offset16", "elapsed_s"]


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def load_1sigma_candidates():
    candidates = {}
    for path, knobs, source in [(V1_FILE, V1_KNOBS, "v1"), (V7_FILE, V7_KNOBS, "v7")]:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            if r["loss"] in (None, "") or float(r["loss"]) > CHI2_1SIGMA_DF4:
                continue
            candidates[(source, int(r["iteration"]))] = {k: float(r[k]) for k in knobs}
    return candidates


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set((row["source"], int(row["iteration"]), row["fp3_knob"], int(row["sign"]))
                   for row in csv.DictReader(f))


def append_row(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def bootstrap_candidate(key, base_knobs):
    """Sequentially compiles this candidate's own B_SC lattice ONCE (a
    single lean group, not the full 10-group compute_M12_M16, to keep
    the added cost small) and returns the resulting hash-based matrix
    name, so the dB-knob sub-tasks below can safely reuse it read-only
    instead of racing each other."""
    tuneChanges = [knob_to_tunechange(k, v) for k, v in base_knobs.items()]
    tag = f"fp3jac_boot_{key[0]}{key[1]}"[:60]
    _, _, _, usedFoxFile = mean_x_at_vd1783('10', tuneChanges, NRAYS, True, tag)
    return usedFoxFile


def evaluate(task):
    (source, iteration), base_knobs, fp3_knob, sign, base_cosyFile = task
    t0 = time.time()
    step = FP3_KNOB_STEP[fp3_knob] * sign
    tuneChanges = [knob_to_tunechange(k, v) for k, v in base_knobs.items()]
    tuneChanges.append(knob_to_tunechange(fp3_knob, step))
    is_db_knob = fp3_knob.endswith('_Exit_dB')
    try:
        if is_db_knob:
            # Doesn't touch the compiled lattice -- read-only against the
            # candidate's already-bootstrapped matrix, safe in parallel.
            M12, M16, per_group, _ = compute_M12_M16(
                tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=base_cosyFile,
                verbose=False, run_cosy_flag=False)
        else:
            tag = f"fp3jac_{source}{iteration}_{fp3_knob}_{sign:+d}"[:60]
            M12, M16, per_group, _ = compute_M12_M16(
                tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=tag, verbose=False)
        dalphaX_vals = np.array([DALPHAX_GROUPS[g] for g in DALPHAX_GROUPS])
        dalphaX_sim_x = np.array([per_group[g][0] for g in DALPHAX_GROUPS])
        Offset12 = np.polyfit(dalphaX_vals, dalphaX_sim_x, 1)[1]
        dE_vals = np.array([DE_GROUPS[g] for g in DE_GROUPS])
        dE_sim_x = np.array([per_group[g][0] for g in DE_GROUPS])
        Offset16 = np.polyfit(dE_vals, dE_sim_x, 1)[1]
        return {"source": source, "iteration": iteration, "fp3_knob": fp3_knob, "sign": sign,
                "M12": M12, "M16": M16, "Offset12": Offset12, "Offset16": Offset16,
                "elapsed_s": time.time() - t0}
    except Exception as e:
        return ("FAILED", source, iteration, fp3_knob, sign, str(e))


if __name__ == "__main__":
    ensure_results_file_exists()
    candidates = load_1sigma_candidates()
    print(f"{len(candidates)} 1-sigma candidates available.")

    rng = np.random.default_rng(SAMPLE_SEED)
    keys = sorted(candidates.keys())
    sample_idx = rng.choice(len(keys), size=N_SAMPLE, replace=False)
    sample_keys = [keys[i] for i in sample_idx]

    extra_rng = np.random.default_rng(EXTRA_V7_SEED)
    v7_remaining = sorted(k for k in keys if k[0] == "v7" and k not in sample_keys)
    extra_v7 = v7_remaining[extra_rng.integers(len(v7_remaining))]
    sample_keys.append(extra_v7)

    print(f"Sampled {len(sample_keys)} candidates "
          f"({sum(1 for k in sample_keys if k[0]=='v1')}x v1, "
          f"{sum(1 for k in sample_keys if k[0]=='v7')}x v7): {sample_keys}\n")

    done = load_done()

    # Only bootstrap candidates that still have at least one dB-knob task
    # left to run (skip the wasted compile if everything's already done).
    needs_db_work = set()
    for key in sample_keys:
        for fp3_knob in FP3_KNOBS:
            if not fp3_knob.endswith('_Exit_dB'):
                continue
            for sign in (-1, 1):
                if (key[0], key[1], fp3_knob, sign) not in done:
                    needs_db_work.add(key)

    print(f"Bootstrapping {len(needs_db_work)} candidates' own B_SC lattice "
          f"(sequential, to avoid the dB sub-task race)...")
    base_cosyfiles = {}
    for key in sample_keys:
        if key not in needs_db_work:
            continue
        base_cosyfiles[key] = bootstrap_candidate(key, candidates[key])
        print(f"  {key}: bootstrapped -> {base_cosyfiles[key]}", flush=True)
    print()

    tasks = []
    for key in sample_keys:
        base_knobs = candidates[key]
        for fp3_knob in FP3_KNOBS:
            for sign in (-1, 1):
                if (key[0], key[1], fp3_knob, sign) in done:
                    continue
                tasks.append((key, base_knobs, fp3_knob, sign, base_cosyfiles.get(key)))

    print(f"{len(tasks)} points remaining to evaluate.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for i, result in enumerate(pool.imap_unordered(evaluate, tasks)):
                if isinstance(result, tuple) and result[0] == "FAILED":
                    print(f"  FAILED {result[1]}{result[2]} {result[3]} sign={result[4]}: {result[5]}", flush=True)
                    continue
                append_row(result)
                print(f"  [{i+1}/{len(tasks)}] {result['source']}{result['iteration']} "
                      f"{result['fp3_knob']} sign={result['sign']:+d}: "
                      f"M12={result['M12']:.4f} M16={result['M16']:.4f}", flush=True)

    print(f"\nDone. Run fp3_lean_jacobian_variability_summary.py next.")
