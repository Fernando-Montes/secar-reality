"""
Full-scale version of fp3_lean_jacobian_variability.py, per user request
2026-08-17: rather than accept a single shared FP3 Jacobian (or an
average across a handful of candidates), build the actual lean (2-point
central-difference) Jacobian for EVERY one of the 46 FP2 1-sigma-
accepted candidates (v1+v7 combined, chi2<=4.7196), so the eventual FP3
family search can sample among candidate-specific Jacobians instead of
assuming one is representative.

Same 7 FP3 knobs, same steps, same race-condition-safe bootstrap
pattern (sequential compile of each candidate's own B_SC lattice before
its dB-knob sub-tasks read it read-only) as the variability check --
but now with the bootstrap phase ALSO parallelized: different
candidates' Q1-Q7 values are (for all practical purposes) always
distinct, so their bootstrap compiles never target the same matrix
directory and can safely run concurrently. Only a given candidate's OWN
dB-knob sub-tasks need to wait for THAT candidate's bootstrap.

Cost (measured from the 6-candidate variability run): ~560s/point
average, 46 candidates x 7 knobs x 2 signs = 644 points -> ~10 hours at
10-way parallelism (plus a few minutes for the now-parallelized
bootstrap phase).

Checkpointed/resumable: same pattern as every other script here -- each
(candidate, knob, sign) point appended to RESULTS_FILE as soon as it
completes; already-done points are skipped on restart.

Run:
    cd AgentSearch
    python3 fp3_full_jacobian_build.py
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

NRAYS = 3000

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp3_full_jacobian_raw_results.csv")
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


def bootstrap_candidate(task):
    """Parallel-safe: different candidates' own B_SC (Q1-Q7) values are
    essentially always distinct, so their hash-based matrix directories
    never collide with each other -- only a given candidate's OWN
    dB-knob sub-tasks need to wait for its bootstrap, not for other
    candidates'."""
    key, base_knobs = task
    tuneChanges = [knob_to_tunechange(k, v) for k, v in base_knobs.items()]
    tag = f"fp3jac_boot_{key[0]}{key[1]}"[:60]
    _, _, _, usedFoxFile = mean_x_at_vd1783('10', tuneChanges, NRAYS, True, tag)
    return key, usedFoxFile


def evaluate(task):
    (source, iteration), base_knobs, fp3_knob, sign, base_cosyFile = task
    t0 = time.time()
    step = FP3_KNOB_STEP[fp3_knob] * sign
    tuneChanges = [knob_to_tunechange(k, v) for k, v in base_knobs.items()]
    tuneChanges.append(knob_to_tunechange(fp3_knob, step))
    is_db_knob = fp3_knob.endswith('_Exit_dB')
    try:
        if is_db_knob:
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
    all_keys = sorted(candidates.keys())
    print(f"{len(all_keys)} 1-sigma candidates: building the full Jacobian for all of them.\n")

    done = load_done()

    needs_db_work = set()
    for key in all_keys:
        for fp3_knob in FP3_KNOBS:
            if not fp3_knob.endswith('_Exit_dB'):
                continue
            for sign in (-1, 1):
                if (key[0], key[1], fp3_knob, sign) not in done:
                    needs_db_work.add(key)

    print(f"Bootstrapping {len(needs_db_work)} candidates' own B_SC lattice "
          f"(parallel across candidates, sequential only within each one)...")
    base_cosyfiles = {}
    if needs_db_work:
        boot_tasks = [(key, candidates[key]) for key in all_keys if key in needs_db_work]
        with Pool(processes=10) as pool:
            for i, (key, usedFoxFile) in enumerate(pool.imap_unordered(bootstrap_candidate, boot_tasks)):
                base_cosyfiles[key] = usedFoxFile
                print(f"  [{i+1}/{len(boot_tasks)}] {key}: bootstrapped -> {usedFoxFile}", flush=True)
    print()

    tasks = []
    for key in all_keys:
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

    print(f"\nDone. {len(all_keys)} candidates' Jacobians available in {RESULTS_FILE}.")
