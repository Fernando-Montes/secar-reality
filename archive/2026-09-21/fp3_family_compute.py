"""
Computes M12/M16 at FP3 (VD1783) for the 44 candidates (v1: 24, v7: 20)
that pass E2 within 1-sigma (chi2 <= 4.7196, df=4) at FP2 -- per user
request 2026-08-14, to overlay their FP3 response as faint gray family
lines on the FP3 slope plot, same style as fp2_slope_plot_with_family.py.

Each candidate's own knob values (from fp2_family_mc_results.csv /
fp2_family_mc_v7_results.csv) are converted to tuneChanges and run
through fp3_metrics.compute_M12_M16 (10 groups: the same
ALL_GROUPS used for the FP3 Experimental fit). This is a real,
per-candidate COSY computation (compile + propagate for 10 groups each)
-- checkpointed/resumable the same way as fp2_family_e3_wide_recheck.py,
parallelized across candidates (each candidate itself runs sequentially:
first group triggers the compile, the rest reuse it).

Run:
    cd AgentSearch
    python3 fp3_family_compute.py
"""
import os
import csv
import numpy as np
from multiprocessing import Pool

from fp3_metrics import compute_M12_M16, DALPHAX_GROUPS, DE_GROUPS

CHI2_THRESHOLD_1SIGMA_DF4 = 4.7196

V1_FILE = "Results/fp2_family_mc_results.csv"
V1_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']
V7_FILE = "Results/fp2_family_mc_v7_results.csv"
V7_KNOBS = ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
            'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']

NRAYS = 3000
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "fp3_family_1sigma_results.csv")
FIELDNAMES = ["source", "iteration", "M12_sim", "M16_sim", "Offset12_sim", "Offset16_sim"]


def knob_to_tunechange(knob_id, value):
    if knob_id.endswith('_Exit_dB'):
        elem = knob_id[: -len('_Exit_dB')]
        return [f'{elem} Exit', 'dB', float(value)]
    return [knob_id.replace('_B_SC', ''), 'B_SC', float(value)]


def load_1sigma_candidates(path, knobs, source):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        if r["loss"] in (None, "") or float(r["loss"]) > CHI2_THRESHOLD_1SIGMA_DF4:
            continue
        iteration = int(r["iteration"])
        out[(source, iteration)] = {k: float(r[k]) for k in knobs}
    return out


def ensure_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set((row["source"], int(row["iteration"])) for row in csv.DictReader(f))


def append_row(row):
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def evaluate(task):
    (source, iteration), knobs = task
    tuneChanges = [knob_to_tunechange(k, v) for k, v in knobs.items()]
    tag = f"fp3fam_{source}{iteration}"[:60]
    try:
        M12, M16, per_group, _ = compute_M12_M16(
            tuneChanges=tuneChanges, nRays=NRAYS, cosyFile=tag, verbose=False)
        dalphaX_vals = np.array([DALPHAX_GROUPS[g] for g in DALPHAX_GROUPS])
        dalphaX_sim_x = np.array([per_group[g][0] for g in DALPHAX_GROUPS])
        Offset12 = np.polyfit(dalphaX_vals, dalphaX_sim_x, 1)[1]
        dE_vals = np.array([DE_GROUPS[g] for g in DE_GROUPS])
        dE_sim_x = np.array([per_group[g][0] for g in DE_GROUPS])
        Offset16 = np.polyfit(dE_vals, dE_sim_x, 1)[1]
        return {"source": source, "iteration": iteration, "M12_sim": M12, "M16_sim": M16,
                "Offset12_sim": Offset12, "Offset16_sim": Offset16}
    except Exception as e:
        return ("FAILED", source, iteration, str(e))


if __name__ == "__main__":
    ensure_file_exists()
    candidates = {}
    candidates.update(load_1sigma_candidates(V1_FILE, V1_KNOBS, "v1"))
    candidates.update(load_1sigma_candidates(V7_FILE, V7_KNOBS, "v7"))
    done = load_done()
    print(f"{len(candidates)} 1-sigma candidates (v1+v7). {len(done)} already computed.\n")

    tasks = [(key, knobs) for key, knobs in candidates.items() if key not in done]
    print(f"{len(tasks)} remaining to compute.\n")

    if tasks:
        with Pool(processes=10) as pool:
            for i, result in enumerate(pool.imap_unordered(evaluate, tasks)):
                if isinstance(result, tuple) and result[0] == "FAILED":
                    print(f"  FAILED {result[1]}{result[2]}: {result[3]}", flush=True)
                    continue
                append_row(result)
                print(f"  [{i+1}/{len(tasks)}] {result['source']}{result['iteration']}: "
                      f"M12={result['M12_sim']:.4f} M16={result['M16_sim']:.4f}", flush=True)

    print(f"\nDone. See {RESULTS_FILE}")
