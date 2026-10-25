"""
Corner extension scan (2026-09-21), follow-up to q14_q15_local_refine_scan.py:
that local grid found transmission to DSSD still climbing toward its
Q14=-18%, Q15=-10% corner (96.83%), i.e. the true optimum sits further out
in that direction (more negative Q14 AND more negative Q15 -- the earlier
1D q14_extension_scan.py had missed this because it held Q15 fixed at
+5%, where Q14's optimum was still -15%/89.4%). This script pushes both
knobs further: Q14 from -18% to -30% in 2% steps, Q15 from -10% to -30%
in 5% steps.

Since transmission is already near 97% and this is approaching a physical
ceiling (100%), expect the surface to flatten or turn over somewhere in
this range rather than keep rising indefinitely -- that's the thing this
scan is checking for.

Same conventions throughout: default stepped_uniform beam, target -> DSSD,
corrected boundary convention (transmission_indices > DSSD_INDEX).

Checkpointed/resumable + parallelized across N_WORKERS separate processes
(see q14_q15_local_refine_scan.py for why processes, not threads: the
os.chdir() in run_cosy_and_save_fort is whole-process state).

Run yourself (per project convention -- COSY/ECAT + multi-point searches
are user-run, checkpointed so they can be stopped/restarted):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 q14_q15_corner_extension_scan.py
"""
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import element_names_stripped
from helper_functions import runECAT

N_WORKERS = 10

DSSD_INDEX = element_names_stripped.index('DSSD')

Q14_VALUES = [-0.18, -0.20, -0.22, -0.24, -0.26, -0.28, -0.30]
Q15_VALUES = [-0.10, -0.15, -0.20, -0.25, -0.30]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "q14_q15_corner_extension_scan_results.csv")
FIELDNAMES = ["Q14_B_SC", "Q15_B_SC", "n_launched", "n_to_dssd", "transmission_pct"]

# points already computed by the local refine scan -- seed so this file's
# own results/plot are self-contained without recompiling
PRIOR_SOURCES = ["Results/q14_q15_local_refine_scan_results.csv"]


def evaluate(q14, q15):
    tuneChanges = []
    if q14 != 0.0:
        tuneChanges.append(['Q14', 'B_SC', float(q14)])
    if q15 != 0.0:
        tuneChanges.append(['Q15', 'B_SC', float(q15)])

    tag = f"pg_June2025_q1415corner_{q14:+.3f}_{q15:+.3f}".replace(".", "p")
    cosyFile, run_cosy_flag = tag[:60], True

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)
    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    return {"Q14_B_SC": q14, "Q15_B_SC": q15, "n_launched": n_total,
            "n_to_dssd": n_to_dssd, "transmission_pct": 100 * n_to_dssd / n_total}


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set((round(float(r["Q14_B_SC"]), 6), round(float(r["Q15_B_SC"]), 6))
                   for r in csv.DictReader(f))


def seed_from_prior_scans(done):
    wanted = {(round(q14, 6), round(q15, 6)) for q14 in Q14_VALUES for q15 in Q15_VALUES}
    with open(RESULTS_FILE, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        for source in PRIOR_SOURCES:
            if not os.path.exists(source):
                continue
            with open(source, "r", newline="") as f:
                for r in csv.DictReader(f):
                    key = (round(float(r["Q14_B_SC"]), 6), round(float(r["Q15_B_SC"]), 6))
                    if key in wanted and key not in done:
                        writer.writerow({k: r[k] for k in FIELDNAMES})
                        done.add(key)
                        print(f"seeded Q14={key[0]*100:+.1f}%, Q15={key[1]*100:+.1f}% "
                              f"from {source} ({r['transmission_pct']}%)")
    return done


if __name__ == "__main__":
    ensure_results_file_exists()
    done = load_done()
    done = seed_from_prior_scans(done)

    tasks = [(q14, q15) for q14 in Q14_VALUES for q15 in Q15_VALUES
             if (round(q14, 6), round(q15, 6)) not in done]
    print(f"\n{len(done)} points already done, {len(tasks)} remaining.\n")

    with open(RESULTS_FILE, "a", newline="") as f, ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        futures = {pool.submit(evaluate, q14, q15): (q14, q15) for q14, q15 in tasks}
        print(f"submitted {len(futures)} points across {N_WORKERS} worker processes\n", flush=True)
        for future in as_completed(futures):
            q14, q15 = futures[future]
            row = future.result()
            writer.writerow(row)
            f.flush()
            print(f"Q14={q14*100:+.1f}%  Q15={q15*100:+.1f}%: "
                  f"transmission = {row['n_to_dssd']}/{row['n_launched']} = {row['transmission_pct']:.2f}%",
                  flush=True)

    print(f"\nDone. Wrote {RESULTS_FILE}")
