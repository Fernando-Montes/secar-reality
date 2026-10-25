"""
Local refinement scan (2026-09-21), follow-up to q14_extension_scan.py:
that 1D scan (Q15 fixed at +5%) found transmission to DSSD peaks near
Q14=-15% (89.42%), bracketed by -10% (83.60%) and -20% (86.77%) on either
side -- i.e. the true peak is somewhere in [-20%, -10%], not necessarily
exactly at -15%. This script does a finer 2D grid around that region:
Q14 in 1% steps from -12% to -18%, crossed with a wider Q15 range than
before (-10% to +15% in 5% steps) to check whether Q15 has more leverage
near the true optimum than it did in the original +-10% grid (where its
effect was weak, ~1-2 points).

Same conventions throughout: default stepped_uniform beam, target -> DSSD,
corrected boundary convention (transmission_indices > DSSD_INDEX).

Checkpointed/resumable: each (Q14,Q15) point written to RESULTS_FILE (by
the main process, flushed immediately) as soon as it completes;
already-done points (including overlaps with q14_q15_transmission_scan.py
/ q14_extension_scan.py, seeded below) are skipped.

Parallelized across N_WORKERS separate OS processes (ProcessPoolExecutor,
not threading): run_cosy_and_save_fort() in helper_functions.py does
os.chdir() around the COSY subprocess call, which is whole-process state
-- plain threads share one process and would race/corrupt each other's
compiles, so this needs real separate processes, each with its own cwd.
Default is 4 workers, leaving headroom on this 14-core laptop since it's
also in active use; adjust N_WORKERS below if desired.

Run yourself (per project convention -- COSY/ECAT + multi-point searches
are user-run, checkpointed so they can be stopped/restarted):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 q14_q15_local_refine_scan.py
"""
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import element_names_stripped
from helper_functions import runECAT

N_WORKERS = 10

DSSD_INDEX = element_names_stripped.index('DSSD')

Q14_VALUES = [-0.12, -0.13, -0.14, -0.15, -0.16, -0.17, -0.18]
Q15_VALUES = [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "q14_q15_local_refine_scan_results.csv")
FIELDNAMES = ["Q14_B_SC", "Q15_B_SC", "n_launched", "n_to_dssd", "transmission_pct"]

# points already computed by earlier scans in this series -- seed them here
# so this file's own results/plot are self-contained without recompiling
PRIOR_SOURCES = ["Results/q14_q15_transmission_scan_results.csv",
                  "Results/q14_extension_scan_results.csv"]


def evaluate(q14, q15):
    tuneChanges = []
    if q14 != 0.0:
        tuneChanges.append(['Q14', 'B_SC', float(q14)])
    if q15 != 0.0:
        tuneChanges.append(['Q15', 'B_SC', float(q15)])

    tag = f"pg_June2025_q1415ref_{q14:+.3f}_{q15:+.3f}".replace(".", "p")
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
