"""
Q14 1D extension scan (2026-09-21), follow-up to q14_q15_transmission_scan.py:
that 5x5 grid showed transmission to DSSD is dominated by Q14 (Q15 only
shifts things by ~1-2 points across its whole +-10% range) and rises
monotonically as Q14 goes more negative, with the grid's best point sitting
at its edge (Q14=-10%, Q15=+5% -> 83.60%). This script holds Q15 fixed at
+5% (best from the grid) and pushes Q14 further negative to find where
transmission peaks or plateaus/reverses.

Same conventions as q14_q15_transmission_scan.py: default stepped_uniform
beam, target -> DSSD, corrected boundary convention
(transmission_indices > DSSD_INDEX).

Checkpointed/resumable: each Q14 point appended to RESULTS_FILE as soon as
it completes -- safe to Ctrl-C and re-run later; already-done points are
skipped.

Run (per project convention, run this yourself rather than backgrounding it
-- COSY/ECAT runs and multi-point searches are user-run on this laptop so
they can be stopped/restarted):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 q14_extension_scan.py
"""
import os
import csv

from config import element_names_stripped
from helper_functions import runECAT

DSSD_INDEX = element_names_stripped.index('DSSD')

Q15_FIXED = 0.05
Q14_VALUES = [-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "q14_extension_scan_results.csv")
FIELDNAMES = ["Q14_B_SC", "Q15_B_SC", "n_launched", "n_to_dssd", "transmission_pct"]


def evaluate(q14, q15):
    tuneChanges = [['Q14', 'B_SC', float(q14)]]
    if q15 != 0.0:
        tuneChanges.append(['Q15', 'B_SC', float(q15)])

    tag = f"pg_June2025_q14ext_{q14:+.3f}_{q15:+.3f}".replace(".", "p")
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


if __name__ == "__main__":
    ensure_results_file_exists()
    done = load_done()

    # the Q14=-10%, Q15=+5% point was already computed in
    # q14_q15_transmission_scan.py -- seed it here too so this file's own
    # results/plot are self-contained, without re-running COSY for it
    seed = (round(-0.10, 6), round(Q15_FIXED, 6))
    if seed not in done:
        with open("Results/q14_q15_transmission_scan_results.csv", "r", newline="") as f:
            for r in csv.DictReader(f):
                if (round(float(r["Q14_B_SC"]), 6), round(float(r["Q15_B_SC"]), 6)) == seed:
                    with open(RESULTS_FILE, "a", newline="") as out:
                        csv.DictWriter(out, fieldnames=FIELDNAMES).writerow(r)
                    done.add(seed)
                    print(f"seeded Q14=-10%, Q15=+5% from existing grid scan "
                          f"({r['transmission_pct']}%)\n")
                    break

    tasks = [q14 for q14 in Q14_VALUES if (round(q14, 6), round(Q15_FIXED, 6)) not in done]
    print(f"{len(done)} points already done, {len(tasks)} remaining.\n")

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for q14 in tasks:
            print(f"Q14={q14*100:+.1f}%  Q15={Q15_FIXED*100:+.1f}% ...", flush=True)
            row = evaluate(q14, Q15_FIXED)
            writer.writerow(row)
            f.flush()
            print(f"  transmission = {row['n_to_dssd']}/{row['n_launched']} = {row['transmission_pct']:.2f}%\n",
                  flush=True)

    print(f"Done. Wrote {RESULTS_FILE}")
