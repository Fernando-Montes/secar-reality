"""
Quad field-error study, ALL 15 quads (2026-09-21), generalizing
q6_field_error_scan.py: for each quad Q1-Q15, sweep that quad's OWN
B_SC by +-1% in 0.1% steps (21 points) around its OWN operating point --
0% (nominal) for every quad except Q14 and Q15, which operate at the
optimized -20% (see q14_q15_corner_extension_scan.py) -- while holding
every OTHER quad fixed at ITS operating point (Q14=-20%, Q15=-20%,
everything else nominal). So this answers "how sensitive is the
already-optimized tune to a +-1% field error in quad X", one quad at a
time, on the same footing.

Same beam/metrics as q6_field_error_scan.py:
  - overall transmission, target -> DSSD (real DSSD aperture)
  - FP2 resolution = |ME(1,7)|/WV at FP2 Slits
  - FP3 resolution = |ME(1,7)|/WV at FP3 Slits
  - DSSD mean X / range X, BOTH the real aperture AND a hypothetical
    50cm x 50cm aperture (see that script's docstring for why
    all_x[i][DSSD_INDEX] is available for every ray regardless of the
    real aperture blocking it, letting the 50cm re-cut be computed with
    no extra COSY run)
WV = max-min range (not std). Transmission boundary convention:
transmission_indices[i] > target_index (not >=).

Q6's 21 points are identical to what q6_field_error_scan.py already
computed (same base tune, same delta grid) -- seeded from
Results/q6_field_error_scan_results.csv instead of recompiling.

15 quads x 21 points = 315 total points, 21 seeded (Q6) -> 294 fresh COSY
compiles. This is a substantially bigger run than anything earlier in
this series -- flagging the size since it will take a while even
parallelized.

Checkpointed/resumable (one combined CSV, (quad, delta) as the resume
key) + parallelized across N_WORKERS separate processes (all 294 tasks
submitted to one pool at once, not quad-by-quad, for max parallel
efficiency). Same process-not-thread rationale as
q14_q15_local_refine_scan.py (run_cosy_and_save_fort's os.chdir() is
whole-process state).

Run yourself (per project convention -- COSY/ECAT runs and multi-point
searches are user-run, checkpointed so they can be stopped/restarted):
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 quad_field_error_scan_all.py
"""
import os
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import element_names_stripped, pkl_directory
from helper_functions import runECAT, load_optics_data
from compute_me17_cumulative import cumulative_linear_matrix

N_WORKERS = 10

QUADS = [f"Q{i}" for i in range(1, 16)]
OPTIMIZED_BASE = {"Q14": -0.20, "Q15": -0.20}  # everything else defaults to 0.0 (nominal)

DSSD_HALF_SIZE_MM = 250.0  # 50cm x 50cm square, centered on the beam axis

FP2_SLITS_INDEX = element_names_stripped.index('FP2 Slits')
FP3_SLITS_INDEX = element_names_stripped.index('FP3 Slits & VD1783')
DSSD_INDEX = element_names_stripped.index('DSSD')

DELTA_VALUES = [round(v, 4) for v in np.arange(-0.010, 0.0101, 0.001)]

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "quad_field_error_scan_all_results.csv")
SEED_SOURCE = os.path.join(OUTPUT_DIR, "q6_field_error_scan_results.csv")
FIELDNAMES = ["quad", "delta", "n_launched", "n_to_dssd", "transmission_pct",
              "fp2_ME17", "fp2_WV_mm", "fp2_resolution",
              "fp3_ME17", "fp3_WV_mm", "fp3_resolution",
              "dssd_mean_x_mm", "dssd_range_x_mm",
              "n_to_dssd_50cm", "transmission_pct_50cm",
              "dssd_mean_x_50cm_mm", "dssd_range_x_50cm_mm"]


def evaluate(quad, delta):
    tuneChanges = []
    for q in ("Q14", "Q15"):
        value = OPTIMIZED_BASE[q] + delta if q == quad else OPTIMIZED_BASE[q]
        tuneChanges.append([q, 'B_SC', value])
    if quad not in ("Q14", "Q15"):
        value = OPTIMIZED_BASE.get(quad, 0.0) + delta
        if value != 0.0:
            tuneChanges.append([quad, 'B_SC', value])

    tag = f"pg_June2025_qerr_{quad}_{delta:+.4f}".replace(".", "p")
    cosyFile, run_cosy_flag = tag[:60], True

    all_x, all_ax, all_y, all_ay, all_dE, transmitted_x, transmitted_y, transmission_indices, \
        chamber_names, beampipes, usedFoxFile = runECAT(
            initialDistribution=None, run_cosy_flag=run_cosy_flag, foxFile=cosyFile,
            tuneChanges=tuneChanges, save_rays_flag=False, final_index=DSSD_INDEX)

    n_total = len(transmission_indices)

    def x_at(idx):
        return np.array([-x[idx] * 1000 for i, x in enumerate(all_x) if transmission_indices[i] > idx])

    n_to_dssd = sum(1 for t in transmission_indices if t > DSSD_INDEX)
    transmission_pct = 100 * n_to_dssd / n_total

    x_dssd = x_at(DSSD_INDEX)
    x_fp2 = x_at(FP2_SLITS_INDEX)
    x_fp3 = x_at(FP3_SLITS_INDEX)
    fp2_wv = x_fp2.max() - x_fp2.min()
    fp3_wv = x_fp3.max() - x_fp3.min()

    _, gdf = load_optics_data(usedFoxFile, pkl_directory)
    fp2_me17 = cumulative_linear_matrix(gdf, FP2_SLITS_INDEX)[0, 6]
    fp3_me17 = cumulative_linear_matrix(gdf, FP3_SLITS_INDEX)[0, 6]

    candidates = [i for i, t in enumerate(transmission_indices) if t >= DSSD_INDEX]
    x_50cm = []
    for i in candidates:
        x_mm = -all_x[i][DSSD_INDEX] * 1000
        y_mm = -all_y[i][DSSD_INDEX] * 1000
        if abs(x_mm) <= DSSD_HALF_SIZE_MM and abs(y_mm) <= DSSD_HALF_SIZE_MM:
            x_50cm.append(x_mm)
    x_50cm = np.array(x_50cm)
    n_to_dssd_50cm = len(x_50cm)

    return {
        "quad": quad, "delta": delta, "n_launched": n_total, "n_to_dssd": n_to_dssd,
        "transmission_pct": transmission_pct,
        "fp2_ME17": fp2_me17, "fp2_WV_mm": fp2_wv, "fp2_resolution": abs(fp2_me17) / (fp2_wv / 1000),
        "fp3_ME17": fp3_me17, "fp3_WV_mm": fp3_wv, "fp3_resolution": abs(fp3_me17) / (fp3_wv / 1000),
        "dssd_mean_x_mm": x_dssd.mean(), "dssd_range_x_mm": x_dssd.max() - x_dssd.min(),
        "n_to_dssd_50cm": n_to_dssd_50cm, "transmission_pct_50cm": 100 * n_to_dssd_50cm / n_total,
        "dssd_mean_x_50cm_mm": x_50cm.mean(), "dssd_range_x_50cm_mm": x_50cm.max() - x_50cm.min(),
    }


def ensure_results_file_exists():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def load_done():
    with open(RESULTS_FILE, "r", newline="") as f:
        return set((r["quad"], round(float(r["delta"]), 4)) for r in csv.DictReader(f))


def seed_q6_from_existing_scan(done):
    if not os.path.exists(SEED_SOURCE):
        return done
    with open(RESULTS_FILE, "a", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDNAMES)
        with open(SEED_SOURCE, "r", newline="") as f:
            for r in csv.DictReader(f):
                delta = round(float(r["Q6_B_SC"]), 4)
                key = ("Q6", delta)
                if key in done:
                    continue
                writer.writerow({
                    "quad": "Q6", "delta": delta,
                    "n_launched": r["n_launched"], "n_to_dssd": r["n_to_dssd"],
                    "transmission_pct": r["transmission_pct"],
                    "fp2_ME17": r["fp2_ME17"], "fp2_WV_mm": r["fp2_WV_mm"], "fp2_resolution": r["fp2_resolution"],
                    "fp3_ME17": r["fp3_ME17"], "fp3_WV_mm": r["fp3_WV_mm"], "fp3_resolution": r["fp3_resolution"],
                    "dssd_mean_x_mm": r["dssd_mean_x_mm"], "dssd_range_x_mm": r["dssd_range_x_mm"],
                    "n_to_dssd_50cm": r["n_to_dssd_50cm"], "transmission_pct_50cm": r["transmission_pct_50cm"],
                    "dssd_mean_x_50cm_mm": r["dssd_mean_x_50cm_mm"],
                    "dssd_range_x_50cm_mm": r["dssd_range_x_50cm_mm"],
                })
                done.add(key)
                print(f"seeded Q6, delta={delta*100:+.1f}% from {SEED_SOURCE}")
    return done


if __name__ == "__main__":
    ensure_results_file_exists()
    done = load_done()
    done = seed_q6_from_existing_scan(done)

    tasks = [(quad, delta) for quad in QUADS for delta in DELTA_VALUES if (quad, delta) not in done]
    print(f"\n{len(done)} points already done, {len(tasks)} remaining "
          f"(out of {len(QUADS) * len(DELTA_VALUES)} total).\n")

    with open(RESULTS_FILE, "a", newline="") as f, ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        futures = {pool.submit(evaluate, quad, delta): (quad, delta) for quad, delta in tasks}
        print(f"submitted {len(futures)} points across {N_WORKERS} worker processes\n", flush=True)
        n_done = 0
        for future in as_completed(futures):
            quad, delta = futures[future]
            row = future.result()
            writer.writerow(row)
            f.flush()
            n_done += 1
            print(f"[{n_done}/{len(futures)}] {quad} delta={delta*100:+.2f}%: "
                  f"transmission={row['transmission_pct']:.2f}%  "
                  f"FP2 res={row['fp2_resolution']:.2f}  FP3 res={row['fp3_resolution']:.2f}",
                  flush=True)

    print(f"\nDone. Wrote {RESULTS_FILE}")
