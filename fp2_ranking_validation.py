"""
Validation of the FP3 ranking methodology by redoing the FP2 ranking
(2026-10-04): same 65 knobs, same loss, same beam as the original
oat_derivative_ranking.png (archive/2026-09-21/fp2_oat_screen.py +
plot_oat_derivative_ranking.py), but through the NEW pipeline used for FP3:
corrected .fox base (pg_June2025_corrected; identical to the old file
upstream of WF2, so FP2 should not change), fast_tracer, single-pass
evaluator (fp3_eval, plane='FP2'), central differences.

Original setup reproduced: at nominal; 2-term loss
((M12+0.2472)/0.02)^2 + ((M16+0.3842)/0.03)^2 (the original's constant
Q6-prior term is dropped -- it is identical everywhere and has no
derivative); E2 FP2 groups at VD1638, 1000 rays per group, TA_Y offset 0.

Steps: each knob's two grid points closest to nominal from the original
10-point OAT grid (+-R/9: 1.111% B_SC, 0.1111% dB, 0.333 mm dX/dY,
0.0556 deg roll, 2.22 mm WF1 length). Every point evaluated here therefore
also exists in the archived OAT results, which gives a point-by-point
check of the new pipeline against the old one (plot_fp2_ranking_validation.py).

1 compile for nominal (dB knobs are traced on it) + 2 per lattice knob.
Checkpointed/resumable (keyed by knob, value), parallel over processes,
compile files deleted after use.

Run yourself:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 fp2_ranking_validation.py
"""
import os
import csv
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

ARCHIVED_OAT = "Results/archive/2026-09-21/fp2_oat_screen_results.csv"
RESULTS_FILE = "Results/fp2_ranking_validation_results.csv"
FIELDNAMES = ["knob", "value", "M12", "Offset12", "M16", "Offset16", "loss", "task_elapsed_s"]
NRAYS = 1000
M12_TARGET, M16_TARGET = -0.2472, -0.3842
SIGMA_M12, SIGMA_M16 = 0.02, 0.03


def loss_fp2(obs):
    return ((obs[0] - M12_TARGET) / SIGMA_M12) ** 2 + ((obs[2] - M16_TARGET) / SIGMA_M16) ** 2


def tunechange(knob, v):
    if knob == 'WF1_length':
        return ['WF1', 'length', v]
    elem, par = knob.rsplit('_', 1)
    if knob.endswith('_Exit_dB'):
        return [f"{knob[:-len('_Exit_dB')]} Exit", 'dB', v]
    if knob.endswith('_B_SC'):
        return [knob[:-len('_B_SC')], 'B_SC', v]
    return [elem, {'Roll': 'Roll', 'dX': 'X', 'dY': 'Y'}[par], v]


def load_steps():
    """knob -> (minus value, plus value): the archived grid points closest to nominal."""
    vals = {}
    for r in csv.DictReader(open(ARCHIVED_OAT)):
        if r["knob_id"] != "NOMINAL":
            vals.setdefault(r["knob_id"], []).append(float(r["value"]))
    steps = {}
    for k, v in vals.items():
        plus = min(x for x in v if x > 0)
        minus = max(x for x in v if x < 0)
        steps[k] = (minus, plus)
    assert len(steps) == 65, len(steps)
    return steps


def run_task(task):
    import fp3_eval
    kind, knob, values = task
    t0 = time.time()
    if kind == 'base':
        variants = [('NOMINAL', 0.0, [])] + [(k, v, [tunechange(k, v)]) for k, v in values]
        res = fp3_eval.evaluate_variants([], [x[2] for x in variants], 'FP2', NRAYS, loss_fp2)
        points = [(x[0], x[1]) for x in variants]
    else:
        res = fp3_eval.evaluate_variants([tunechange(knob, values)], [[]], 'FP2', NRAYS, loss_fp2)
        points = [(knob, values)]
    el = round(time.time() - t0, 1)
    return [{"knob": k, "value": v, "M12": o[0], "Offset12": o[1], "M16": o[2], "Offset16": o[3],
             "loss": l, "task_elapsed_s": el} for (k, v), (o, l, _) in zip(points, res)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    steps = load_steps()
    db_knobs = sorted(k for k in steps if k.endswith('_Exit_dB'))
    lattice = sorted(k for k in steps if k not in db_knobs)

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    done = {(r["knob"], round(float(r["value"]), 8)) for r in csv.DictReader(open(RESULTS_FILE))}

    tasks = []
    db_points = [(k, v) for k in db_knobs for v in steps[k]]
    if ('NOMINAL', 0.0) not in done or any((k, round(v, 8)) not in done for k, v in db_points):
        tasks.append(('base', None, db_points))
    for k in lattice:
        for v in steps[k]:
            if (k, round(v, 8)) not in done:
                tasks.append(('knob', k, v))
    print(f"{len(steps)} knobs ({len(lattice)} lattice, {len(db_knobs)} dB): "
          f"{1 + 2 * len(lattice)} compiles total, {len(tasks)} tasks to do, {args.workers} workers.\n", flush=True)

    t_start = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_task, t): t for t in tasks}
    n = 0
    try:
        with open(RESULTS_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            for fut in as_completed(futures):
                kind, knob, v = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"FAILED {kind} {knob} {v}: {e}", flush=True)
                    continue
                w.writerows(rows)
                f.flush()
                n += 1
                wall = time.time() - t_start
                label = 'nominal+dB' if kind == 'base' else f"{knob} {v:+.6g}"
                print(f"[{n}/{len(tasks)}] {label:<22} task {rows[0]['task_elapsed_s']:5.1f} s | "
                      f"elapsed {wall/60:4.1f} min, ETA {wall/n*(len(tasks)-n)/60:4.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\nStopping (finished points saved; re-run to resume).", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    pool.shutdown()
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
