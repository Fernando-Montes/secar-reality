"""
FP3 knob ranking by local derivative dLoss/dknob (2026-10-04), on the
corrected lattice (pg_June2025_corrected.fox, WF2 aligned), at each of the
46 FP2 1-sigma candidates (24 from fp2_family_mc v1, 22 from v7; FP2 is
upstream of WF2 so those solutions are unaffected by the .fox correction).

Loss: E2 FP3 Mahalanobis loss (fp3_loss.py). Observables M12/Off12/M16/Off16
at VD1783 from the same per-group aperture beams as the FP2 family search
(fp2_metrics.build_initial_distribution, 3000 rays, seed 42, TA_Y offset 0).

Knobs (42), chosen by the user 2026-10-04 -- everything between FP2 and FP3:
  Q8-Q11 strength (B_SC)                 step 1%
  B5, B6, WF2 strength (Exit dB)         step 0.25%
  X, Y of Q8-Q11, B5, B6, WF2            step 0.25 mm
  Roll                                   step 0.1 deg
  Yaw, Pitch                             step 0.05 deg
Central differences: each knob at +step and -step, everything else at the
candidate's FP2 solution (FP3-section knobs at nominal).

Every FP3 knob is downstream of the FP1 slits, so the same rays pass the
slit cut in every evaluation, and the rays themselves are seeded: the
derivatives carry essentially no sampling noise.

Task layout: per candidate, one "base" task (1 compile; traces the base
point and the six dB +-step points, which only change tracing) and one
task per (lattice knob, sign) (1 compile each): 1 + 78 compiles.

Checkpointed/resumable: each evaluated point is appended to RESULTS_FILE
(keyed by source, iteration, knob, sign) as soon as its task finishes;
finished points are skipped on restart. Parallel over N worker processes
(processes, not threads: runECAT's COSY step uses os.chdir). Each compile's
files are deleted right after use (fp3_eval).

Run yourself (COSY runs are user-run). Time one candidate first:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 fp3_knob_ranking.py --candidates 1
then all 46:
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 fp3_knob_ranking.py
Ctrl-C stops (finished points are kept); re-run to resume.
"""
import os
import csv
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

ARCHIVE = "Results/archive/2026-09-21/"
FP2_SOURCES = [
    (ARCHIVE + "fp2_family_mc_results.csv", "v1",
     ['Q1_B_SC', 'Q2_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC', 'WF1_Exit_dB']),
    (ARCHIVE + "fp2_family_mc_v7_results.csv", "v7",
     ['Q1_B_SC', 'Q2_B_SC', 'Q3_B_SC', 'Q4_B_SC', 'Q5_B_SC', 'Q6_B_SC', 'Q7_B_SC',
      'WF1_Exit_dB', 'B1_Exit_dB', 'B2_Exit_dB', 'B3_Exit_dB', 'B4_Exit_dB']),
]
FP2_1SIGMA = 4.7196

RESULTS_FILE = "Results/fp3_knob_ranking_results.csv"
FIELDNAMES = ["source", "iteration", "knob", "sign", "step", "unit",
              "M12", "Offset12", "M16", "Offset16", "loss", "task_elapsed_s"]

# knob -> (tuneChange builder, step in native .fox/tuneChanges units, display unit, factor native->display)
KNOBS = {}
for q in ('Q8', 'Q9', 'Q10', 'Q11'):
    KNOBS[f"{q}_B_SC"] = ((lambda v, q=q: [q, 'B_SC', v]), 0.01, '%', 100)
for e in ('B5', 'B6', 'WF2'):
    KNOBS[f"{e}_Exit_dB"] = ((lambda v, e=e: [f'{e} Exit', 'dB', v]), 0.0025, '%', 100)
for e in ('Q8', 'Q9', 'Q10', 'Q11', 'B5', 'B6', 'WF2'):
    KNOBS[f"{e}_X"] = ((lambda v, e=e: [e, 'X', v]), 0.00025, 'mm', 1000)
    KNOBS[f"{e}_Y"] = ((lambda v, e=e: [e, 'Y', v]), 0.00025, 'mm', 1000)
    KNOBS[f"{e}_Roll"] = ((lambda v, e=e: [e, 'Roll', v]), 0.1, 'deg', 1)
    KNOBS[f"{e}_Yaw"] = ((lambda v, e=e: [e, 'Yaw', v]), 0.05, 'deg', 1)
    KNOBS[f"{e}_Pitch"] = ((lambda v, e=e: [e, 'Pitch', v]), 0.05, 'deg', 1)
DB_KNOBS = [k for k in KNOBS if k.endswith('_Exit_dB')]
LATTICE_KNOBS = [k for k in KNOBS if k not in DB_KNOBS]


def fp2_tunechange(knob, value):
    if knob.endswith('_Exit_dB'):
        return [f"{knob[:-len('_Exit_dB')]} Exit", 'dB', value]
    return [knob[:-len('_B_SC')], 'B_SC', value]


def load_candidates():
    cands = []
    for path, src, knobs in FP2_SOURCES:
        for r in csv.DictReader(open(path)):
            if r["loss"] not in (None, "") and float(r["loss"]) <= FP2_1SIGMA:
                cands.append(((src, int(r["iteration"])),
                              [fp2_tunechange(k, float(r[k])) for k in knobs]))
    cands.sort(key=lambda c: (c[0][0], c[0][1]))
    assert len(cands) == 46, f"expected 46 FP2 1-sigma candidates, found {len(cands)}"
    return cands


def run_task(task):
    """task = (cand_key, fp2_tc, kind, knob, sign). Returns list of result rows."""
    import fp3_eval
    cand_key, fp2_tc, kind, knob, sign = task
    t0 = time.time()
    if kind == 'base':
        variants = [('base', 0, [])]
        for k in DB_KNOBS:
            build, step, _, _ = KNOBS[k]
            variants += [(k, +1, [build(+step)]), (k, -1, [build(-step)])]
        results = fp3_eval.evaluate_variants(fp2_tc, [v[2] for v in variants])
        points = [(v[0], v[1]) for v in variants]
    else:
        build, step, _, _ = KNOBS[knob]
        results = fp3_eval.evaluate_variants(fp2_tc + [build(sign * step)], [[]])
        points = [(knob, sign)]
    elapsed = time.time() - t0
    rows = []
    for (k, s), (obs, loss, _) in zip(points, results):
        step, unit = (0.0, '') if k == 'base' else (KNOBS[k][1], KNOBS[k][2])
        rows.append({"source": cand_key[0], "iteration": cand_key[1], "knob": k, "sign": s,
                     "step": step, "unit": unit, "M12": obs[0], "Offset12": obs[1],
                     "M16": obs[2], "Offset16": obs[3], "loss": loss,
                     "task_elapsed_s": round(elapsed, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=46, help="how many candidates (in fixed order)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    cands = load_candidates()[:args.candidates]
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    done = {(r["source"], int(r["iteration"]), r["knob"], int(r["sign"]))
            for r in csv.DictReader(open(RESULTS_FILE))}

    tasks = []
    for key, tc in cands:
        if not ((key[0], key[1], 'base', 0) in done and
                all((key[0], key[1], k, s) in done for k in DB_KNOBS for s in (1, -1))):
            tasks.append((key, tc, 'base', None, 0))
        for k in LATTICE_KNOBS:
            for s in (1, -1):
                if (key[0], key[1], k, s) not in done:
                    tasks.append((key, tc, 'knob', k, s))
    n_total = len(cands) * (1 + 2 * len(LATTICE_KNOBS))
    print(f"{len(cands)} candidate(s), {len(KNOBS)} knobs: {n_total} compiles total, "
          f"{len(tasks)} still to do, {args.workers} workers.\n", flush=True)

    t_start = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_task, t): t for t in tasks}
    n_done = 0
    try:
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            for fut in as_completed(futures):
                key, _, kind, knob, sign = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"FAILED {key} {kind} {knob} {sign:+d}: {e}", flush=True)
                    continue
                for r in rows:
                    writer.writerow(r)
                f.flush()
                n_done += 1
                wall = time.time() - t_start
                eta = wall / n_done * (len(tasks) - n_done)
                label = 'base+dB' if kind == 'base' else f"{knob} {sign:+d}"
                print(f"[{n_done}/{len(tasks)}] {key[0]}:{key[1]} {label:<16} loss={rows[0]['loss']:9.3f} "
                      f"task {rows[0]['task_elapsed_s']:5.1f} s | elapsed {wall/60:5.1f} min, "
                      f"ETA {eta/60:5.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\nStopping: cancelling queued tasks (finished points are saved; re-run to resume).", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    pool.shutdown()
    print(f"\nDone in {(time.time() - t_start)/60:.1f} min. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
