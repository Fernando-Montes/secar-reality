"""
FP3 knob range check (2026-10-04), complementing fp3_knob_ranking.py.

The local-derivative ranking uses small steps. Redoing the FP2 ranking that
way (fp2_ranking_validation.py) showed some knobs -- rolls especially --
respond very differently over a realistic range than at the operating point,
and at FP3 the Y/pitch/roll knobs are dominated by a quadratic term. So here
each knob is also evaluated over a realistic range, to catch knobs that are
flat at the operating point but matter at realistic amplitudes.

Same knobs, loss, beam and candidates as fp3_knob_ranking.py. Each knob is
set to -R, -R/2, +R/2, +R (everything else at the candidate's FP2 solution);
the base point is re-evaluated in the same run. Ranges (edit RANGES below):
  B_SC +-10%, dB +-1%, X/Y +-3 mm, Roll +-0.5 deg  (the FP2 OAT screen's ranges)
  Yaw/Pitch +-0.1 deg  (not in the FP2 screen; measured yaws/pitches in the
                        .fox are all < 0.05 deg)
Because the result is a loss change over a realistic range, all knobs can be
compared on one scale, unlike derivatives in %, mm and deg.

A point where some group keeps no rays is recorded with status 'no_rays'
(loss empty) instead of stopping the run.

Default candidates: v1:20 (already in the derivative ranking), the last v1
and the last v7 in the fixed candidate order. Override with
--candidates v1:20,v7:5,...

Checkpointed/resumable (keyed by source, iteration, knob, frac), parallel
over processes, compile files deleted after use.

Run yourself:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 fp3_knob_range_check.py
"""
import os
import csv
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from fp3_knob_ranking import KNOBS, DB_KNOBS, LATTICE_KNOBS, load_candidates

RESULTS_FILE = "Results/fp3_knob_range_check_results.csv"
FIELDNAMES = ["source", "iteration", "knob", "frac", "value", "unit", "status",
              "M12", "Offset12", "M16", "Offset16", "loss", "task_elapsed_s"]
FRACS = (-1.0, -0.5, 0.5, 1.0)


def range_of(knob):
    """Half-range R in native tuneChanges units."""
    if knob.endswith('_B_SC'):
        return 0.10
    if knob.endswith('_Exit_dB'):
        return 0.01
    if knob.endswith('_X') or knob.endswith('_Y'):
        return 0.003
    if knob.endswith('_Roll'):
        return 0.5
    if knob.endswith('_Yaw') or knob.endswith('_Pitch'):
        return 0.1
    raise KeyError(knob)


RANGES = {k: range_of(k) for k in KNOBS}


def _eval(fp3_eval, lattice_tc, variants):
    """evaluate_variants, but one trace failure (a group with no rays) doesn't sink the others."""
    groups = fp3_eval.compile_maps(lattice_tc)
    out = []
    for extra in variants:
        dwf = fp3_eval.prepareTuneChanges(lattice_tc + extra)[2]
        try:
            obs, _, _ = fp3_eval.observables_from_maps(groups, dwf or None)
            out.append(('ok', obs, fp3_eval.fp3_loss.loss(obs)))
        except RuntimeError:
            out.append(('no_rays', None, None))
    return out


def run_task(task):
    import fp3_eval
    cand_key, fp2_tc, kind, knob, frac = task
    t0 = time.time()
    if kind == 'base':
        pts = [('base', 0.0)] + [(k, f) for k in DB_KNOBS for f in FRACS]
        variants = [[] if k == 'base' else [KNOBS[k][0](f * RANGES[k])] for k, f in pts]
        try:
            res = _eval(fp3_eval, fp2_tc, variants)
        except RuntimeError:
            res = [('no_rays', None, None)] * len(pts)
    else:
        pts = [(knob, frac)]
        try:
            res = _eval(fp3_eval, fp2_tc + [KNOBS[knob][0](frac * RANGES[knob])], [[]])
        except RuntimeError:
            res = [('no_rays', None, None)]
    el = round(time.time() - t0, 1)
    rows = []
    for (k, f), (status, obs, loss) in zip(pts, res):
        row = {"source": cand_key[0], "iteration": cand_key[1], "knob": k, "frac": f,
               "value": 0.0 if k == 'base' else f * RANGES[k], "unit": '' if k == 'base' else KNOBS[k][2],
               "status": status, "task_elapsed_s": el}
        row.update(dict(zip(["M12", "Offset12", "M16", "Offset16"], obs)) if obs is not None else
                   {"M12": "", "Offset12": "", "M16": "", "Offset16": ""})
        row["loss"] = loss if loss is not None else ""
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=None, help="comma list like v1:20,v7:5 (default: 3 spread out)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    all_c = load_candidates()
    if args.candidates:
        want = [(s, int(i)) for s, i in (x.split(':') for x in args.candidates.split(','))]
        by_key = dict(all_c)
        missing = [w for w in want if w not in by_key]
        if missing:
            raise SystemExit(f"not among the 46 candidates: {missing}")
        cands = [(w, by_key[w]) for w in want]
    else:
        cands = [all_c[0], all_c[23], all_c[45]]

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    done = {(r["source"], int(r["iteration"]), r["knob"], float(r["frac"]))
            for r in csv.DictReader(open(RESULTS_FILE))}

    tasks = []
    for key, tc in cands:
        base_pts = [('base', 0.0)] + [(k, f) for k in DB_KNOBS for f in FRACS]
        if any((key[0], key[1], k, f) not in done for k, f in base_pts):
            tasks.append((key, tc, 'base', None, 0.0))
        for k in LATTICE_KNOBS:
            for f in FRACS:
                if (key[0], key[1], k, f) not in done:
                    tasks.append((key, tc, 'knob', k, f))
    print(f"candidates: {[f'{k[0]}:{k[1]}' for k, _ in cands]}")
    print(f"{len(KNOBS)} knobs x {len(FRACS)} points: {len(cands) * (1 + len(LATTICE_KNOBS) * len(FRACS))} "
          f"compiles total, {len(tasks)} tasks to do, {args.workers} workers.\n", flush=True)

    t_start = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_task, t): t for t in tasks}
    n = 0
    try:
        with open(RESULTS_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            for fut in as_completed(futures):
                key, _, kind, knob, frac = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"FAILED {key} {kind} {knob} {frac}: {e}", flush=True)
                    continue
                w.writerows(rows)
                f.flush()
                n += 1
                wall = time.time() - t_start
                label = 'base+dB' if kind == 'base' else f"{knob} {frac:+.1f}R"
                r0 = rows[0]
                loss_txt = f"{r0['loss']:9.2f}" if r0['status'] == 'ok' else '  no rays'
                print(f"[{n}/{len(tasks)}] {key[0]}:{key[1]} {label:<18} loss={loss_txt} "
                      f"task {r0['task_elapsed_s']:5.1f} s | elapsed {wall/60:5.1f} min, "
                      f"ETA {wall/n*(len(tasks)-n)/60:5.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\nStopping (finished points saved; re-run to resume).", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)
    pool.shutdown()
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
