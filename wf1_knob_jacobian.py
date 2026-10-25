"""
Derivatives of the E2 observables at FP2 (VD1638) and FP3 (VD1783) with
respect to ECAT's existing WF1 knobs, at nominal (2026-10-19). Used to test
whether efl-d7's WF1 EFL scenarios are degenerate with these knobs
(analyse_wf1_efl_scenarios.py).

Lattice: pg_June2025_corrected, nominal, MISALIGN = 1. Beam: E2 groups, 3000
rays/group, seed 42 (fp3_eval). Observables: [M12, Offset12, M16, Offset16]
at each plane, with the FP2 or FP3 group sets.

Knobs (central differences at +-step):
  WF1_Exit_dB  +-0.1%      dB kick at the WF1 exit, applied in tracing (no compile)
  WF1_length   +-2.222 mm  WC length with compensating DL18/DL19 (lines 575/593/609)
  WF1_rhoE     +-7 mm      electric radius in the WC statement (7.000 m nominal; 0.1%)
  WF1_rhoB     +-7 mm      magnetic radius in the WC statement (0.1%)
  WF1_X        +-0.333 mm  horizontal misalignment of WF1 (SA lines 589/597)
8 COSY compiles (one per lattice knob and sign), parallel; compile files deleted
after use (fp3_eval). Checkpointed: finished points are appended to
Results/wf1_knob_jacobian_results.csv and skipped on restart.

Run yourself:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 wf1_knob_jacobian.py
"""
import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RESULTS_FILE = "Results/wf1_knob_jacobian_results.csv"
FIELDS = ["knob", "value", "plane", "M12", "Offset12", "M16", "Offset16", "elapsed_s"]
NRAYS = 3000
KNOBS = {   # knob -> (tuneChange builder, step in native units)
    'WF1_Exit_dB': (lambda v: ['WF1 Exit', 'dB', v], 0.001),
    'WF1_length': (lambda v: ['WF1', 'length', v], 0.002222),
    'WF1_rhoE': (lambda v: ['WF1', 'rhoE', v], 0.007),
    'WF1_rhoB': (lambda v: ['WF1', 'rhoB', v], 0.007),
    'WF1_X': (lambda v: ['WF1', 'X', v], 0.000333),
}


def run(task):
    import fp3_eval
    kind, knob, sign = task
    t0 = time.time()
    if kind == 'base':                                   # nominal + the dB points: one shared map set
        groups = fp3_eval.compile_maps([])
        b, step = KNOBS['WF1_Exit_dB']
        pts = [('NOMINAL', 0.0, [])] + [('WF1_Exit_dB', s * step, [b(s * step)]) for s in (1, -1)]
    else:
        b, step = KNOBS[knob]
        groups = fp3_eval.compile_maps([b(sign * step)])
        pts = [(knob, sign * step, [])]
    out = []
    for k, v, extra in pts:
        dwf = fp3_eval.prepareTuneChanges(extra)[2] if extra else None
        for plane in ('FP2', 'FP3'):
            obs, _, _ = fp3_eval.observables_from_maps(groups, dwf or None, plane, NRAYS)
            out.append({"knob": k, "value": v, "plane": plane, "M12": obs[0], "Offset12": obs[1],
                        "M16": obs[2], "Offset16": obs[3], "elapsed_s": round(time.time() - t0, 1)})
    return out


if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    done = {(r["knob"], round(float(r["value"]), 9)) for r in csv.DictReader(open(RESULTS_FILE))}
    tasks = []
    if ('NOMINAL', 0.0) not in done or ('WF1_Exit_dB', 0.001) not in done:
        tasks.append(('base', None, 0))
    for k, (_, step) in KNOBS.items():
        if k == 'WF1_Exit_dB':
            continue
        for s in (1, -1):
            if (k, round(s * step, 9)) not in done:
                tasks.append(('knob', k, s))
    print(f"{len(tasks)} tasks ({sum(t[0] == 'knob' for t in tasks)} compiles)", flush=True)
    with ProcessPoolExecutor(max_workers=min(9, max(1, len(tasks)))) as pool, \
            open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        futs = {pool.submit(run, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:
                print("FAILED", futs[fut], e, flush=True)
                continue
            w.writerows(rows)
            f.flush()
            print(f"done {futs[fut]}  ({rows[0]['elapsed_s']} s)", flush=True)
    print(f"wrote {RESULTS_FILE}")
