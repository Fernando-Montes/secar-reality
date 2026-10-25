"""
Compile test of queue_eval.py (eval_queue/SPEC.md v0.3), run by Fernando
(2026-10-23). Checks that queue_eval's private compiles give the same
observables as the existing path, before DigitalTwin's acceptance tests.

Lattice: config.fox_file (pg_June2025_corrected), MISALIGN as in the deck.
Beam: E2 groups, 3000 rays/group, seed 42. Planes FP2 and FP3.
7 COSY runs, sequential (~2-3 min). Both parameters are passed on every
deck, zeros included (SPEC f6), on both paths, so they compile the same text:
  1. base (zeros): queue_eval private compile vs fp3_eval.compile_maps
     (1b, info only: vs the cached unedited pkl_Files/<fox_file>.pkl)
  2. Q9_B_SC +1%: queue_eval                  vs fp3_eval.compile_maps
  3. WF2_Yaw +0.05 deg: queue_eval            vs fp3_eval.compile_maps
  4. timeout: cosy_timeout_s = 1              -> expect status cosy_timeout, no COSY left running
Required: identical ray counts per group, observables and per-group mean X
equal to 1e-12 mm, and no c_* folder left in the run folder.

Run:
    cd /Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 test_queue_eval.py
Output: printed, and Results/test_queue_eval.txt.
"""
import os
import subprocess
import time

import config
import fp3_eval
import queue_eval
from helper_functions import load_optics_data

RUN_DIR = os.path.join(config.results_directory, "test_queue_eval_run")
OUT = os.path.join(config.results_directory, "test_queue_eval.txt")
PLANES = ["FP2", "FP3"]
PARAMS = [
    {"name": "Q9_B_SC", "tune_change": ["Q9", "B_SC"], "applied": "deck", "step": 0.01},
    {"name": "WF2_Yaw", "tune_change": ["WF2", "Yaw"], "applied": "deck", "step": 0.05},
]


def settings(timeout=300):
    return {"evaluation": {"planes": PLANES, "nrays_per_group": 3000, "cosy_timeout_s": timeout,
                           "deck_cut": "none"},
            "parameters": PARAMS, "upstream": []}


def reference(maps):
    rec = {}
    for p in PLANES:
        obs, mean_x, n = fp3_eval.observables_from_maps(maps, None, p, 3000)
        rec.update({f"{p}_M12": obs[0], f"{p}_Offset12": obs[1], f"{p}_M16": obs[2], f"{p}_Offset16": obs[3]})
        for g in fp3_eval.PLANES[p]['groups'].ALL_GROUPS:
            rec[f"{p}_meanx_g{g}"] = mean_x[g]
            rec[f"{p}_n_g{g}"] = n[g]
    return rec


def compare(label, got, ref, log):
    if got["status"] != "ok":
        log(f"{label}: FAIL status {got['status']}")
        return False
    cols = queue_eval.quantities(PLANES)
    counts = [c for c in cols if "_n_g" in c]
    values = [c for c in cols if "_n_g" not in c]
    bad_n = [c for c in counts if got[c] != ref[c]]
    dmax = max(abs(got[c] - ref[c]) for c in values)
    ok = not bad_n and dmax <= 1e-12
    log(f"{label}: {'PASS' if ok else 'FAIL'}  max|diff| {dmax:.3g}  count mismatches {bad_n}  "
        f"FP3 M12 {got['FP3_M12']:.6f}  eval_s {got['eval_s']:.1f}")
    return ok


def main():
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log(f"test_queue_eval {time.strftime('%Y-%m-%d %H:%M:%S')}  lattice {config.fox_file}  run_dir {RUN_DIR}")
    queue_eval.init(settings(), RUN_DIR)
    zero = {"Q9_B_SC": 0.0, "WF2_Yaw": 0.0}
    results = []

    # Reference decks carry the same parameters, zeros included, so both paths compile the same deck text.
    # 1. base (all parameters at zero) vs fp3_eval; and, for information only, vs the cached unedited
    #    base map (zero-valued edits, e.g. the WF2 block activated with zeros, are physically the identity)
    got = queue_eval.evaluate_group(zero, [("base", {})], settings())[0]
    ref = reference(fp3_eval.compile_maps([["Q9", "B_SC", 0.0], ["WF2", "Yaw", 0.0]]))
    results.append(compare("1 base (zeros) vs fp3_eval", got, ref, log))
    compare("1b base (zeros) vs cached unedited pkl [info only]", got,
            reference(load_optics_data(config.fox_file, config.pkl_directory)[1]), log)

    # 2-3. one parameter at +step vs fp3_eval's existing compile path
    for i, (name, value) in enumerate((("Q9_B_SC", 0.01), ("WF2_Yaw", 0.05)), start=2):
        vals = dict(zero, **{name: value})
        got = queue_eval.evaluate_group(vals, [(name, {})], settings())[0]
        ref = reference(fp3_eval.compile_maps([["Q9", "B_SC", vals["Q9_B_SC"]], ["WF2", "Yaw", vals["WF2_Yaw"]]]))
        results.append(compare(f"{i} {name} +step vs fp3_eval", got, ref, log))

    # 4. timeout
    queue_eval.init(settings(timeout=1), RUN_DIR)
    got = queue_eval.evaluate_group(zero, [("timeout", {})], settings(timeout=1))[0]
    time.sleep(2)
    left = subprocess.run(["pgrep", "-f", "cosy deck.fox"], capture_output=True, text=True).stdout.split()
    ok = got["status"] == "cosy_timeout" and not left
    log(f"4 timeout: {'PASS' if ok else 'FAIL'}  status {got['status']}  COSY processes left {left}")
    results.append(ok)

    leftovers = [d for d in os.listdir(RUN_DIR) if d.startswith("c_")]
    log(f"compile folders left in run_dir: {leftovers}  -> {'PASS' if not leftovers else 'FAIL'}")
    results.append(not leftovers)
    log(f"OVERALL: {'PASS' if all(results) else 'FAIL'}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
