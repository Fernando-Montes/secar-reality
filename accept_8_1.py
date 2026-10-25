"""
Acceptance test 1 of eval_queue/SPEC.md v0.3 (§8.1, engine agreement), 2026-10-23.

Three FP2 1-sigma solutions (fp3_knob_ranking.load_candidates): v1:20, v1:143
and v7:10 (v7 also carries B1-B4 Exit dB). One campaign per solution
(sr_acc81_<src>_<iter>), with the solution as `upstream` and five parameters,
one per kind:
  Q9_B_SC (deck, B_SC), B5_Exit_dB (tracing), Q9_X (deck, misalign block),
  WF2_Yaw (deck, WF2's commented-out block), Q9_Roll (deck, misalign block).
Points per campaign (one request): base (all five at 0) and each parameter at
+-step (fp3_knob_ranking steps), 11 lines, 5 compiles (dB shares the base).
Planes FP2 and FP3, E2 groups, 3000 rays/group, seed 42; lattice
pg_June2025_corrected (MISALIGN 1, WFs on).

Upstream B_SC values are rounded as the FP2 family's decks were written
(.6f of nominal + value), so the solutions are the ones that were validated;
upstream dB values are exact (applied in tracing).

Steps:
  local    reference values with queue_eval in this process tree
           (15 COSY compiles, 10 processes, ~1 min): Results/acc81_reference.csv.
           Run by Fernando.
  submit   write spec.json + req_000000.csv of the three campaigns into
           eval_queue/outbox/ for DigitalTwin's worker. Fernando starts the worker.
  compare  worker results vs the local reference: per-group ray counts identical,
           observables and per-group mean X within 1e-12 mm (SPEC §8.1); then,
           for information, FP3 observables vs Results/fp3_knob_ranking_results.csv
           (2026-10-04, fp3_eval path; decks there carried only the one varied
           parameter, here all five at zero or +-step). Writes DONE in each campaign
           folder once its results are complete. Output Results/acc81_compare.txt.

Run (from secar-reality/, SECAR python):
    python3 accept_8_1.py local | submit | compare
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import config
from fp3_knob_ranking import load_candidates
import jac_loop

CANDS = [("v1", 20), ("v1", 143), ("v7", 10)]
PARAMS = [
    {"name": "Q9_B_SC", "tune_change": ["Q9", "B_SC"], "unit": "fraction (0.01 = +1%)", "applied": "deck",
     "step": 0.01, "trust": [-0.15, 0.15]},
    {"name": "B5_Exit_dB", "tune_change": ["B5 Exit", "dB"], "unit": "fraction (0.0025 = +0.25%)",
     "applied": "tracing", "step": 0.0025, "trust": [-0.01, 0.01]},
    {"name": "Q9_X", "tune_change": ["Q9", "X"], "unit": "m", "applied": "deck", "step": 0.00025,
     "trust": [-0.003, 0.003]},
    {"name": "WF2_Yaw", "tune_change": ["WF2", "Yaw"], "unit": "deg", "applied": "deck", "step": 0.05,
     "trust": [-0.1, 0.1]},
    {"name": "Q9_Roll", "tune_change": ["Q9", "Roll"], "unit": "deg", "applied": "deck", "step": 0.1,
     "trust": [-0.5, 0.5]},
]
NAMES = [p["name"] for p in PARAMS]
REF = "Results/acc81_reference.csv"
OUT = "Results/acc81_compare.txt"
RANKING = "Results/fp3_knob_ranking_results.csv"
RUN_DIR = "Results/acc81_local_run"


def campaign(src, it):
    return f"sr_acc81_{src}_{it}"


def upstream_of(tc):
    out = []
    for elem, kind, v in tc:
        if kind == "dB":
            out.append({"name": f"{elem.replace(' Exit', '')}_Exit_dB", "tune_change": [elem, "dB"],
                        "unit": "fraction", "applied": "tracing", "value": v})
        else:
            nom = config.cosy_lines[elem]["B_SC"]["nom"]
            out.append({"name": f"{elem}_B_SC", "tune_change": [elem, "B_SC"], "unit": "fraction",
                        "applied": "deck", "value": float(f"{nom + v:.6f}") - nom,
                        "note": "rounded as written in the FP2 family's decks (.6f)"})
    return out


def configs():
    cands = dict(load_candidates())
    out = []
    for src, it in CANDS:
        out.append({
            "campaign": campaign(src, it),
            "lattice": {"fox": "pg_June2025_corrected", "fox_sha1": "fca3af5780afba3e60dbbdea52f891cde18b45ca",
                        "misalign": 1, "wien_filters": "on"},
            "upstream": upstream_of(cands[(src, it)]),
            "evaluation": {"planes": ["FP2", "FP3"], "nrays_per_group": 3000, "cosy_timeout_s": 300,
                           "deck_cut": "none"},
            "parameters": PARAMS,
            "plan": {"kind": "SPEC 8.1 engine agreement, fixed points, no loop",
                     "fp2_solution": f"{src}:{it}"},
        })
    return out


def points():
    pts = [("r0s000", dict.fromkeys(NAMES, 0.0))]
    for k, p in enumerate(PARAMS):
        for sgn, j in ((+1, 1), (-1, 2)):
            v = dict.fromkeys(NAMES, 0.0)
            v[p["name"]] = sgn * p["step"]
            pts.append((f"r0s{2 * k + j:03d}", v))
    return pts


def point_label(cid):
    n = int(cid[3:])
    return ("base", 0) if n == 0 else (NAMES[(n - 1) // 2], +1 if n % 2 else -1)


def _local_task(task):
    import queue_eval
    spec, deck_values, cands = task
    queue_eval.init(spec, os.path.join(RUN_DIR, spec["campaign"]))
    return spec["campaign"], queue_eval.evaluate_group(deck_values, cands, spec)


def local():
    tasks = []
    for cfg in configs():
        spec = jac_loop.build_spec(cfg)
        groups = {}
        for cid, v in points():
            deck = tuple(v[p["name"]] for p in PARAMS if p["applied"] == "deck")
            groups.setdefault(deck, []).append((cid, {p["name"]: v[p["name"]] for p in PARAMS
                                                      if p["applied"] == "tracing"}))
        deck_names = [p["name"] for p in PARAMS if p["applied"] == "deck"]
        tasks += [(spec, dict(zip(deck_names, d)), c) for d, c in groups.items()]
    print(f"{len(tasks)} compiles, 10 processes", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(10) as ex:
        for camp, recs in ex.map(_local_task, tasks):
            rows += [dict(r, campaign=camp) for r in recs]
            print(f"  {camp}: {[r['cand_id'] for r in recs]} {[r['status'] for r in recs]}", flush=True)
    cols = ["campaign", "cand_id", "status", "eval_s"] + jac_loop.build_spec(configs()[0])["quantities"]
    with open(REF, "w") as f:
        f.write(f"# SPEC 8.1 local reference, queue_eval sha1 {jac_loop.sha1_file('queue_eval.py')}, "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}, lattice pg_June2025_corrected, units as SPEC 3.4\n")
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["campaign"], r["cand_id"])):
            w.writerow({c: (repr(r[c]) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})
    print(f"wrote {REF} ({len(rows)} rows) in {time.time() - t0:.0f} s")


def submit():
    for cfg in configs():
        q = jac_loop.Queue(jac_loop.build_spec(cfg))
        os.makedirs(q.dir, exist_ok=True)
        if os.path.exists(os.path.join(q.dir, "spec.json")):
            sys.exit(f"{q.dir} already has a spec.json: campaign names are never reused")
        jac_loop.write_atomic(os.path.join(q.dir, "spec.json"), json.dumps(q.spec, indent=1) + "\n")
        lines = points()
        body = "cand_id," + ",".join(NAMES) + "\n" + "".join(
            f"{c}," + ",".join(repr(float(v[n])) for n in NAMES) + "\n" for c, v in lines)
        head = (f"# protocol_version: {jac_loop.PROTOCOL}\n# spec_id: {q.spec['spec_id']}\n"
                f"# campaign: {q.spec['campaign']}\n# request: 000000\n"
                f"# created: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n# n_candidates: {len(lines)}\n"
                f"# units: native tuneChanges units, offsets from nominal (SPEC 3.1)\n")
        jac_loop.write_atomic(os.path.join(q.dir, "req_000000.csv"), head + body)
        print(f"submitted {q.spec['campaign']} spec {q.spec['spec_id']}: {len(lines)} points")


def compare():
    ref = {}
    for r in csv.DictReader(l for l in open(REF) if not l.startswith("#")):
        ref[(r["campaign"], r["cand_id"])] = r
    quantities = jac_loop.build_spec(configs()[0])["quantities"]
    counts = [q for q in quantities if "_n_g" in q]
    values = [q for q in quantities if "_n_g" not in q]
    rank = {}
    for r in csv.DictReader(open(RANKING)):
        rank[(r["source"], int(r["iteration"]), r["knob"], int(r["sign"]))] = r
    lines, ok_all = [], True

    def log(s):
        print(s)
        lines.append(s)

    log(f"SPEC 8.1 compare {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for src, it in CANDS:
        camp = campaign(src, it)
        res_path = os.path.join(jac_loop.WORKER_OUTBOX, camp, "res_000000.csv")
        got = jac_loop.read_results(res_path) if os.path.exists(res_path) else {}
        if len(got) < len(points()):
            log(f"{camp}: {len(got)}/{len(points())} results so far; not complete")
            ok_all = False
            continue
        for cid, _ in points():
            g, r = got[cid], ref[(camp, cid)]
            lab = point_label(cid)
            if g["status"] != "ok" or r["status"] != "ok":
                log(f"{camp} {cid} {lab}: FAIL status worker {g['status']} / local {r['status']}")
                ok_all = False
                continue
            bad_n = [c for c in counts if int(float(g[c])) != int(float(r[c]))]
            d = max(abs(float(g[c]) - float(r[c])) for c in values)
            ok = not bad_n and d <= 1e-12
            ok_all &= ok
            k = (src, it, lab[0], lab[1])
            x = ""
            if k in rank:
                dr = max(abs(float(g[f"FP3_{o}"]) - float(rank[k][o])) for o in jac_loop.OBS)
                x = f"  [info] vs ranking file max|diff| {dr:.2g}"
            log(f"{camp} {cid} {lab[0]}{'' if lab[1] == 0 else ('+' if lab[1] > 0 else '-')}: "
                f"{'PASS' if ok else 'FAIL'} max|diff| {d:.2g} count mismatches {bad_n}{x}")
        jac_loop.write_atomic(os.path.join(jac_loop.OUTBOX, camp, "DONE"),
                              f"{time.strftime('%Y-%m-%dT%H:%M:%S')} SPEC 8.1 results read\n")
    log(f"OVERALL: {'PASS' if ok_all else 'FAIL or incomplete'}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"local": local, "submit": submit, "compare": compare}[sys.argv[1]]()
