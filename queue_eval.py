"""
Evaluator for secar-reality campaigns on DigitalTwin's eval queue
(eval_queue/SPEC.md v0.3, 2026-10-23). Pinned by sha1 in each campaign's
spec.json and imported by the worker from its frozen copy, after the worker
has repointed config's paths and deck name to its own (SPEC §2).

Entry points (SPEC §3.2):
  init(settings, run_dir)                  once per pool process
  deck_edits(deck_values) -> [[line, text], ...]
                                           the exact .fox edits compiled for
                                           these deck values (upstream included)
  evaluate_group(deck_values, [(cand_id, tracing_values), ...], settings)
                                           one compile, one trace per candidate
  quantities(planes) -> [column, ...]      result columns, in order

settings = the campaign's spec.json (keys used: evaluation, parameters,
upstream). Upstream entries are fixed for the campaign: "deck" ones are
compiled into every deck, "tracing" ones (dB) are applied in every trace. deck_values / tracing_values = {parameter name: value} for the
parameters with applied == "deck" / "tracing"; every campaign parameter must
be given on every call, zeros included (SPEC §3.1, f6).

Units: parameter values in the native tuneChanges units (B_SC and dB as
fractions, X/Y/length/rhoE/rhoB in m, Roll/Yaw/Pitch in deg), as offsets from
the nominal deck. Results: M12 in mm/mrad, M16 in mm/%, offsets and mean X in
mm with the ECAT readout sign (X = -x*1000); counts are rays.

Differences from fp3_eval (same physics and readout):
  - every deck, the unedited one included, is compiled from the deck text in
    a private mkdtemp folder under run_dir (no shared COSY10.2/Matrix_Files/
    pkl_Files names, no cached base map; SPEC f1, f2);
  - COSY runs with a timeout and an exit-status and map-count check (f4);
  - deck values are written with DECK_FMT/WFR_FMT digits (f3);
  - a group with no counted ray gives "no rays for group <g> at <P>" (f8).
The readout is fp3_eval.observables_from_maps, unchanged.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time

import config
import helper_functions
from helper_functions import prepareTuneChanges, prepareCOSYchanges, read_file, split_fort, process_section
import fp3_eval

DECK_FMT = ".12f"
WFR_FMT = ".12f"
FORT_PATTERN = re.compile(r'^fort\.(25[0-9]|2[6-9][0-9]|30[0-9]|31[0-9]|32[0-9])$')   # as process_fort_files
NO_RAYS = re.compile(r"No rays survived the FP1-slit\+transmission cut for group (\S+)")

_state = {}


def quantities(planes):
    cols = []
    for p in planes:
        cols += [f"{p}_M12", f"{p}_Offset12", f"{p}_M16", f"{p}_Offset16"]
        for g in fp3_eval.PLANES[p]['groups'].ALL_GROUPS:
            cols += [f"{p}_meanx_g{g}", f"{p}_n_g{g}"]
    return cols


def _tune_changes(entries, values):
    return [[e['tune_change'][0], e['tune_change'][1], float(values[e['name']])] for e in entries]


def init(settings, run_dir):
    ev = settings['evaluation']
    if ev['deck_cut'] != 'none':
        raise ValueError(f"deck_cut {ev['deck_cut']!r} is not implemented")
    for p in ev['planes']:
        if p not in fp3_eval.PLANES:
            raise ValueError(f"unknown plane {p!r}")
    helper_functions.DECK_FMT = DECK_FMT
    helper_functions.WFR_FMT = WFR_FMT
    with open(os.path.join(config.COSY_directory, f"{config.fox_file}.fox")) as f:
        deck_lines = f.readlines()
    params = settings['parameters']
    upstream = settings.get('upstream', [])
    os.makedirs(run_dir, exist_ok=True)
    _state.clear()
    _state.update(
        run_dir=run_dir,
        deck_lines=deck_lines,
        deck_params=[e for e in params if e['applied'] == 'deck'],
        tracing_params=[e for e in params if e['applied'] == 'tracing'],
        upstream_tc=_tune_changes([e for e in upstream if e['applied'] == 'deck'],
                                  {e['name']: e['value'] for e in upstream}),
        upstream_tracing_tc=_tune_changes([e for e in upstream if e['applied'] == 'tracing'],
                                          {e['name']: e['value'] for e in upstream}),
        planes=list(ev['planes']),
        nrays=int(ev['nrays_per_group']),
        timeout=float(ev['cosy_timeout_s']),
        n_maps=len(config.element_names_stripped) - 1,
    )
    names = [e['name'] for e in params]
    if len(set(names)) != len(names) or set(names) & {e['name'] for e in upstream}:
        raise ValueError("parameter names must be unique and not repeat an upstream name")
    # two deck parameters that rewrite the same .fox line would overwrite each other
    # (e.g. WF1 length and rhoE/rhoB both rewrite line 593): refuse the campaign
    probe = deck_edits({e['name']: e['step'] for e in _state['deck_params']})
    lines = [n for n, _ in probe]
    if len(set(lines)) != len(lines):
        dup = sorted({n for n in lines if lines.count(n) > 1})
        raise ValueError(f"deck parameters edit the same .fox line(s) {dup}")


def deck_edits(deck_values):
    tc = _state['upstream_tc'] + _tune_changes(_state['deck_params'], deck_values)
    params, values, dwf = prepareTuneChanges(tc)
    if dwf:
        raise ValueError("a deck parameter or deck upstream entry is applied in tracing (dB)")
    return prepareCOSYchanges(params, values)[0] if params else []


def _compile(deck_values):
    """-> (maps or None, status)."""
    lines = list(_state['deck_lines'])
    for n, text in deck_edits(deck_values):
        lines[n - 1] = text
    d = tempfile.mkdtemp(prefix="c_", dir=_state['run_dir'])
    try:
        with open(os.path.join(d, "deck.fox"), 'w') as f:
            f.writelines(lines)
            f.write('\n')
        for exe in ("cosy", "cosy.bin"):
            os.symlink(os.path.join(config.COSY_directory, exe), os.path.join(d, exe))
        try:
            r = subprocess.run(["./cosy", "deck.fox"], cwd=d, timeout=_state['timeout'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return None, "cosy_timeout"
        forts = sorted((f for f in os.listdir(d) if FORT_PATTERN.match(f)), key=lambda x: int(x.split('.')[1]))
        frames = []
        for name in forts:
            with open(os.path.join(d, name)) as f:
                frames.extend(process_section(s) for s in split_fort(read_file(f.read())))
        maps = [frames[i:i + 8] for i in range(0, len(frames), 8)]
        if r.returncode != 0 or len(maps) != _state['n_maps'] or len(frames) % 8:
            return None, f"cosy_failed:{len(maps)}maps"
        return maps, "ok"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _trace(maps, tracing_values):
    dwf = prepareTuneChanges(_state['upstream_tracing_tc'] +
                             _tune_changes(_state['tracing_params'], tracing_values))[2]
    rec = {}
    for p in _state['planes']:
        try:
            obs, mean_x, n = fp3_eval.observables_from_maps(maps, dwf or None, p, _state['nrays'])
        except RuntimeError as e:
            m = NO_RAYS.search(str(e))
            if m:
                return {"status": f"error: no rays for group {m.group(1)} at {p}"}
            raise
        rec.update({f"{p}_M12": obs[0], f"{p}_Offset12": obs[1], f"{p}_M16": obs[2], f"{p}_Offset16": obs[3]})
        for g in fp3_eval.PLANES[p]['groups'].ALL_GROUPS:
            rec[f"{p}_meanx_g{g}"] = mean_x[g]
            rec[f"{p}_n_g{g}"] = n[g]
    rec["status"] = "ok"
    return rec


def evaluate_group(deck_values, candidates, settings):
    """candidates = [(cand_id, tracing_values), ...] sharing deck_values.
    Returns one record per candidate, in order: {cand_id, status, eval_s, <quantities>}.
    eval_s = the shared compile time divided among the candidates + the candidate's own trace time."""
    t0 = time.time()
    maps, status = _compile(deck_values)
    share = (time.time() - t0) / len(candidates)
    out = []
    for cand_id, tracing_values in candidates:
        t1 = time.time()
        rec = {"status": status} if maps is None else _trace(maps, tracing_values)
        rec["cand_id"] = cand_id
        rec["eval_s"] = share + time.time() - t1
        out.append(rec)
    return out
