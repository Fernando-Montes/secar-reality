"""
COSY transfer maps from the Q7 Exit to the DL19 exit (= HEX3 entrance =
'H3 Entrance', element 35), for efl-d7 (2026-10-17). Range: DL18
(DL 0.496188), the WF1 block (WF1 misalignment TA/RA/SA, CB, WC ..., CB,
undo, FD, FR 3) and DL19 (DL 0.498516-0.1; DL 0.1). Total length 3.359704 m.

Lattice: pg_June2025_corrected.fox, nominal optics, order 4 (OV 4 3 2), as in
the deck. Works only on COPIES; the original deck and cached maps are not
modified.

Outputs in Results/efl_q7exit_to_dl19/:
  ecat_cached_segments/  copies of the three maps ECAT actually applies over
                         this range (MISALIGN = 1, the deck default):
                         fort.282 (Q7 Exit -> WF1 Entrance, DL18),
                         fort.283 (WF1 Entrance -> WF1 Exit, WF1 block),
                         fort.284 (WF1 Exit -> H3 Entrance, DL19)
  map_q7exit_to_dl19exit_misalign1.fort   single composite map, MISALIGN = 1
  map_q7exit_to_dl19exit_misalign0.fort   single composite map, MISALIGN = 0
  deck_misalign{1,0}.fox, cosy_stdout_misalign{1,0}.txt   provenance
The composite maps come from a copy of the deck with the two intermediate WMI
calls removed (lines 576 after DL18 and 607 after the WF1 block), so the WMI
after DL19 writes the map accumulated from the Q7 Exit (the WMI after Q7
resets it with UM). Same COSY WM format as ECAT's fort files, zeroth-order
terms included.

Checks (no extra COSY): header length L = 3.359704; the MISALIGN = 1
composite reproduces, ray by ray, applying ECAT's three cached segment maps
in sequence to the group-10 E2 rays at the Q7 Exit; the MISALIGN = 0 map has
no zeroth-order terms.

Two COSY runs (~20 s each, in private work directories).

Run yourself:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 make_q7exit_dl19_maps.py
"""
import os
import re
import shutil
import subprocess
import numpy as np

from config import COSY_directory, matrix_directory, pkl_directory, element_names_stripped
from helper_functions import read_file, split_fort, process_section, load_optics_data, generateInitialDistribution
from fast_tracer import compile_map, apply_map, rays_to_array, VARS
from fp2_metrics import build_initial_distribution

BASE = "pg_June2025_corrected"
OUT = "Results/efl_q7exit_to_dl19"
CACHED = os.path.join(matrix_directory, BASE)
SEG = {282: "Q7 Exit -> WF1 Entrance (DL18)", 283: "WF1 Entrance -> WF1 Exit (WF1 block)",
       284: "WF1 Exit -> H3 Entrance (DL19)"}
REMOVE_WMI = (576, 607)       # 1-based lines in the deck
MISALIGN_LINE = 1035
TOTAL_L = 0.496188 + 2.365 + 0.498516


def parse(path):
    with open(path) as f:
        secs = [process_section(s) for s in split_fort(read_file(f.read()))]
    assert len(secs) == 8, (path, len(secs))
    return secs


def header_length(path):
    with open(path) as f:
        return float(f.readline().split()[1].replace('D', 'E'))


def make_deck(misalign):
    lines = open(os.path.join(COSY_directory, f"{BASE}.fox")).readlines()
    for n in REMOVE_WMI:
        assert lines[n - 1].strip() == "WMI;", (n, lines[n - 1])
    assert lines[575 - 1].strip().startswith("DL 0.496188; {DL18}"), lines[574]
    assert lines[611 - 1].strip() == "WMI;" and "DL 0.1;" in lines[610 - 1], lines[609:611]
    assert re.match(r"MISALIGN:=1;", lines[MISALIGN_LINE - 1]), lines[MISALIGN_LINE - 1]
    for n in REMOVE_WMI:
        lines[n - 1] = "{WMI; removed: composite map from Q7 Exit to DL19 exit}\n"
    lines[MISALIGN_LINE - 1] = lines[MISALIGN_LINE - 1].replace("MISALIGN:=1;", f"MISALIGN:={misalign};")
    return lines


def run(misalign):
    work = os.path.join(OUT, f"work_misalign{misalign}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    name = f"q7dl19_m{misalign}"
    deck = make_deck(misalign)
    with open(os.path.join(work, f"{name}.fox"), "w") as f:
        f.writelines(deck)
    shutil.copy(os.path.join(COSY_directory, "cosy"), work)
    shutil.copy(os.path.join(COSY_directory, "cosy.bin"), work)
    p = subprocess.run(["./cosy", f"{name}.fox"], cwd=work, capture_output=True, text=True)
    with open(os.path.join(OUT, f"cosy_stdout_misalign{misalign}.txt"), "w") as f:
        f.write(p.stdout + p.stderr)
    src = os.path.join(work, "fort.282")
    if not os.path.exists(src):
        raise RuntimeError(f"COSY did not write fort.282 (MISALIGN={misalign}); see cosy_stdout_misalign{misalign}.txt")
    dst = os.path.join(OUT, f"map_q7exit_to_dl19exit_misalign{misalign}.fort")
    shutil.copy(src, dst)
    shutil.copy(os.path.join(work, f"{name}.fox"), os.path.join(OUT, f"deck_misalign{misalign}.fox"))
    shutil.rmtree(work)
    L = header_length(dst)
    assert abs(L - TOTAL_L) < 1e-9, f"length {L} != {TOTAL_L}: wrong segment"
    return dst


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "ecat_cached_segments"), exist_ok=True)
    for k, what in SEG.items():
        shutil.copy(os.path.join(CACHED, f"fort.{k}"), os.path.join(OUT, "ecat_cached_segments", f"fort.{k}"))
        print(f"copied ECAT cached fort.{k}: {what}, L = {header_length(os.path.join(CACHED, f'fort.{k}')):.6f} m")

    maps = {m: run(m) for m in (1, 0)}
    for m, p in maps.items():
        print(f"MISALIGN={m}: {p}  (L = {header_length(p):.6f} m)")

    # check 1: composite (MISALIGN=1) vs ECAT's three cached segments in sequence
    _, groups = load_optics_data(BASE, pkl_directory)
    q7 = element_names_stripped.index('Q7 Exit')
    X = rays_to_array(generateInitialDistribution(build_initial_distribution('10', 1000)))
    for g in groups[:q7]:
        X = apply_map(X, *compile_map(g))
    ok = np.all(np.abs(X[:, [0, 2]]) < 0.1, axis=1) & np.all(np.isfinite(X), axis=1)
    Xq = X[ok]
    seq = Xq
    for k in SEG:
        seq = apply_map(seq, *compile_map(parse(os.path.join(CACHED, f"fort.{k}"))))
    comp = apply_map(Xq, *compile_map(parse(maps[1])))
    d = np.abs(comp - seq)
    names = ['x', 'a', 'y', 'b', 'l', 'dE', 'dm', 'dz']
    print(f"\ncheck: composite MISALIGN=1 vs ECAT's three cached segments, {len(Xq)} group-10 rays at Q7 Exit:")
    print("  max |diff| per coordinate: " + ", ".join(f"{n} {v:.1e}" for n, v in zip(names, d.max(axis=0))))

    # check 2: zeroth-order terms
    for m, p in maps.items():
        z = [float(df.loc[(df[VARS].astype(float) == 0).all(axis=1), 'Coefficient'].astype(float).sum())
             for df in parse(p)]
        print(f"  zeroth-order terms, MISALIGN={m}: " + ", ".join(f"{n} {v:+.3e}" for n, v in zip(names, z)))
    print(f"\nall outputs in {OUT}/")
