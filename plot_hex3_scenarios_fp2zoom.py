"""
Zoom of plot_hex3_scenarios.py around FP2 (2026-10-19): same 10 rays (E2 group
10), black = ECAT throughout, blue = efl-d7's WF1 model for Q7 Exit -> HEX3
entrance and ECAT elsewhere. Per scenario:
  left:  x vs z for z in [ZMIN, ZMAX] (WF1 exit -> past FP2), chamber walls in grey,
         FP2 Slits and VD1638 (the E2 FP2 readout) marked;
  right: dx = x(blue) - x(black) vs z over the same range, one line per ray
         (rays that pass the group-10 FP1 slit window, i.e. the ones that count
         in the E2 means, solid; the others dotted).
Rays are drawn up to their first aperture loss.

Run:
    cd AgentSearch
    python3 plot_hex3_scenarios_fp2zoom.py <efl csv of the 10-ray scenarios>
"""
import sys
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import contains_xy

from config import element_names_stripped as NAMES, z_coordinates as Z, beamline_geometry, pkl_directory, fox_file
from helper_functions import load_optics_data, find_polygons_at_z
from fast_tracer import compile_map, apply_map
from beamline_walls import draw_chamber_walls, zoom_to_rays

RAY_FILE = "Results/rays_q7exit_group10_10000.csv"
IDS_FILE = "Results/hex3_scenario_ray_ids.txt"
OUT = "Results/hex3_scenarios_fp2zoom.png"
ZMIN, ZMAX = 14.0, 20.5
Q7, H3 = NAMES.index('Q7 Exit'), NAMES.index('H3 Entrance')
LOC = {'WF1_entrance': NAMES.index('WF1 Entrance'), 'WF1_exit': NAMES.index('WF1 Exit'), 'H3_entrance': H3}
VD1638, FP2S = NAMES.index('VD1638'), NAMES.index('FP2 Slits')
END = max(i for i, z in enumerate(Z) if z <= ZMAX + 1.0)

ids = [int(v) for v in open(IDS_FILE).read().split()]
rows = {int(r['ray_id']): r for r in csv.DictReader(l for l in open(RAY_FILE) if not l.startswith('#'))}
X0 = np.array([[float(rows[i]['x0_m']), float(rows[i]['a0_rad']), float(rows[i]['y0_m']), float(rows[i]['b0_rad']),
                0.0, float(rows[i]['dE0']), 0.0, 0.0] for i in ids])
fp1 = np.array([rows[i]['passes_fp1_slits'] == '1' for i in ids])
_, groups = load_optics_data(fox_file, pkl_directory)
maps = [compile_map(g) for g in groups[:END]]
with open(beamline_geometry, 'rb') as f:
    _, polys = find_polygons_at_z(pickle.load(f), Z)


def prop(X, start):
    S = [X]
    for k in range(start, END):
        S.append(apply_map(S[-1], *maps[k]))
    return np.stack(S, axis=1)


def n_inside(locs, states):
    for j, (loc, s) in enumerate(zip(locs, states)):
        if polys[loc] is not None and not contains_xy(polys[loc], s[0], s[2]):
            return j
    return len(locs)


REG = prop(X0, 0)
efl = {}
for r in csv.DictReader(open(sys.argv[1])):
    rid = int(r['ray_id'])
    if rid not in ids:
        continue
    i = ids.index(rid)
    loc = LOC[r['location'].strip()]
    s = REG[i, loc].copy()
    s[[0, 1, 2, 3, 5]] = [float(r[c]) for c in ('x_m', 'a_rad', 'y_m', 'b_rad', 'dE')]
    if r.get('l_m'):
        s[4] = float(r['l_m'])
    efl.setdefault(r['scenario'].strip(), {}).setdefault(i, {})[loc] = s
scen = sorted(efl, key=lambda s: (len(s), s))

fig, axes = plt.subplots(len(scen), 2, figsize=(15, 3.2 * len(scen) + 1), squeeze=False)
reg_n = [n_inside(range(END + 1), REG[i]) for i in range(len(ids))]
for row, sc in enumerate(scen):
    ax, axd = axes[row]
    zr = np.asarray(Z[:END + 1])
    for i in np.where(fp1)[0]:                         # only rays that pass group 10's FP1 slit window
        pts = efl[sc][i]
        locs = list(range(Q7 + 1)) + [LOC['WF1_entrance'], LOC['WF1_exit']] + list(range(H3, END + 1))
        st = np.vstack([REG[i, :Q7 + 1], pts[LOC['WF1_entrance']], pts[LOC['WF1_exit']], prop(pts[H3][None], H3)[0]])
        nm = n_inside(locs, st)
        zm = np.array([Z[l] for l in locs])
        ax.plot(zm[:nm], st[:nm, 0] * 1000, color='tab:blue', lw=2.4, alpha=0.55, zorder=2)
        nr = reg_n[i]
        ax.plot(zr[:nr], REG[i, :nr, 0] * 1000, color='black', lw=0.8, zorder=3)
        # st is indexed by location (WF1 entrance/exit = 33/34 sit between Q7 = 32 and H3 = 35)
        n = min(nr, nm)
        if n > Q7:
            d = (st[Q7:n, 0] - REG[i, Q7:n, 0]) * 1000
            axd.plot(Z[Q7:n], d, color='tab:blue', lw=1.4 if fp1[i] else 0.9,
                     ls='-' if fp1[i] else ':', alpha=0.9)
    sel = (zr >= ZMIN) & (zr <= ZMAX)
    vals = [REG[i, :reg_n[i]][(zr[:reg_n[i]] >= ZMIN) & (zr[:reg_n[i]] <= ZMAX), 0] * 1000 for i in np.where(fp1)[0]]
    m = max(np.abs(np.concatenate(vals)).max(), 1.0)
    ax.set_xlim(ZMIN, ZMAX)
    ax.set_ylim(-1.3 * m, 1.3 * m)
    ax.set_autoscaley_on(False)
    draw_chamber_walls(ax, 'x', scale=1000, zmax=ZMAX + 1)
    axd.set_xlim(ZMIN, ZMAX)
    for a in (ax, axd):
        for loc, lab in ((FP2S, 'FP2 Slits'), (VD1638, 'VD1638')):
            a.axvline(Z[loc], color='tab:red', lw=0.8, ls='--')
            a.text(Z[loc], a.get_ylim()[1], f" {lab}", color='tab:red', fontsize=7, va='top', rotation=90)
        a.grid(alpha=0.3)
    axd.axhline(0, color='black', lw=0.6)
    ax.set_ylabel("x [mm]")
    axd.set_ylabel("dx = blue - black [mm]")
    ax.set_title(f"scenario {sc}: x vs z near FP2", fontsize=9, loc='left')
    axd.set_title("difference blue - black", fontsize=8, loc='left')
for a in axes[-1]:
    a.set_xlabel("z along beamline [m]")
fig.suptitle(f"Zoom near FP2: the {fp1.sum()} of the 10 sampled group-10 rays that pass the group's FP1 slit window "
             "(the ones entering the E2 mean)\nBlack = ECAT throughout; blue = efl-d7 WF1 model (Q7 Exit -> HEX3), "
             "ECAT elsewhere; grey = chamber walls", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
