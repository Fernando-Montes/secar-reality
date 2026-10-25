"""
Beam envelopes with efl-d7's model replacing ECAT for the Q7 Exit -> HEX3
entrance segment (2026-10-19, requested by Fernando).

Rays: the 10 ray_ids in Results/hex3_scenario_ray_ids.txt, from
Results/rays_q7exit_group10_10000.csv (E2 group 10, seed 42, 10000 rays;
all 10 reach the Q7 Exit). Lattice: nominal pg_June2025_corrected,
MISALIGN = 1 (cached maps, no COSY). Tracing: fast_tracer maps, apertures from
the beamline geometry, as in ECAT.

Black ("regular"): ECAT all the way, target -> DSSD.
Blue ("modified"): ECAT target -> Q7 Exit (identical to black), then efl-d7's
coordinates at WF1 Entrance / WF1 Exit (if provided) and at the H3 Entrance,
then ECAT again from the H3 Entrance to the DSSD (same maps and apertures).
Apertures are checked at every location for both, including efl-d7's points;
a lost ray is drawn up to its last location inside the apertures and marked
with an 'x'. Where efl-d7 gives no intermediate points, the blue line goes
straight from the Q7 Exit to the H3 Entrance (dashed).

Input CSV from efl-d7 (path as argument), columns:
  scenario, ray_id, x_m, a_rad, y_m, b_rad, dE   (required)
  l_m, location, lost, description               (optional)
location in {WF1_entrance, WF1_exit, H3_entrance} (default H3_entrance). If
l_m is missing, l at the H3 Entrance is taken from the regular ECAT ray; dm,
dz always from the regular ray (both are unchanged by this segment in ECAT).

Plot: x = COSY x [mm] and y [mm] vs z, as in helper_functions.plot_rays (no
sign flip). One row per scenario: x vs z, y vs z.

Run:
    cd AgentSearch
    python3 plot_hex3_scenarios.py <efl_csv>
"""
import sys
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import element_names_stripped as NAMES, z_coordinates as Z, beamline_geometry, pkl_directory, fox_file
from helper_functions import load_optics_data, find_polygons_at_z
from fast_tracer import compile_map, apply_map
from shapely import contains_xy
from beamline_walls import draw_chamber_walls, zoom_to_rays

RAY_FILE = "Results/rays_q7exit_group10_10000.csv"
IDS_FILE = "Results/hex3_scenario_ray_ids.txt"
OUT_PNG = "Results/hex3_scenarios_envelope.png"
OUT_CSV = "Results/hex3_scenarios_traced.csv"
Q7 = NAMES.index('Q7 Exit')
LOC = {'WF1_entrance': NAMES.index('WF1 Entrance'), 'WF1_exit': NAMES.index('WF1 Exit'),
       'H3_entrance': NAMES.index('H3 Entrance')}
H3 = LOC['H3_entrance']
DSSD = NAMES.index('DSSD')

efl_path = sys.argv[1]
ids = [int(v) for v in open(IDS_FILE).read().split()]

# initial states (x0, a0, y0, b0, l=0, dE0, dm=0, dz=0) for the chosen rays
rows = {int(r['ray_id']): r for r in csv.DictReader(l for l in open(RAY_FILE) if not l.startswith('#'))}
X0 = np.array([[float(rows[i]['x0_m']), float(rows[i]['a0_rad']), float(rows[i]['y0_m']),
                float(rows[i]['b0_rad']), 0.0, float(rows[i]['dE0']), 0.0, 0.0] for i in ids])

_, groups = load_optics_data(fox_file, pkl_directory)
maps = [compile_map(g) for g in groups[:DSSD]]
with open(beamline_geometry, 'rb') as f:
    _, polys = find_polygons_at_z(pickle.load(f), Z)


def propagate(X, start):
    """states at locations start..DSSD, given X at location `start`: (n_rays, n_loc, 8)."""
    S = [X]
    for k in range(start, DSSD):
        S.append(apply_map(S[-1], *maps[k]))
    return np.stack(S, axis=1)


def inside(x, y, loc):
    return np.ones_like(x, bool) if polys[loc] is None else contains_xy(polys[loc], x, y)


REG = propagate(X0, 0)                                  # (10, DSSD+1, 8)
assert np.allclose(REG[:, Q7, [0, 1, 2, 3, 5]],
                   [[float(rows[i][c]) for c in ('x_m', 'a_rad', 'y_m', 'b_rad', 'dE')] for i in ids]), \
    "regular trace does not reproduce the Q7-exit coordinates in the ray file"


def first_loss(path_locs, path_states):
    """index into path of the first location outside the apertures, or None."""
    for j, (loc, s) in enumerate(zip(path_locs, path_states)):
        if not inside(np.array([s[0]]), np.array([s[2]]), loc)[0]:
            return j
    return None


# efl-d7 data: scen -> ray -> {location: state(8)}, plus lost flags / descriptions
efl, desc, lost_flag = {}, {}, {}
for r in csv.DictReader(open(efl_path)):
    sc, rid = r['scenario'].strip(), int(r['ray_id'])
    if rid not in ids:
        continue
    loc = (r.get('location') or 'H3_entrance').strip()
    if r.get('description'):
        desc.setdefault(sc, r['description'].strip())
    if str(r.get('lost', '0')).strip().lower() in ('1', 'true', 'yes'):
        lost_flag[(sc, rid)] = True
        continue
    i = ids.index(rid)
    s = REG[i, LOC[loc]].copy()                         # dm, dz (and l if not given) from the regular ray
    s[[0, 1, 2, 3, 5]] = [float(r['x_m']), float(r['a_rad']), float(r['y_m']), float(r['b_rad']), float(r['dE'])]
    if r.get('l_m') not in (None, ''):
        s[4] = float(r['l_m'])
    efl.setdefault(sc, {}).setdefault(rid, {})[loc] = s
scenarios = sorted(set(efl) | {sc for sc, _ in lost_flag})
print(f"{len(scenarios)} scenarios: {scenarios}")

traced = []
fig, axes = plt.subplots(len(scenarios), 2, figsize=(16, 3.0 * len(scenarios) + 1), sharex=True, squeeze=False)
reg_locs = list(range(DSSD + 1))
for row, sc in enumerate(scenarios):
    axx, axy = axes[row]
    for i, rid in enumerate(ids):
        # regular
        jl = first_loss(reg_locs, REG[i])
        n = len(reg_locs) if jl is None else jl
        for ax, c in ((axx, 0), (axy, 2)):
            ax.plot(Z[:n], REG[i, :n, c] * 1000, color='black', lw=0.8, zorder=3)
            if jl is not None:
                ax.plot(Z[jl], REG[i, jl, c] * 1000, 'kx', ms=5, zorder=4)
        # modified
        pts = efl.get(sc, {}).get(rid, {})
        if (sc, rid) in lost_flag or 'H3_entrance' not in pts:
            for ax, c in ((axx, 0), (axy, 2)):
                ax.plot(Z[:Q7 + 1], REG[i, :Q7 + 1, c] * 1000, color='tab:blue', lw=2.4, alpha=0.55, zorder=2)
                ax.plot(Z[Q7], REG[i, Q7, c] * 1000, 'x', color='tab:blue', ms=6)
            traced.append({"scenario": sc, "ray_id": rid, "regular_lost_at": "" if jl is None else NAMES[jl],
                           "modified_lost_at": "lost in efl-d7 segment"})
            continue
        mid = [k for k in ('WF1_entrance', 'WF1_exit') if k in pts]
        locs = list(range(Q7 + 1)) + [LOC[k] for k in mid]
        states = list(REG[i, :Q7 + 1]) + [pts[k] for k in mid]
        down = propagate(pts['H3_entrance'][None, :], H3)[0]
        locs += list(range(H3, DSSD + 1))
        states += list(down)
        states = np.array(states)
        jm = first_loss(locs, states)
        n = len(locs) if jm is None else jm
        zz = np.array([Z[l] for l in locs])
        for ax, c in ((axx, 0), (axy, 2)):
            ax.plot(zz[:n], states[:n, c] * 1000, color='tab:blue', lw=2.4, alpha=0.55, zorder=2)
            if not mid and n > Q7 + 1:
                ax.plot([Z[Q7], Z[H3]], [states[Q7, c] * 1000, states[Q7 + 1, c] * 1000], color='tab:blue',
                        lw=2.4, alpha=0.55, ls='--', zorder=2)
            if jm is not None:
                ax.plot(zz[jm], states[jm, c] * 1000, 'x', color='tab:blue', ms=6)
        rec = {"scenario": sc, "ray_id": rid, "regular_lost_at": "" if jl is None else NAMES[jl],
               "modified_lost_at": "" if jm is None else NAMES[locs[jm]]}
        for lab, loc in (('H3', H3), ('FP2_VD1638', NAMES.index('VD1638')),
                         ('FP3_VD1783', NAMES.index('FP3 Slits & VD1783')), ('DSSD', DSSD)):
            jj = locs.index(loc)
            rec.update({f"dx_{lab}_mm": (states[jj, 0] - REG[i, loc, 0]) * 1000,
                        f"dy_{lab}_mm": (states[jj, 2] - REG[i, loc, 2]) * 1000})
        traced.append(rec)
    title = f"scenario {sc}" + (f": {desc[sc]}" if sc in desc else "")
    axx.set_title(title + "  -- x", fontsize=9, loc='left')
    axy.set_title("y", fontsize=9, loc='left')
    for ax, lab, c in ((axx, "x [mm]", 'x'), (axy, "y [mm]", 'y')):
        ax.axvspan(Z[Q7], Z[H3], color='tab:blue', alpha=0.08)
        zoom_to_rays(ax)
        draw_chamber_walls(ax, c, scale=1000, zmax=Z[DSSD])
        ax.set_xlim(0, Z[DSSD] + 0.5)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("z along beamline [m]")
fig.suptitle("10 rays (E2 group 10). Black (thin, on top): ECAT throughout.  Blue (wide, underneath): efl-d7 model\n"
             "for Q7 Exit -> HEX3 entrance (shaded), ECAT elsewhere.  Black on blue = identical;  x = lost at an aperture;  "
             "grey = chamber walls", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=130)

keys = sorted({k for r in traced for k in r}, key=lambda k: (k not in ("scenario", "ray_id"), k))
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(traced)
print(f"wrote {OUT_PNG} and {OUT_CSV}")
for sc in scenarios:
    t = [r for r in traced if r["scenario"] == sc and "dx_H3_mm" in r]
    if t:
        print(f"  {sc}: max |dx| at H3 {max(abs(r['dx_H3_mm']) for r in t):.3f} mm, at DSSD "
              f"{max(abs(r['dx_DSSD_mm']) for r in t):.3f} mm; regular lost {sum(bool(r['regular_lost_at']) for r in t)}, "
              f"modified lost {sum(bool(r['modified_lost_at']) for r in t)}")
