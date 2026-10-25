"""
E2 observables per WF1 EFL scenario, and whether the change is degenerate
with ECAT's existing WF1 knobs (2026-10-19).

Inputs:
  Results/rays_q7exit_e2_all_groups.csv   E2 beam at the Q7 Exit (ray_uid, group, all 8 coords)
  <efl csv> (argument)                    efl-d7: scenario, description, location, ray_uid,
                                          x_m, a_rad, y_m, b_rad, l_m, dE, lost
  Results/wf1_knob_jacobian_results.csv   (optional) WF1 knob derivatives, wf1_knob_jacobian.py

For each scenario: ECAT target -> Q7 Exit (the file), efl-d7's model Q7 Exit ->
H3 entrance, ECAT from the H3 entrance to the readout. Apertures are checked
at every location before the readout, including efl-d7's WF1 entrance/exit
points (efl-d7's model has none). Ray selection, group means and fits exactly
as fp3_eval / fp2_metrics / fp3_metrics: a ray counts if it is inside the
apertures at every location before the readout (FP1-slit cut already applied
in the input file).

"ECAT" reference = the same rays continued with ECAT's own maps from the Q7
Exit (identical to fp3_eval's standard FP2/FP3 evaluation at nominal; checked).

Degeneracy test: d = obs(scenario) - obs(reference) for the 8 observables
[M12, Off12, M16, Off16] x [FP2, FP3], in units of their loss uncertainties
(FP2: the 4-term FP2 family loss sigmas 0.02, 0.2 mm, 0.03, 1.5 mm; FP3: the FP3
loss metric H^1/2). Least-squares fit d ~ J k with J = WF1 knob derivatives
(WF1_Exit_dB, length, rhoE, rhoB, X); "explained" = 1 - |residual|^2 / |d|^2.
Done against the ECAT reference and for pairs of scenarios that differ only
in the B EFL (3 vs 4, 5 vs 6).

Output: Results/wf1_efl_scenarios_observables.csv, the per-group mean X behind
them in Results/wf1_efl_scenarios_group_means.csv, and a printed report.

Run:
    cd AgentSearch
    python3 analyse_wf1_efl_scenarios.py <efl csv>
"""
import os
import sys
import csv
import pickle
import numpy as np

from config import element_names_stripped as NAMES, z_coordinates, beamline_geometry, pkl_directory, fox_file
from helper_functions import load_optics_data, find_polygons_at_z
from fast_tracer import compile_map, apply_map
from shapely import contains_xy
import fp3_eval
import fp2_metrics
import fp3_metrics
import fp3_loss

RAYS = "Results/rays_q7exit_e2_all_groups.csv"
KNOB_FILE = "Results/wf1_knob_jacobian_results.csv"
OUT = "Results/wf1_efl_scenarios_observables.csv"
MEANS_OUT = "Results/wf1_efl_scenarios_group_means.csv"
Q7, H3 = NAMES.index('Q7 Exit'), NAMES.index('H3 Entrance')
LOC = {'WF1_entrance': NAMES.index('WF1 Entrance'), 'WF1_exit': NAMES.index('WF1 Exit'), 'H3_entrance': H3}
READ = {'FP2': NAMES.index('VD1638'), 'FP3': NAMES.index('FP3 Slits & VD1783')}
GMOD = {'FP2': fp2_metrics, 'FP3': fp3_metrics}
OBS = ['M12', 'Offset12', 'M16', 'Offset16']

rows = [r for r in csv.DictReader(l for l in open(RAYS) if not l.startswith('#'))]
uid = np.array([int(r['ray_uid']) for r in rows])
grp = np.array([r['group'] for r in rows])
X7 = np.array([[float(r[c]) for c in ('x_m', 'a_rad', 'y_m', 'b_rad', 'l_m', 'dE', 'dm', 'dz')] for r in rows])
pos = {u: i for i, u in enumerate(uid)}

_, groups = load_optics_data(fox_file, pkl_directory)
last = max(READ.values())
maps = {k: compile_map(groups[k]) for k in range(Q7, last)}
with open(beamline_geometry, 'rb') as f:
    _, polys = find_polygons_at_z(pickle.load(f), z_coordinates)


def ok_at(S, loc):
    return np.ones(len(S), bool) if polys[loc] is None else contains_xy(polys[loc], S[:, 0], S[:, 2])


def observables(states_by_loc, start):
    """states_by_loc: {loc: (n, 8)} for locs before `start`; propagate from `start` with ECAT."""
    good = np.ones(len(uid), bool)
    for loc, S in states_by_loc.items():
        good &= ok_at(S, loc)
    S = states_by_loc[start]
    out, means = {}, {}
    for loc in range(start, last + 1):
        for plane, idx in READ.items():
            if loc == idx:
                m = GMOD[plane]
                mean_x = {}
                for g in m.ALL_GROUPS:
                    sel = good & (grp == g)
                    if not sel.any():
                        raise RuntimeError(f"no rays left for group {g} at {plane}")
                    mean_x[g] = float((-S[sel, 0] * 1000).mean())
                out[plane] = fp3_eval.fit_observables(mean_x, plane)
                means[plane] = mean_x
        if loc == last:
            break
        good &= ok_at(S, loc)
        S = apply_map(S, *maps[loc])
    return np.concatenate([out['FP2'], out['FP3']]), means


# ECAT reference: propagate from the Q7 Exit with ECAT's maps
ref, ref_means = observables({Q7: X7}, Q7)

# check against the standard evaluation (fp3_eval) at nominal
std = np.concatenate([fp3_eval.observables_from_maps(groups, None, 'FP2', 3000)[0],
                      fp3_eval.observables_from_maps(groups, None, 'FP3', 3000)[0]])
print(f"check: ECAT reference vs standard fp3_eval evaluation, max |diff| = {np.max(np.abs(ref - std)):.1e}")

# efl-d7 scenarios
efl = {}
desc = {}
for r in csv.DictReader(open(sys.argv[1])):
    sc = r['scenario'].strip()
    desc.setdefault(sc, (r.get('description') or '').strip())
    st = efl.setdefault(sc, {k: np.full((len(uid), 8), np.nan) for k in LOC})
    i = pos[int(r['ray_uid'])]
    s = X7[i].copy()
    s[[0, 1, 2, 3, 5]] = [float(r['x_m']), float(r['a_rad']), float(r['y_m']), float(r['b_rad']), float(r['dE'])]
    if r.get('l_m') not in (None, ''):
        s[4] = float(r['l_m'])
    if str(r.get('lost', '0')).strip().lower() in ('1', 'true', 'yes'):
        s[[0, 2]] = 1e3                                   # outside every aperture
    st[r['location'].strip()][i] = s
results = {'ECAT': ref}
group_means = {'ECAT': ref_means}
for sc in sorted(efl, key=lambda s: (len(s), s)):
    st = efl[sc]
    miss = np.isnan(st['H3_entrance'][:, 0]).sum()
    if miss:
        raise RuntimeError(f"scenario {sc}: {miss} rays missing at H3_entrance")
    byloc = {Q7: X7}
    for k in ('WF1_entrance', 'WF1_exit'):
        if not np.isnan(st[k][:, 0]).all():
            byloc[LOC[k]] = np.where(np.isnan(st[k]), 0.0, st[k])
    byloc[H3] = st['H3_entrance']
    results[sc], group_means[sc] = observables(byloc, H3)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "description"] + [f"{p}_{o}" for p in ('FP2', 'FP3') for o in OBS])
    for k, v in results.items():
        w.writerow([k, desc.get(k, 'ECAT maps throughout')] + list(v))

with open(MEANS_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "description", "plane", "group", "mean_x_mm"])
    for k, m in group_means.items():
        for plane, mx in m.items():
            for g, v in mx.items():
                w.writerow([k, desc.get(k, 'ECAT maps throughout'), plane, g, v])

# whitening for the 8 observables
ev, evec = np.linalg.eigh(fp3_loss.H)
W = np.zeros((8, 8))
W[:4, :4] = np.diag(1 / np.array([0.02, 0.2, 0.03, 1.5]))
W[4:, 4:] = evec @ np.diag(np.sqrt(ev)) @ evec.T

print(f"\n{'case':<6}" + "".join(f"{p+' '+o:>14}" for p in ('FP2', 'FP3') for o in OBS))
for k, v in results.items():
    print(f"{k:<6}" + "".join(f"{x:14.4f}" for x in v))
print("\nchange vs ECAT (scenario - ECAT):")
for k, v in results.items():
    if k != 'ECAT':
        d = v - ref
        print(f"{k:<6}" + "".join(f"{x:+14.4f}" for x in d) + f"   |d| = {np.linalg.norm(W @ d):.2f} sigma")

if not os.path.exists(KNOB_FILE):
    print(f"\n{KNOB_FILE} not found: run wf1_knob_jacobian.py for the degeneracy test.")
    sys.exit(0)
K = {}
for r in csv.DictReader(open(KNOB_FILE)):
    K.setdefault((r['knob'], round(float(r['value']), 9)), {})[r['plane']] = np.array([float(r[o]) for o in OBS])
nom = np.concatenate([K[('NOMINAL', 0.0)]['FP2'], K[('NOMINAL', 0.0)]['FP3']])
print(f"\ncheck: knob-run nominal vs ECAT reference, max |diff| = {np.max(np.abs(nom - ref)):.1e}")
names, cols = [], []
for k in ('WF1_Exit_dB', 'WF1_length', 'WF1_rhoE', 'WF1_rhoB', 'WF1_X'):
    vals = sorted(v for kk, v in K if kk == k)
    if len(vals) != 2:
        continue
    vm, vp = vals
    jm = np.concatenate([K[(k, vm)]['FP2'], K[(k, vm)]['FP3']])
    jp = np.concatenate([K[(k, vp)]['FP2'], K[(k, vp)]['FP3']])
    names.append(k)
    cols.append((jp - jm) / (vp - vm))
J = np.array(cols).T                                   # 8 x n_knobs, per native unit
Jw = W @ J
print(f"\nWF1 knobs: {names}; singular values of the whitened 8x{len(names)} J (per native unit): "
      f"{np.round(np.linalg.svd(Jw, compute_uv=False), 3)}")


def fit(d, label):
    dw = W @ d
    k, *_ = np.linalg.lstsq(Jw, dw, rcond=None)
    res = dw - Jw @ k
    expl = 1 - (res @ res) / (dw @ dw) if dw @ dw > 0 else float('nan')
    kk = ", ".join(f"{n} {v:+.3g}" for n, v in zip(names, k))
    print(f"  {label:<14} |d| {np.linalg.norm(dw):7.2f} sigma -> residual {np.linalg.norm(res):7.2f} sigma "
          f"({100*expl:5.1f}% explained)  knobs: {kk}")


print("\nCan the WF1 knobs reproduce each change? (least squares in sigma units)")
for k, v in results.items():
    if k != 'ECAT':
        fit(v - ref, f"{k} vs ECAT")
for a, b in (('3', '4'), ('5', '6')):
    if a in results and b in results:
        fit(results[b] - results[a], f"{b} - {a} (EFL_B)")
print(f"\nwrote {OUT}")
