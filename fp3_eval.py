"""
Single-pass E2 evaluation at FP2 (VD1638) or FP3 (VD1783) (2026-10-04).
Same physics and filtering as fp2_metrics / fp3_metrics.compute_M12_M16,
but:
  - one COSY compile per lattice, shared by any number of dB variants
    (dB knobs are applied during tracing, not in the .fox file);
  - beamline polygons and the groups' rays are built once per process and
    cached (aperture rays are seeded with 42 on every call, so they are
    identical each time anyway);
  - all groups' rays are traced together in one fast_tracer call;
  - a compile's own artifacts (Matrix_Files/<name>/, pkl_Files/<name>.pkl,
    COSY10.2/<name>.fox) are deleted once the maps are loaded. The base
    lattice itself (no .fox edits) is compiled once under its own name and
    kept as a cached baseline.

Ray selection per group (identical to fp2/fp3_metrics): a ray counts if
transmission_index >= readout index and its X at 'FP1 Slits' (-x*1000) lies
inside the group's FP1 slit window; readout is -x*1000 at the plane.
"""
import os
import shutil
import pickle
import numpy as np

from config import (element_names_stripped, z_coordinates, beamline_geometry,
                    pkl_directory, matrix_directory, COSY_directory, fox_file)
from helper_functions import (prepareTuneChanges, prepareCOSYchanges, createCosyFile,
                              run_cosy_and_save_fort, process_fort_files, load_optics_data,
                              find_polygons_at_z, generateInitialDistribution)
from fast_tracer import trace_rays, rays_to_array
from ViewerAnalysis import groupsInfo
from fp2_metrics import build_initial_distribution
import fp2_metrics
import fp3_metrics
import fp3_loss

FP1_SLITS_INDEX = element_names_stripped.index('FP1 Slits')
VD1783_INDEX = element_names_stripped.index('FP3 Slits & VD1783')
PLANES = {
    'FP2': {'index': element_names_stripped.index('VD1638'), 'groups': fp2_metrics},
    'FP3': {'index': VD1783_INDEX, 'groups': fp3_metrics},
}
NRAYS = 3000

_cache = {}


def _setup(plane, nrays):
    key = (plane, nrays)
    if key in _cache:
        return _cache[key]
    if 'polys' not in _cache:
        with open(beamline_geometry, 'rb') as f:
            _cache['polys'] = find_polygons_at_z(pickle.load(f), z_coordinates)[1]
    idx = PLANES[plane]['index']
    arrays, labels = [], []
    for g in PLANES[plane]['groups'].ALL_GROUPS:
        rays = generateInitialDistribution(build_initial_distribution(g, nrays))
        arrays.append(rays_to_array(rays))
        labels += [g] * len(rays)
    _cache[key] = dict(polys=_cache['polys'][:idx + 1], rays=np.vstack(arrays), labels=np.array(labels))
    return _cache[key]


def fit_observables(mean_x_by_group, plane):
    """{group: mean X [mm]} -> np.array([M12, Off12, M16, Off16]) using the plane's group sets."""
    m = PLANES[plane]['groups']
    a = np.array([m.DALPHAX_GROUPS[g] for g in m.DALPHAX_GROUPS])
    M12, Off12 = np.polyfit(a, [mean_x_by_group[g] for g in m.DALPHAX_GROUPS], 1)
    e = np.array([m.DE_GROUPS[g] for g in m.DE_GROUPS])
    M16, Off16 = np.polyfit(e, [mean_x_by_group[g] for g in m.DE_GROUPS], 1)
    return np.array([M12, Off12, M16, Off16])


def observables_from_maps(groups, dipoleWFchanges, plane='FP3', nrays=NRAYS):
    """Trace all of the plane's groups through `groups` (full map list) -> (obs, mean_x_by_group, n_by_group)."""
    c = _setup(plane, nrays)
    idx = PLANES[plane]['index']
    all_x, _, _, _, _, _, _, ti = trace_rays(c['rays'], groups[:idx], c['polys'], dipoleWFchanges)
    X = -np.array(all_x) * 1000
    ti = np.array(ti)
    mean_x, n = {}, {}
    for g in PLANES[plane]['groups'].ALL_GROUPS:
        lo, hi = groupsInfo[g]['fp1_slits']['slits_mm']
        sel = (c['labels'] == g) & (ti >= idx) & (X[:, FP1_SLITS_INDEX] >= lo) & (X[:, FP1_SLITS_INDEX] <= hi)
        if not sel.any():
            raise RuntimeError(f"No rays survived the FP1-slit+transmission cut for group {g}")
        mean_x[g] = float(X[sel, idx].mean())
        n[g] = int(sel.sum())
    return fit_observables(mean_x, plane), mean_x, n


def _cleanup(name):
    if name == fox_file:
        return
    shutil.rmtree(os.path.join(matrix_directory, name), ignore_errors=True)
    for path in (os.path.join(pkl_directory, f"{name}.pkl"), os.path.join(COSY_directory, f"{name}.fox")):
        if os.path.exists(path):
            os.remove(path)


def compile_maps(lattice_tuneChanges):
    """Compile the lattice defined by the non-dB entries of lattice_tuneChanges; return the map list.
    With no .fox edits, the base lattice is compiled once under its own name and reused."""
    params, values, _ = prepareTuneChanges(lattice_tuneChanges)
    cosyChanges, name = prepareCOSYchanges(params, values)
    if not cosyChanges:
        if not os.path.exists(os.path.join(pkl_directory, f"{fox_file}.pkl")):
            run_cosy_and_save_fort(f"./cosy {fox_file}.fox", fox_file)
            process_fort_files(fox_file)
        return load_optics_data(fox_file, pkl_directory)[1]
    try:
        createCosyFile(cosyChanges, file_name=name)
        run_cosy_and_save_fort(f"./cosy {name}.fox", name)
        process_fort_files(name)
        _, groups = load_optics_data(name, pkl_directory)
    finally:
        _cleanup(name)
    return groups


def evaluate_variants(lattice_tuneChanges, dB_variants, plane='FP3', nrays=NRAYS, loss_fn=fp3_loss.loss):
    """One compile for lattice_tuneChanges, then one trace per variant.
    dB_variants: list of extra dB tuneChanges lists ([] = no extra).
    Returns a list of (obs, loss, n_by_group), one per variant."""
    groups = compile_maps(lattice_tuneChanges)
    out = []
    for extra in dB_variants:
        dwf = prepareTuneChanges(lattice_tuneChanges + extra)[2]
        obs, _, n = observables_from_maps(groups, dwf or None, plane, nrays)
        out.append((obs, loss_fn(obs), n))
    return out
