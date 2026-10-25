"""
Check AgentSearch/fast_tracer.trace_rays against the Cython tracer
(ray_functions.process_all_rays + extract_coordinates) ray by ray, including
dipole/WF dB knobs (dipoleWFchanges), which the Emulator project's original
fast_tracer did not support. No COSY compile: uses the cached
pkl_Files/pg_June2025.pkl maps (tracer equivalence does not depend on which
lattice is loaded).

Ray sets: the default 189-ray stepped_uniform grid, plus 500 seeded random
rays spanning a wider phase space (so some are lost at varied locations).

Run:
    cd AgentSearch
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 test_fast_tracer.py
"""
import time, pickle
import numpy as np

import config
import helper_functions as hf
from ray_functions import process_all_rays, extract_coordinates
from fast_tracer import trace_rays

with open(config.beamline_geometry, 'rb') as f:
    _, polys = hf.find_polygons_at_z(pickle.load(f), config.z_coordinates)
k = config.element_names_stripped.index('DSSD')
_, groups = hf.load_optics_data('pg_June2025', config.pkl_directory)

rng = np.random.default_rng(12345)
R = np.column_stack([rng.uniform(-1.5e-3, 1.5e-3, 500), rng.uniform(-35e-3, 35e-3, 500),
                     rng.uniform(-1.5e-3, 1.5e-3, 500), rng.uniform(-35e-3, 35e-3, 500),
                     np.zeros(500), rng.uniform(-4e-2, 4e-2, 500), np.zeros(500), np.zeros(500)])


def ray_sets():
    # fresh objects every call: the Cython tracer can mutate input rays in place
    return {'default189': hf.generateInitialDistribution(None),
            'random500': [tuple([v] for v in row) for row in R]}


cases = {
    'no dB': [],
    'WF1 Exit dB': [['WF1 Exit', 'dB', 0.0025]],
    'multi dB (unsorted)': [['WF2 Exit', 'dB', -0.003], ['B1 Exit', 'dB', 0.01],
                            ['WF1 Exit', 'dB', 0.0025], ['B6 Exit', 'dB', 0.004]],
}

ok_all = True
for case, tc in cases.items():
    dwf = hf.prepareTuneChanges(tc)[2] if tc else None
    for name in ray_sets():
        rays_a, rays_b = ray_sets()[name], ray_sets()[name]
        t0 = time.perf_counter()
        a = extract_coordinates(process_all_rays(rays_a, groups[:k], config.z_coordinates[:k + 1], polys[:k + 1], dwf))
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = trace_rays(rays_b, groups[:k], polys[:k + 1], dwf)
        t_new = time.perf_counter() - t0

        same_ti = list(a[7]) == list(b[7])
        same_len = all(len(p) == len(q) for p, q in zip(a[5], b[5])) and all(len(p) == len(q) for p, q in zip(a[6], b[6]))
        # Both tracers keep propagating a ray after it is lost; far-off-axis lost
        # rays blow up (up to ~1e167 m), where float rounding alone exceeds any
        # absolute tolerance. So: strict absolute check on each ray's physical
        # path -- every location up to AND including its first-failure location
        # (the latter is what the 50x50cm DSSD analysis reads) -- and a relative
        # check on the post-loss remainder.
        n_loc = len(a[0][0])
        ti = np.array(b[7])
        phys = np.arange(n_loc)[None, :] <= np.minimum(ti, n_loc - 1)[:, None]
        d_phys, r_rest = {}, {}
        for c, lab in enumerate(['x', 'ax', 'y', 'ay', 'dE']):
            A, B = np.array(a[c], float), np.array(b[c], float)
            D = np.abs(A - B)
            d_phys[lab] = D[phys].max()
            fin = ~phys & np.isfinite(A) & np.isfinite(B)
            r_rest[lab] = (D[fin] / np.maximum(np.abs(A[fin]), 1e-12)).max() if fin.any() else 0.0
        ok = same_ti and same_len and max(d_phys.values()) < 1e-12 and max(r_rest.values()) < 1e-6
        ok_all &= ok
        print(f"{case:<20} {name:<10} old {t_old:6.2f} s  new {t_new:5.2f} s (x{t_old/t_new:4.0f}) | "
              f"loss idx identical: {same_ti} ({sum(t == k + 1 for t in b[7])}/{len(b[7])} reach DSSD) | "
              f"physical path max|d| {max(d_phys.values()):.1e} m | post-loss max rel {max(r_rest.values()):.1e} "
              f"| {'PASS' if ok else 'FAIL'}", flush=True)

# sanity: a dB knob must actually change something, or the comparison above proves nothing
base = trace_rays(ray_sets()['default189'], groups[:k], polys[:k + 1], None)
kick = trace_rays(ray_sets()['default189'], groups[:k], polys[:k + 1],
                  hf.prepareTuneChanges([['WF1 Exit', 'dB', 0.0025]])[2])
moved = np.max(np.abs(np.array(base[0]) - np.array(kick[0])))
print(f"\nsanity: WF1 Exit dB=+0.25% moves x by up to {moved*1e3:.3f} mm (must be > 0)")
ok_all &= moved > 0
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
