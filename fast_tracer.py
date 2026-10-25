"""
Vectorized ray tracer: same physics as ray_functions.process_all_rays +
extract_coordinates, but pushes ALL rays through each element's COSY map at
once instead of one ray at a time (~100x faster).

Copied 2026-10-04 from Emulator/emulatorAgent/fast_tracer.py (verified there
ray-by-ray identical to the Cython tracer), with one addition: support for
dipoleWFchanges (dipole/WF dB tune changes), which the original rejected.
That logic mirrors process_single_ray (ray_functions.pyx:569-584) exactly,
including its quirks:
  - a single forward-only pointer k over dipoleWFchanges, advanced only when
    the current element index matches dipoleWFchanges[k][0] (callers get a
    sorted list from prepareTuneChanges);
  - before the matched element, the stored incoming state's dE is shifted by
    -2*dB in place (so the recorded dE at that location is shifted too),
    and the outgoing state's dE gets +2*dB back.

    from fast_tracer import trace_rays
    all_x, all_ax, all_y, all_ay, all_dE, tx, ty, ti = trace_rays(rays, groups[:k], polygons[:k + 1], dipoleWFchanges)
"""
import numpy as np
from shapely import contains_xy

VARS = ['x', 'ax', 'y', 'ay', 'l', 'dE', 'dm', 'dz']


def compile_map(group):
    """8 DataFrames -> (exponents (U, 8) int, coefficients (U, 8)): union of monomials over the 8 outputs."""
    rows = {}
    for out, df in enumerate(group):
        exps = df[VARS].to_numpy(float).astype(int)
        coef = df['Coefficient'].to_numpy(float)
        for e, c in zip(map(tuple, exps), coef):
            rows.setdefault(e, np.zeros(8))[out] += c
    E = np.array(list(rows.keys()), dtype=int).reshape(-1, 8)
    C = np.array(list(rows.values())).reshape(-1, 8)
    return E, C


def apply_map(X, E, C):
    """X (N, 8) ray states -> (N, 8) after the map."""
    kmax = int(E.max()) if E.size else 0
    # rays already lost upstream keep being propagated and can overflow; their
    # values past the loss point are never used, so silence the warnings
    with np.errstate(over='ignore', invalid='ignore'):
        powers = np.ones((kmax + 1,) + X.shape)      # powers[k, n, j] = X[n, j]**k  (0**0 = 1)
        for k in range(1, kmax + 1):
            powers[k] = powers[k - 1] * X
        M = np.ones((X.shape[0], E.shape[0]))
        for j in range(8):
            M *= powers[E[:, j], :, j].T              # (N, U)
        return M @ C


def rays_to_array(rays):
    """Rays in ray_functions format (tuples of 8 one-element lists) -> (N, 8) array."""
    return np.array([[r[k][0] for k in range(8)] for r in rays], dtype=float)


def trace_rays(rays, groups, polygons, dipoleWFchanges=None):
    """Drop-in for extract_coordinates(process_all_rays(rays, groups, z, polygons, dipoleWFchanges)).
    groups: element maps to apply (len L); polygons: one per location (len L + 1, None = open)."""
    X = rays_to_array(rays) if not isinstance(rays, np.ndarray) else rays.astype(float)
    states = [X]
    k = 0
    for j, g in enumerate(groups):
        hit = bool(dipoleWFchanges) and k < len(dipoleWFchanges) and j == dipoleWFchanges[k][0]
        if hit:
            dB = dipoleWFchanges[k][1]
            X = X.copy()
            X[:, 5] -= 2 * dB                         # dE/E = 2*dB/B, applied to the stored incoming state
            states[-1] = X
        E, C = compile_map(g)
        X = apply_map(X, E, C)
        if hit:
            X[:, 5] += 2 * dB
            k += 1
        states.append(X)
    S = np.stack(states, axis=1)                      # (N, L+1, 8)
    N, n_loc = S.shape[0], S.shape[1]

    inside = np.ones((N, n_loc), dtype=bool)
    for i, poly in enumerate(polygons[:n_loc]):
        if poly is not None:
            inside[:, i] = contains_xy(poly, S[:, i, 0], S[:, i, 2])
    # transmission index = number of locations before the first failure (n_loc if never fails)
    ti = np.where(inside.all(axis=1), n_loc, np.argmin(inside, axis=1))

    all_x, all_ax, all_y, all_ay, all_dE = (S[:, :, c].tolist() for c in (0, 1, 2, 3, 5))
    tx = [all_x[n][:ti[n]] for n in range(N)]
    ty = [all_y[n][:ti[n]] for n in range(N)]
    return all_x, all_ax, all_y, all_ay, all_dE, tx, ty, ti.tolist()
