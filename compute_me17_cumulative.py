"""
Extracts the EXACT first-order (linear) transfer-matrix element ME(1,7)
(row=x, column=dm) from target to a given beamline location, by composing
the LOCAL element-to-element linear submatrices already present in the
parsed COSY matrix data (grouped_data_frames) -- per user request
2026-09-21.

Why this is exact (not an approximation): grouped_data_frames[k] holds
element k's LOCAL Taylor map (input = ray state exiting element k-1,
output = ray state exiting element k) -- see matrix_transform/
parameter_transform in ray_functions.pyx. The FIRST-ORDER part of a
composition of maps is exactly the product of their Jacobians (chain
rule) -- true regardless of what higher-order terms each local map has.
So composing each local map's own 8x8 LINEAR submatrix via ordinary
matrix multiplication gives the exact cumulative linear matrix, with no
need to compose the full nonlinear polynomial maps.

Column/row order in the parsed data (confirmed against
matrix_transform's df_x/df_ax/.../df_dz unpacking and
load_optics_data's group_size=8 chunking): [x, ax, y, ay, l, dE, dm, dz]
-- an 8-variable scheme where dm genuinely is variable 7 (1-indexed),
matching the user's ME(1,7) = (x-row, dm-column).

Does NOT need a fresh COSY run if the target tune's matrix is already
compiled/cached (e.g. "pg_June2025" for nominal) -- just re-reads the
existing pkl via load_optics_data.

Run:
    cd AgentSearch
    python3 compute_me17_cumulative.py
"""
import numpy as np

from config import element_names_stripped, pkl_directory
from helper_functions import load_optics_data

VARS_ORDER = ['x', 'ax', 'y', 'ay', 'l', 'dE', 'dm', 'dz']


def local_linear_matrix(group):
    """group = grouped_data_frames[k]: list of 8 DataFrames (one per
    output variable, in VARS_ORDER order). Returns the 8x8 local
    first-order transfer matrix R_k: R_k[out][inp] is the Taylor
    coefficient of the term purely linear in input variable `inp`
    (order 1 there, order 0 in every other tracked variable) for
    output variable `out`. Missing terms (not printed by COSY because
    they're exactly zero) default to 0.0."""
    R = np.zeros((8, 8))
    for out_idx, df in enumerate(group):
        for inp_idx, inp_name in enumerate(VARS_ORDER):
            mask = (df[inp_name] == 1)
            for other_name in VARS_ORDER:
                if other_name != inp_name:
                    mask &= (df[other_name] == 0)
            rows = df[mask]
            if len(rows) == 0:
                continue
            R[out_idx, inp_idx] = rows['Coefficient'].astype(float).sum()
    return R


def cumulative_linear_matrix(grouped_data_frames, final_index, verbose=False):
    """Composes local linear matrices into the cumulative first-order
    matrix from the start of the beamline (target) to element_names_stripped[final_index].

    Index convention (matches process_single_ray in ray_functions.pyx
    exactly -- found 2026-09-21 via a real mismatch against COSY's own
    native ME(1,7) at FP2 Slits): ray_matrices starts with ONE entry (the
    untransformed state, corresponding to element_names_stripped[0]), and
    each grouped_data_frames[k] applied appends one more entry. So the
    state AT element_names_stripped[m] is reached using
    grouped_data_frames[0..m-1] -- m matrices, NOT m+1. An earlier version
    of this function used range(final_index + 1) (one matrix too many,
    landing one element downstream of the intended target) -- caught
    because it gave -1.5154 vs COSY's native -1.4205 at FP2 Slits."""
    R_cum = np.eye(8)
    for k in range(final_index):
        R_k = local_linear_matrix(grouped_data_frames[k])
        R_cum = R_k @ R_cum
        if verbose:
            # arriving AT element_names_stripped[k+1] after k+1 matrices
            # applied (0..k) -- NOT element_names_stripped[k], which is
            # where this segment STARTED from. Fixed 2026-09-21 (was
            # mislabeled by one position; the underlying calculation was
            # already correct, only this printout was off).
            print(f"  after applying segment {k} [{element_names_stripped[k]} -> "
                  f"{element_names_stripped[k + 1]}]: cumulative ME(1,7)={R_cum[0, 6]:.6f}")
    return R_cum


if __name__ == "__main__":
    FOX_FILE = "pg_June2025"
    _, grouped_data_frames = load_optics_data(FOX_FILE, pkl_directory)
    print(f"Loaded {len(grouped_data_frames)} elements' local matrices from {FOX_FILE}.pkl "
          f"({len(element_names_stripped)} elements expected)\n")

    for label, elem in [("FP2 Slits", "FP2 Slits"), ("FP3 Slits", "FP3 Slits & VD1783")]:
        idx = element_names_stripped.index(elem)
        R_cum = cumulative_linear_matrix(grouped_data_frames, idx)
        print(f"{label} (index {idx}): cumulative linear matrix from target computed.")
        print(f"  ME(1,7) [x row, dm column] = {R_cum[0, 6]:.6f}")
        print(f"  (full first row, x-output vs all 8 inputs [x,ax,y,ay,l,dE,dm,dz]): {R_cum[0]}\n")
