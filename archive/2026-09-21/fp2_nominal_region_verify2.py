"""
Second Gauss-Newton iteration on the near-nominal minimum-norm solution:
the first iteration (fp2_nominal_region_verify.py) reached loss=0.787 at
Q6-2.445%/Q2-0.849%/Q5+0.117%/WF1+0.352%, with a small residual gap from
the linear prediction due to nonlinearity at this magnitude. This point
adds one more Newton-step correction (computed from the same OAT-slope
Jacobian, no new sim needed for the correction itself) to close that gap:
    Q6_B_SC = -0.02437, Q2_B_SC = -0.00819, Q5_B_SC = +0.00136,
    WF1_Exit_dB = +0.00449

Run:
    cd AgentSearch
    python3 fp2_nominal_region_verify2.py
"""
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn

TUNE_CHANGES = [['Q6', 'B_SC', -0.02437], ['Q2', 'B_SC', -0.00819],
                 ['Q5', 'B_SC', 0.00136], ['WF1 Exit', 'dB', 0.00449]]
NRAYS = 3000
COSY_FILE = "pg_June2025_nomregion_verify2"

if __name__ == "__main__":
    M12, M16, per_group, usedFoxFile = compute_M12_M16(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=COSY_FILE,
        ta_y_offset=0.0, verbose=True)

    loss = loss_fn(M12, M16)

    width = compute_fp2_width(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=usedFoxFile,
        run_cosy_flag=False, verbose=True)

    print(f"\nGauss-Newton corrected near-nominal solution "
          f"(Q6-2.437%, Q2-0.819%, Q5+0.136%, WF1+0.449%):")
    print(f"  M12_sim={M12:.4f}  M16_sim={M16:.4f}  (target: -0.2472, -0.3842)")
    print(f"  loss (v2, M12+M16-only) = {loss:.3f}")
    print(f"  fp2_slits_std={width['fp2_std']:.3f} mm   vd1638_std={width['vd1638_std']:.3f} mm")
    print(f"\nFor comparison: 1st-iteration near-nominal solution   loss=0.787")
    print(f"                 far-off BO v3 best                    loss=4.575")
    print(f"                 user's full E2-best                   loss=0.524")
