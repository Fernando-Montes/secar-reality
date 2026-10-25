"""
Verifies the minimum-norm linear solution (2026-08-10, solved analytically
from the OAT screen's per-knob slopes via Moore-Penrose pseudoinverse of
the 2x4 [Q6,Q2,Q5,WF1] -> [M12,M16] sensitivity matrix, no COSY runs
needed for that step) with an actual joint COSY evaluation:
    Q6_B_SC = -0.02445, Q2_B_SC = -0.00849, Q5_B_SC = +0.00117,
    WF1_Exit_dB = +0.00352
Predicted (linear): M12=-0.2472, M16=-0.3842 (exact target hit, by
construction). This checks how much the real (nonlinear) response
deviates from that at this magnitude, and reports the actual loss.

Run:
    cd AgentSearch
    python3 fp2_nominal_region_verify.py
"""
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn

TUNE_CHANGES = [['Q6', 'B_SC', -0.02445], ['Q2', 'B_SC', -0.00849],
                 ['Q5', 'B_SC', 0.00117], ['WF1 Exit', 'dB', 0.00352]]
NRAYS = 3000
COSY_FILE = "pg_June2025_nomregion_verify"

if __name__ == "__main__":
    M12, M16, per_group, usedFoxFile = compute_M12_M16(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=COSY_FILE,
        ta_y_offset=0.0, verbose=True)

    loss = loss_fn(M12, M16)

    width = compute_fp2_width(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=usedFoxFile,
        run_cosy_flag=False, verbose=True)

    print(f"\nMinimum-norm near-nominal solution "
          f"(Q6-2.445%, Q2-0.849%, Q5+0.117%, WF1+0.352%):")
    print(f"  M12_sim={M12:.4f}  M16_sim={M16:.4f}  (linear prediction: -0.2472, -0.3842)")
    print(f"  loss (v2, M12+M16-only) = {loss:.3f}")
    print(f"  fp2_slits_std={width['fp2_std']:.3f} mm   vd1638_std={width['vd1638_std']:.3f} mm")
    print(f"\nFor comparison: far-off BO v3 best (Q6+0.46%,Q2-7.70%,Q5+9.68%,WF1=0)  loss=4.575")
    print(f"                 user's full E2-best (Q6-2.7%,WF1+0.25%,TA_Y+0.1mm)     loss=0.524")
