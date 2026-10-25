"""
Same as fp2_user_best_loss.py, but isolating just the Q6 -2.7% change from
the user's E2-best tune (WF1 Exit dB and TA_Y left at nominal/0) -- to see
how much of that tune's loss=0.524 comes from Q6 alone (a knob the BO v2
search already covers) vs. from WF1 Exit dB / TA_Y (knobs outside the
BO's 3-knob Q6/Q2/Q5 family).

Run:
    cd AgentSearch
    python3 fp2_user_best_q6only_loss.py
"""
from fp2_metrics import compute_M12_M16, compute_fp2_width
from fp2_bo_search_v2 import loss_fn

TUNE_CHANGES = [['Q6', 'B_SC', -0.027]]
NRAYS = 3000
COSY_FILE = "pg_June2025_userbest_q6only_loss"

if __name__ == "__main__":
    M12, M16, per_group, usedFoxFile = compute_M12_M16(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=COSY_FILE,
        ta_y_offset=0.0, verbose=True)

    loss = loss_fn(M12, M16)

    width = compute_fp2_width(
        tuneChanges=TUNE_CHANGES, nRays=NRAYS, cosyFile=usedFoxFile,
        run_cosy_flag=False, verbose=True)

    print(f"\nQ6 -2.7% only (WF1 Exit dB and TA_Y left at nominal):")
    print(f"  M12_sim={M12:.4f}  M16_sim={M16:.4f}")
    print(f"  loss (v2, M12+M16-only) = {loss:.3f}")
    print(f"  fp2_slits_std={width['fp2_std']:.3f} mm   vd1638_std={width['vd1638_std']:.3f} mm")
    print(f"\nFor comparison: full E2-best (Q6-2.7%, WF1+0.25%, TA_Y+0.1mm) loss=0.524")
    print(f"                 BO v2 best (Q6+0.46%, Q2-7.70%, Q5+9.68%)     loss=4.575")
