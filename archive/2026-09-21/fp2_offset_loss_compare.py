"""
Compares the 4-term loss (M12, M16, and now Offset12/Offset16 too, per
the user's 2026-08-10 suggestion) between two candidates:
  1. "Manual" -- the user's own hand-tuned E2-best: Q6-2.7%, WF1 Exit
     dB+0.25%, TA_Y+0.1mm. Already known to hit M12/M16 well (loss=0.524
     under the old 2-term loss) -- unknown how well it does on offsets,
     since those were never computed for this tune before.
  2. "MinNorm11" -- the new minimum-norm solution across all 11 B_SC/dB
     candidate knobs (H1,H3,O1,Q1-Q7 B_SC + WF1 Exit dB), solved via
     pseudoinverse of the full 4x11 Jacobian to hit all 4 targets
     (combined norm ~24%, well outside the previously-validated ~2.6%
     linear regime -- this checks how it actually performs against real
     COSY, not just the linear prediction).

Offsets (intercepts) are computed the same way as the Jacobian scan: take
per_group from compute_M12_M16 and fit slope+intercept locally, instead
of just the slope compute_M12_M16 itself returns.

Run:
    cd AgentSearch
    python3 fp2_offset_loss_compare.py
"""
import numpy as np
from fp2_metrics import compute_M12_M16, DALPHAX_GROUPS, DE_GROUPS

M12_TARGET, M16_TARGET = -0.2472, -0.3842
OFFSET12_TARGET, OFFSET16_TARGET = 1.543, 0.537
SIGMA_M12, SIGMA_M16 = 0.02, 0.03
SIGMA_OFFSET12, SIGMA_OFFSET16 = 0.2, 1.5


def loss_fn_extended(M12, M16, Offset12, Offset16):
    t12 = ((M12 - M12_TARGET) / SIGMA_M12) ** 2
    t16 = ((M16 - M16_TARGET) / SIGMA_M16) ** 2
    tOff12 = ((Offset12 - OFFSET12_TARGET) / SIGMA_OFFSET12) ** 2
    tOff16 = ((Offset16 - OFFSET16_TARGET) / SIGMA_OFFSET16) ** 2
    return t12 + t16 + tOff12 + tOff16, (t12, t16, tOff12, tOff16)


def fit_slope_offset(per_group, groups_dict):
    xs = np.array([groups_dict[g] for g in groups_dict])
    ys = np.array([per_group[g][0] for g in groups_dict])
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


CASES = {
    "Manual (Q6-2.7%, WF1+0.25%, TA_Y+0.1mm)": {
        "tuneChanges": [['Q6', 'B_SC', -0.027], ['WF1 Exit', 'dB', 0.0025]],
        "ta_y_offset": 0.1 / 1000,
        "cosyFile": "pg_June2025_offcmp_manual",
    },
    "MinNorm11 (11-knob pseudoinverse solution)": {
        "tuneChanges": [
            ['H1', 'B_SC', -0.081230], ['H3', 'B_SC', -0.017929], ['O1', 'B_SC', 0.002833],
            ['Q1', 'B_SC', -0.072888], ['Q2', 'B_SC', 0.189417], ['Q3', 'B_SC', -0.031746],
            ['Q4', 'B_SC', 0.001610], ['Q5', 'B_SC', 0.009583], ['Q6', 'B_SC', -0.095299],
            ['Q7', 'B_SC', 0.022331], ['WF1 Exit', 'dB', 0.002261],
        ],
        "ta_y_offset": 0.0,
        "cosyFile": "pg_June2025_offcmp_minnorm11",
    },
}
NRAYS = 3000

if __name__ == "__main__":
    for name, cfg in CASES.items():
        M12, M16, per_group, usedFoxFile = compute_M12_M16(
            tuneChanges=cfg["tuneChanges"], nRays=NRAYS, cosyFile=cfg["cosyFile"],
            ta_y_offset=cfg["ta_y_offset"], verbose=False)
        _, Offset12 = fit_slope_offset(per_group, DALPHAX_GROUPS)
        _, Offset16 = fit_slope_offset(per_group, DE_GROUPS)
        loss, (t12, t16, tOff12, tOff16) = loss_fn_extended(M12, M16, Offset12, Offset16)

        print(f"{name}:")
        print(f"  M12={M12:.4f} (target {M12_TARGET})   M16={M16:.4f} (target {M16_TARGET})")
        print(f"  Offset12={Offset12:.4f} (target {OFFSET12_TARGET})   Offset16={Offset16:.4f} (target {OFFSET16_TARGET})")
        print(f"  loss = {loss:.3f}   [t12={t12:.3f} t16={t16:.3f} tOff12={tOff12:.3f} tOff16={tOff16:.3f}]\n")
