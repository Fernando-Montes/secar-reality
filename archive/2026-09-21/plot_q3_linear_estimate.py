"""
Quick LINEAR estimate (no COSY) of M12 vs Q3 and M16 vs Q3, per user
request 2026-08-14 -- sanity check on the claim that Q3 alone should
flip M12's sign within a small range. Uses the same Jacobian
(dM12/dQ3, dM16/dQ3 at nominal) already built from cached OAT data by
fp2_family_mc_q3q4wf1.py's build_jacobian() -- just a straight-line
extrapolation from nominal, y = y0 + slope*Q3, no new simulation.

Run:
    cd AgentSearch
    python3 plot_q3_linear_estimate.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fp2_family_mc_q3q4wf1 import build_jacobian, KNOBS

M12_TARGET, M16_TARGET = -0.2472, -0.3842

OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

J, nominal_vals = build_jacobian()
M12_0, M16_0, Off12_0, Off16_0 = nominal_vals
q3_idx = KNOBS.index('Q3_B_SC')
dM12_dQ3 = J[0, q3_idx]
dM16_dQ3 = J[1, q3_idx]
print(f"nominal: M12={M12_0:.4f}  M16={M16_0:.4f}")
print(f"dM12/dQ3={dM12_dQ3:.3f}  dM16/dQ3={dM16_dQ3:.3f}  (per unit Q3, i.e. per 100%)")

q3 = np.linspace(-0.10, 0.10, 200)
M12_lin = M12_0 + dM12_dQ3 * q3
M16_lin = M16_0 + dM16_dQ3 * q3

q3_M12_zero = -M12_0 / dM12_dQ3
q3_M16_zero = -M16_0 / dM16_dQ3
print(f"linear M12=0 crossing at Q3={q3_M12_zero*100:.3f}%")
print(f"linear M16=0 crossing at Q3={q3_M16_zero*100:.3f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

ax1.plot(q3 * 100, M12_lin, color="tab:blue")
ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax1.axhline(M12_TARGET, color="black", linestyle="--", label=f"M12 target ({M12_TARGET})")
ax1.axvline(0, color="gray", linewidth=0.8)
ax1.set_ylabel("M12 (linear estimate)")
ax1.set_title("M12 vs Q3_B_SC (linear extrapolation from nominal, NO COSY)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(q3 * 100, M16_lin, color="tab:red")
ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax2.axhline(M16_TARGET, color="black", linestyle="--", label=f"M16 target ({M16_TARGET})")
ax2.axvline(0, color="gray", linewidth=0.8)
ax2.set_xlabel("Q3_B_SC delta [%]")
ax2.set_ylabel("M16 (linear estimate)")
ax2.set_title("M16 vs Q3_B_SC (linear extrapolation from nominal, NO COSY)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q3_linear_estimate_M12_M16.png")
fig.savefig(outfile, dpi=200)
print(f"\nwrote {outfile}")
