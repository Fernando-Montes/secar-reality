"""
Diagnostic (2026-08-12, user's question): why did Q3 rank last in the
original OAT screen but rank near the top in the z-crossing sensitivity
test? Plots the 2-term (M12+M16 only) loss vs Q3_B_SC from the ORIGINAL
OAT screen data (Results/fp2_oat_screen_results.csv, +-10% grid, 10
points) to see the shape directly.

Run:
    cd AgentSearch
    python3 plot_q3_oat_loss.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_oat_screen_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
M12_TARGET, M16_TARGET = -0.2472, -0.3842
SIGMA_M12, SIGMA_M16 = 0.02, 0.03


def loss2(m12, m16):
    return ((m12 - M12_TARGET) / SIGMA_M12) ** 2 + ((m16 - M16_TARGET) / SIGMA_M16) ** 2


rows = list(csv.DictReader(open(RESULTS_FILE)))
q3 = sorted([r for r in rows if r["knob_id"] == "Q3_B_SC"], key=lambda r: float(r["value"]))
nominal = [r for r in rows if r["knob_id"] == "NOMINAL"][0]
nom_loss = loss2(float(nominal["M12_sim"]), float(nominal["M16_sim"]))

vals = np.array([float(r["value"]) for r in q3]) * 100
losses = np.array([loss2(float(r["M12_sim"]), float(r["M16_sim"])) for r in q3])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(vals, losses, "o-", color="tab:red", label="Q3_B_SC (OAT grid, 10 points)")
ax.axhline(nom_loss, color="gray", linestyle=":", label=f"nominal loss ({nom_loss:.0f})")
ax.axvline(0, color="black", linestyle=":", alpha=0.5)
ax.set_xlabel("Q3_B_SC [%]")
ax.set_ylabel("loss (M12+M16 only)")
ax.set_title("Q3_B_SC: 2-term loss vs knob value, from the original OAT screen's own +-10% grid\n"
             "(every sampled point is worse than nominal -- the true minimum is narrower than the grid spacing)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "q3_oat_loss.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")

print(f"\nnominal loss: {nom_loss:.2f}")
for v, l in zip(vals, losses):
    print(f"  Q3={v:+.3f}%  loss={l:.2f}")
