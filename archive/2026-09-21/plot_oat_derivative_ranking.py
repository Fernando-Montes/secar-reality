"""
Corrected knob ranking (2026-08-12): the original OAT ranking (plot_oat_screen.py)
ranked knobs by the BEST loss achieved on their +-range grid -- which,
we discovered, badly under-ranked Q3 (dM16/dQ3 ~ -41.5, the steepest
knob of all) purely because its true minimum is narrower than the
grid's spacing, so every sampled point overshot it (see
plot_q3_oat_loss.py). That failure mode affects any knob whose response
is steep relative to its own tested grid density, not just Q3.

This ranks knobs instead by the LOCAL DERIVATIVE dLoss/d(knob) evaluated
AT NOMINAL -- a fit across all of a knob's grid points (robust to grid
coarseness, since it's fitting the underlying M12/M16 line, not
searching among discrete loss values):
    dLoss/dknob = 2*(M12_nom-M12_target)/sigma12^2 * dM12/dknob
                + 2*(M16_nom-M16_target)/sigma16^2 * dM16/dknob
via the chain rule on loss = ((M12-target)/sigma)^2 + ((M16-target)/sigma)^2,
with dM12/dknob and dM16/dknob each an OLS fit across the knob's own OAT
grid points. This uses the SAME 2-term (M12/M16-only) loss the OAT
screen itself was built on.

Three subplots, one per knob's natural unit (matching how the OAT screen
itself defined each knob's range): B_SC/dB knobs -> dLoss/d(%), dX/dY/
WF1_length -> dLoss/d(mm), Roll -> dLoss/d(deg). Sorted by |derivative|
descending within each panel.

Run:
    cd AgentSearch
    python3 plot_oat_derivative_ranking.py
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

rows = list(csv.DictReader(open(RESULTS_FILE)))
nominal = [r for r in rows if r["knob_id"] == "NOMINAL"][0]
M12_nom, M16_nom = float(nominal["M12_sim"]), float(nominal["M16_sim"])
dLoss_dM12 = 2 * (M12_nom - M12_TARGET) / SIGMA_M12 ** 2
dLoss_dM16 = 2 * (M16_nom - M16_TARGET) / SIGMA_M16 ** 2
print(f"Nominal: M12={M12_nom:.4f} M16={M16_nom:.4f}  dLoss/dM12={dLoss_dM12:.2f}  dLoss/dM16={dLoss_dM16:.2f}\n")

knob_ids = sorted(set(r["knob_id"] for r in rows if r["knob_id"] != "NOMINAL"))


def unit_group(knob_id):
    if knob_id.endswith("_B_SC") or knob_id.endswith("_Exit_dB"):
        return "%", 0.01  # tuneChanges value is a fraction; dLoss/d(fraction) -> dLoss/d(percent) = /100
    if knob_id.endswith("_Roll"):
        return "deg", 1.0  # OAT screen's Roll values are already in degrees
    if knob_id.endswith("_dX") or knob_id.endswith("_dY") or knob_id == "WF1_length":
        return "mm", 0.001  # tuneChanges value is in meters; dLoss/d(m) -> dLoss/d(mm) = /1000
    return None, None


groups = {"%": [], "mm": [], "deg": []}
for knob in knob_ids:
    unit, scale = unit_group(knob)
    if unit is None:
        print(f"  skipping {knob}: unrecognized unit")
        continue
    pts = [r for r in rows if r["knob_id"] == knob]
    if len(pts) < 3:
        continue
    vals = np.array([float(p["value"]) for p in pts])
    m12s = np.array([float(p["M12_sim"]) for p in pts])
    m16s = np.array([float(p["M16_sim"]) for p in pts])
    dM12_dknob = np.polyfit(vals, m12s, 1)[0]
    dM16_dknob = np.polyfit(vals, m16s, 1)[0]
    dLoss_dknob_native = dLoss_dM12 * dM12_dknob + dLoss_dM16 * dM16_dknob
    dLoss_dunit = dLoss_dknob_native * scale
    groups[unit].append((knob, dLoss_dunit))

fig, axes = plt.subplots(3, 1, figsize=(11, 16))
for ax, unit in zip(axes, ["%", "mm", "deg"]):
    items = sorted(groups[unit], key=lambda kv: -abs(kv[1]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["tab:red" if v > 0 else "tab:blue" for v in values]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"dLoss / d({unit})  (at nominal)")
    ax.set_title(f"Knobs with native unit '{unit}' -- sorted by |derivative|")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Loss sensitivity ranking (2-term M12+M16 loss), by LOCAL DERIVATIVE at nominal\n"
              "-- robust to grid coarseness, unlike the original best-loss-on-grid ranking", fontsize=12)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "oat_derivative_ranking.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
