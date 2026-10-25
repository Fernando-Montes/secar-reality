"""
Bar chart of z_crossing_sensitivity_summary.py's ranking: %/knob needed
to shift the M16 crossing and the M12 crossing each by +0.28m
(downstream), per knob. Filtered (2026-08-12, per user's request) to
ONLY the knobs that can do BOTH within +-5%, shown SIGNED (not absolute
value) on a linear y-axis fixed to [-5%, 5%] -- so both which knobs
matter and which direction (increase vs decrease) is visible directly.

Run:
    cd AgentSearch
    python3 plot_z_crossing_sensitivity.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/z_crossing_sensitivity_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SHIFT_TARGET = 0.28
MAX_REALISTIC_PCT = 5.0

rows = list(csv.DictReader(open(RESULTS_FILE)))
nominal = [r for r in rows if r["knob_id"] == "NOMINAL"][0]
nom_m16 = float(nominal["m16_crossing_z"])
nom_m12 = float(nominal["m12_crossing_z"])

knob_ids = sorted(set(r["knob_id"] for r in rows if r["knob_id"] != "NOMINAL"))
results = []
for knob in knob_ids:
    pts = [r for r in rows if r["knob_id"] == knob]
    m16_pts = [(float(p["value"]), float(p["m16_crossing_z"])) for p in pts if p["m16_crossing_z"] != ""]
    m12_pts = [(float(p["value"]), float(p["m12_crossing_z"])) for p in pts if p["m12_crossing_z"] != ""]

    req16 = req12 = None
    if len(m16_pts) >= 2:
        vals = [0.0] + [v for v, _ in m16_pts]
        zs = [nom_m16] + [z for _, z in m16_pts]
        slope = np.polyfit(vals, zs, 1)[0]
        req16 = (SHIFT_TARGET / slope) * 100 if slope != 0 else np.inf
    if len(m12_pts) >= 2:
        vals = [0.0] + [v for v, _ in m12_pts]
        zs = [nom_m12] + [z for _, z in m12_pts]
        slope = np.polyfit(vals, zs, 1)[0]
        req12 = (SHIFT_TARGET / slope) * 100 if slope != 0 else np.inf

    worst = max(abs(req16) if req16 is not None else np.inf, abs(req12) if req12 is not None else np.inf)
    results.append({"knob_id": knob, "req16": req16, "req12": req12, "sort_key": worst})

def realistic(pct):
    return pct is not None and abs(pct) <= MAX_REALISTIC_PCT

results = [r for r in results if realistic(r["req16"]) and realistic(r["req12"])]
results.sort(key=lambda r: r["sort_key"])
print(f"{len(results)} knobs can shift both crossings within +-{MAX_REALISTIC_PCT:.0f}%: "
      f"{[r['knob_id'] for r in results]}")

knobs = [r["knob_id"] for r in results]
x = np.arange(len(knobs))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 7))
r16 = [r["req16"] for r in results]
r12 = [r["req12"] for r in results]
ax.bar(x - width / 2, r16, width, label="M16 crossing: required %", color="tab:blue")
ax.bar(x + width / 2, r12, width, label="M12 crossing: required %", color="tab:orange")
ax.axhline(0, color="black", linewidth=1)
ax.set_ylim(-MAX_REALISTIC_PCT, MAX_REALISTIC_PCT)
ax.set_xticks(x)
ax.set_xticklabels(knobs, rotation=45, ha="right")
ax.set_ylabel("required %/knob to shift crossing +0.28m (signed)")
ax.set_title(f"Knobs that can shift BOTH the M16 and M12 crossings +0.28m within +-{MAX_REALISTIC_PCT:.0f}%\n"
              "(positive = increase knob, negative = decrease knob)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
outfile = os.path.join(OUTPUT_DIR, "z_crossing_sensitivity_bar.png")
fig.savefig(outfile, dpi=150)
print(f"wrote {outfile}")
