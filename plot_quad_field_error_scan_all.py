"""
Plots quad_field_error_scan_all.py's results as one page per quad (5
panels each, same layout as plot_q6_field_error_scan.py), combined into a
single multi-page PDF -- per user request 2026-09-21. Y-axis limits for
each panel are shared across ALL 15 quads (computed from the full
dataset, with padding), so the quads can be visually compared directly
page to page; the X-axis (field-error delta, +-1%) is already common by
construction.

Run:
    cd AgentSearch
    python3 plot_quad_field_error_scan_all.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESULTS_FILE = "Results/quad_field_error_scan_all_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPTIMIZED_BASE = {"Q14": -0.20, "Q15": -0.20}
QUADS = [f"Q{i}" for i in range(1, 16)]

PANELS = [
    ("transmission_pct", "Transmission\ntarget->DSSD [%]", "tab:blue"),
    ("fp2_resolution", "FP2 resolution\n|ME(1,7)|/WV", "tab:orange"),
    ("fp3_resolution", "FP3 resolution\n|ME(1,7)|/WV", "tab:green"),
    ("dssd_mean_x_50cm_mm", "DSSD mean X [mm]\n(50x50cm)", "tab:red"),
    ("dssd_range_x_50cm_mm", "DSSD X range [mm]\n(50x50cm)", "tab:purple"),
]

rows = list(csv.DictReader(open(RESULTS_FILE)))
by_quad = {q: sorted([r for r in rows if r["quad"] == q], key=lambda r: float(r["delta"]))
           for q in QUADS}

missing = [q for q in QUADS if len(by_quad[q]) < len(set(float(r["delta"]) for r in rows))]
if missing:
    print(f"NOTE: these quads have incomplete data (fewer points than the full delta grid): {missing}")
present_quads = [q for q in QUADS if by_quad[q]]

# shared y-limits per panel, computed across every quad's data, with 5% padding
ylims = {}
for key, _, _ in PANELS:
    vals = np.array([float(r[key]) for r in rows])
    lo, hi = vals.min(), vals.max()
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ylims[key] = (lo - pad, hi + pad)

outfile = os.path.join(OUTPUT_DIR, "quad_field_error_scan_all.pdf")
with PdfPages(outfile) as pdf:
    for quad in present_quads:
        quad_rows = by_quad[quad]
        base_pct = OPTIMIZED_BASE.get(quad, 0.0) * 100
        delta_pct = np.array([float(r["delta"]) * 100 for r in quad_rows])

        fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.5, 3 * len(PANELS)), sharex=True)
        for ax, (key, ylabel, color) in zip(axes, PANELS):
            y = np.array([float(r[key]) for r in quad_rows])
            ax.plot(delta_pct, y, "o-", color=color, markersize=4)
            ax.axvline(0, color="grey", linestyle=":", linewidth=1)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(*ylims[key])
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel(f"{quad}_B_SC field error [%] (relative to its operating point, "
                             f"{base_pct:+.0f}%)")
        fig.suptitle(f"{quad} field-error scan (base Q14=-20%, Q15=-20%; +-1%, 0.1% steps)\n"
                     f"mean/width at DSSD assuming 50x50cm aperture", fontsize=12)
        fig.tight_layout(h_pad=1.5)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"added page for {quad} ({len(quad_rows)} points)")

print(f"\nwrote {outfile} ({len(present_quads)} pages)")
