"""
Quad sensitivity ranking (2026-09-21), from quad_field_error_scan_all.py's
results: for transmission, FP2 resolution, FP3 resolution, and DSSD X
position independently, rank the 15 quads by how sensitive that metric is
to a +-1% field error in that quad (all other quads held at their
operating point -- Q14=-20%, Q15=-20%, everything else nominal).

DSSD X position uses dssd_mean_x_50cm_mm (the 50cm x 50cm hypothetical
aperture) rather than the real-aperture dssd_mean_x_mm, because the real
aperture's mean X is noisy/non-monotonic (individual rays dropping in and
out at the small real aperture edge -- see the 2026-09-21 discussion),
which would inflate peak-to-peak sensitivity with aperture-edge artifacts
rather than genuine optical sensitivity. Transmission uses the real
aperture (transmission_pct), unchanged.

Two sensitivity measures per quad per metric, per 2026-09-21 discussion:
  - peak-to-peak: max-min of the metric across the 21 points (+-1%,
    0.1% steps). PRIMARY ranking criterion -- the direct "worst-case
    change if this quad drifts within tolerance" number, robust to
    non-monotonic/asymmetric response shapes (several metrics here peak
    or dip near nominal rather than responding linearly).
  - slope at nominal: central difference using the +-0.1% points
    immediately adjacent to delta=0, i.e. (f(+0.1%)-f(-0.1%))/0.2%,
    units of [metric]/[% field error]. SECONDARY/comparison criterion,
    the more standard local-derivative sensitivity coefficient (matches
    this project's earlier OAT-derivative-ranking style), but assumes a
    near-linear response and can rank a strongly peaked-at-zero quad low
    even when its peak-to-peak effect is large.

Quads are ranked by |peak-to-peak| descending; each quad's slope-based
rank position is reported alongside so large disagreements (peaked/
asymmetric responses) are visible.

Run:
    cd AgentSearch
    python3 rank_quad_field_error_sensitivity.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/quad_field_error_scan_all_results.csv"
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUADS = [f"Q{i}" for i in range(1, 16)]

METRICS = [
    ("transmission_pct", "Transmission target->DSSD [%]", "transmission"),
    ("fp2_resolution", "FP2 resolution |ME(1,7)|/WV", "fp2_resolution"),
    ("fp3_resolution", "FP3 resolution |ME(1,7)|/WV", "fp3_resolution"),
    ("dssd_mean_x_50cm_mm", "DSSD X position [mm] (50x50cm)", "dssd_x_position"),
]

rows = list(csv.DictReader(open(RESULTS_FILE)))
by_quad = {q: sorted([r for r in rows if r["quad"] == q], key=lambda r: float(r["delta"]))
           for q in QUADS}

summary_rows = []

for key, label, slug in METRICS:
    entries = []
    for quad in QUADS:
        quad_rows = by_quad[quad]
        deltas = np.array([float(r["delta"]) * 100 for r in quad_rows])  # in %
        vals = np.array([float(r[key]) for r in quad_rows])

        peak_to_peak = vals.max() - vals.min()

        # nearest points to delta=0 on each side (grid is 0.1% steps, so
        # this is +-0.1% unless a point is missing)
        pos_idx = np.argmin(np.where(deltas > 0, deltas, np.inf))
        neg_idx = np.argmin(np.where(deltas < 0, -deltas, np.inf))
        d_pos, d_neg = deltas[pos_idx], deltas[neg_idx]
        slope = (vals[pos_idx] - vals[neg_idx]) / (d_pos - d_neg)

        entries.append({"quad": quad, "peak_to_peak": peak_to_peak, "slope": slope})

    entries.sort(key=lambda e: e["peak_to_peak"], reverse=True)
    for rank, e in enumerate(entries, start=1):
        e["rank_ptp"] = rank

    slope_order = sorted(entries, key=lambda e: abs(e["slope"]), reverse=True)
    for rank, e in enumerate(slope_order, start=1):
        e["rank_slope"] = rank

    for e in entries:
        e["metric"] = slug
        e["rank_diff"] = e["rank_slope"] - e["rank_ptp"]
        summary_rows.append(e)

    # bar chart, sorted by peak-to-peak (primary ranking), annotated with
    # slope-based rank for comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    quads_sorted = [e["quad"] for e in entries]
    ptp_sorted = [e["peak_to_peak"] for e in entries]
    y_pos = np.arange(len(quads_sorted))[::-1]
    ax.barh(y_pos, ptp_sorted, color="tab:blue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(quads_sorted)
    for y, e in zip(y_pos, entries):
        ax.text(e["peak_to_peak"], y, f"  slope rank #{e['rank_slope']}", va="center", fontsize=8, color="grey")
    ax.set_xlabel(f"Peak-to-peak change over +-1% field error\n({label})")
    ax.set_title(f"Quad sensitivity ranking: {label}\n(primary: peak-to-peak; annotation: rank by slope at nominal)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, f"quad_ranking_{slug}.png")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}")
    print(f"  top 3 by peak-to-peak: {[(e['quad'], round(e['peak_to_peak'], 3)) for e in entries[:3]]}")

summary_file = os.path.join(OUTPUT_DIR, "quad_ranking_summary.csv")
with open(summary_file, "w", newline="") as f:
    fieldnames = ["metric", "quad", "peak_to_peak", "slope", "rank_ptp", "rank_slope", "rank_diff"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in summary_rows:
        writer.writerow({k: r[k] for k in fieldnames})
print(f"\nwrote {summary_file}")
