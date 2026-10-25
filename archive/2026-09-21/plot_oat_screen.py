"""
Reads fp2_oat_screen_results.csv (partial or complete) and produces:
  1. oat_ranking.csv        -- one row per knob, sorted by best (lowest)
                                loss achieved anywhere in its tested range,
                                alongside the loss range spanned (a knob
                                that never beats nominal but still spans a
                                huge range may just need a wider range).
  2. oat_ranking_bar.png    -- bar chart of best loss per knob, sorted,
                                with the nominal baseline loss marked.
  3. oat_top_knobs.png      -- small-multiples grid: loss vs. knob value,
                                one panel per knob, for the N_TOP most
                                promising knobs.
Safe to run on a partial results file. Run with:
    python3 plot_oat_screen.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FILE = "Results/fp2_oat_screen_results.csv"  # moved here 2026-08-08, OAT screen complete
OUTPUT_DIR = "Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_TOP = 12


def load_results():
    rows = []
    with open(RESULTS_FILE, "r", newline="") as f:
        for row in csv.DictReader(f):
            row["value"] = float(row["value"])
            for k in ["M12_sim", "M16_sim", "loss", "fp2_slits_std", "vd1638_std",
                      "fp2_slits_mean", "vd1638_mean", "elapsed_s"]:
                row[k] = float(row[k])
            rows.append(row)
    return rows


if __name__ == "__main__":
    rows = load_results()
    print(f"Loaded {len(rows)} completed points from {RESULTS_FILE}")
    if not rows:
        raise SystemExit("No results yet.")

    nominal_rows = [r for r in rows if r["knob_id"] == "NOMINAL"]
    if not nominal_rows:
        raise SystemExit("Nominal baseline not in results file yet -- wait for it to be computed first.")
    nominal_loss = nominal_rows[0]["loss"]
    print(f"Nominal baseline loss: {nominal_loss:.1f}")

    by_knob = {}
    for r in rows:
        if r["knob_id"] == "NOMINAL":
            continue
        by_knob.setdefault(r["knob_id"], []).append(r)

    summary = []
    for knob_id, knob_rows in by_knob.items():
        losses = [r["loss"] for r in knob_rows]
        best_row = min(knob_rows, key=lambda r: r["loss"])
        summary.append({
            "knob_id": knob_id,
            "n_points": len(knob_rows),
            "best_loss": best_row["loss"],
            "best_value": best_row["value"],
            "worst_loss": max(losses),
            "loss_range": max(losses) - min(losses),
            "improves_on_nominal": best_row["loss"] < nominal_loss,
        })
    summary.sort(key=lambda s: s["best_loss"])

    ranking_csv = os.path.join(OUTPUT_DIR, "oat_ranking.csv")
    with open(ranking_csv, "w", newline="") as f:
        fieldnames = ["knob_id", "n_points", "best_loss", "best_value",
                      "worst_loss", "loss_range", "improves_on_nominal"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary:
            writer.writerow(s)
    print(f"wrote {ranking_csv} ({len(summary)} knobs)")

    print("\nTop 15 knobs by best achievable loss:")
    print(f"{'knob':<16}{'n':>4}{'best_loss':>12}{'best_value':>14}{'beats nominal':>15}")
    for s in summary[:15]:
        print(f"{s['knob_id']:<16}{s['n_points']:>4}{s['best_loss']:>12.1f}"
              f"{s['best_value']:>14.5f}{str(s['improves_on_nominal']):>15}")

    # --- ranking bar chart ---
    fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(summary))))
    knob_ids = [s["knob_id"] for s in summary]
    best_losses = [s["best_loss"] for s in summary]
    colors = ["tab:green" if s["improves_on_nominal"] else "tab:gray" for s in summary]
    y_pos = np.arange(len(knob_ids))
    ax.barh(y_pos, best_losses, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(knob_ids, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(nominal_loss, color="red", linestyle="--", label=f"nominal loss = {nominal_loss:.1f}")
    ax.set_xlabel("best (lowest) loss achieved across the knob's tested range")
    ax.set_title("OAT screen: knobs ranked by best achievable loss\n(green = beats nominal, gray = does not, within tested range)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    bar_png = os.path.join(OUTPUT_DIR, "oat_ranking_bar.png")
    fig.savefig(bar_png, dpi=150)
    print(f"wrote {bar_png}")

    # --- small multiples for top N knobs ---
    top = summary[:N_TOP]
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, s in enumerate(top):
        ax = axes[i // ncols][i % ncols]
        knob_rows = sorted(by_knob[s["knob_id"]], key=lambda r: r["value"])
        vals = [r["value"] for r in knob_rows]
        losses = [r["loss"] for r in knob_rows]
        ax.plot(vals, losses, "o-", markersize=4)
        ax.axhline(nominal_loss, color="red", linestyle="--", linewidth=1)
        ax.set_title(s["knob_id"], fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(top), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Top {len(top)} knobs: loss vs. knob value (red dashed = nominal loss)")
    fig.tight_layout()
    top_png = os.path.join(OUTPUT_DIR, "oat_top_knobs.png")
    fig.savefig(top_png, dpi=150)
    print(f"wrote {top_png}")
