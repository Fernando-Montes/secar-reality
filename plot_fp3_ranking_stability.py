"""
How much does the FP3 knob ranking depend on which FP2 solution is chosen?
(2026-10-04, from Results/fp3_knob_ranking_results.csv.)

For every finished candidate, dLoss/dknob is computed the same way as in
plot_fp3_knob_ranking.py (chain rule on the four observables at the
candidate's base point), and knobs are ranked by |dLoss/dknob| WITHIN each
unit group (%, mm, deg) -- ranks across units are not comparable. 1 = most
sensitive in its group.

Output, Results/fp3_ranking_stability.pdf, one set of pages per unit group:
  1. rank heatmap: knobs (rows, ordered by median rank) x candidates
     (columns); the rank each knob gets at each candidate;
  2. rank spread: each knob's rank across candidates (box = 25-75%,
     whiskers = full range);
  3. candidate-vs-candidate Spearman correlation of the rankings (1 = same
     order, 0 = unrelated).
Also prints, per unit group: median pairwise Spearman, and how often each
knob lands in that group's top 5. Per-candidate derivatives and ranks are
written to Results/fp3_ranking_per_candidate.csv.

Safe to run while fp3_knob_ranking.py is still going (only candidates with
every knob complete are used).

Run:
    cd AgentSearch
    python3 plot_fp3_ranking_stability.py
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fp3_loss
from fp3_knob_ranking import KNOBS

RESULTS_FILE = "Results/fp3_knob_ranking_results.csv"
OUT_PDF = "Results/fp3_ranking_stability.pdf"
OUT_CSV = "Results/fp3_ranking_per_candidate.csv"
UNITS = ['%', 'mm', 'deg']
TOP_K = 5

pts = {}
for r in csv.DictReader(open(RESULTS_FILE)):
    pts[(r["source"], int(r["iteration"]), r["knob"], int(r["sign"]))] = np.array(
        [float(r[k]) for k in ("M12", "Offset12", "M16", "Offset16")])

cands = []
deriv = {}                                           # (cand, knob) -> dLoss per display unit
for c in sorted({(s, i) for s, i, k, _ in pts if k == 'base'}):
    obs0 = pts[(c[0], c[1], 'base', 0)]
    vals = {}
    for k, (_, step, _, factor) in KNOBS.items():
        p, m = pts.get((c[0], c[1], k, 1)), pts.get((c[0], c[1], k, -1))
        if p is None or m is None:
            break
        vals[k] = fp3_loss.loss_gradient(obs0, (p - m) / (2 * step)) / factor
    else:
        cands.append(c)
        for k, v in vals.items():
            deriv[(c, k)] = v
if len(cands) < 2:
    raise SystemExit(f"{len(cands)} complete candidate(s); need at least 2 to compare rankings.")
labels = [f"{s}:{i}" for s, i in cands]
print(f"{len(cands)} complete candidates.\n")

rank = {}                                            # (cand, knob) -> rank within its unit group
for u in UNITS:
    ks = [k for k in KNOBS if KNOBS[k][2] == u]
    for c in cands:
        order = sorted(ks, key=lambda k: -abs(deriv[(c, k)]))
        for r_, k in enumerate(order, start=1):
            rank[(c, k)] = r_

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "iteration", "knob", "unit", "dLoss_per_unit", "rank_in_unit"])
    for c in cands:
        for k in KNOBS:
            w.writerow([c[0], c[1], k, KNOBS[k][2], deriv[(c, k)], rank[(c, k)]])


def spearman_matrix(R):
    """R: (n_knobs, n_cands) ranks -> (n_cands, n_cands) Spearman (Pearson on ranks; no ties)."""
    Z = (R - R.mean(axis=0)) / R.std(axis=0)
    return (Z.T @ Z) / R.shape[0]


with PdfPages(OUT_PDF) as pdf:
    for u in UNITS:
        ks = [k for k in KNOBS if KNOBS[k][2] == u]
        R = np.array([[rank[(c, k)] for c in cands] for k in ks], dtype=float)
        order = np.argsort(np.median(R, axis=1), kind="stable")
        ks_o, R_o = [ks[i] for i in order], R[order]
        S = spearman_matrix(R)
        iu = np.triu_indices(len(cands), 1)
        top = (R_o <= TOP_K).mean(axis=1)

        print(f"=== unit '{u}' ({len(ks)} knobs) ===")
        print(f"  pairwise Spearman between candidates: median {np.median(S[iu]):.3f}, "
              f"min {S[iu].min():.3f}, 10th pct {np.percentile(S[iu], 10):.3f}")
        print(f"  fraction of candidates where the knob is in the top {TOP_K} of this group:")
        for k, t, row in zip(ks_o, top, R_o):
            print(f"    {k:<16} {100*t:5.0f}%   rank median {np.median(row):4.1f}, range {int(row.min())}-{int(row.max())}")
        print()

        # 1. rank heatmap
        fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(cands) + 3), 0.35 * len(ks) + 2))
        im = ax.imshow(R_o, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(ks))
        ax.set_yticks(range(len(ks_o)))
        ax.set_yticklabels(ks_o, fontsize=8)
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("FP2 candidate")
        ax.set_title(f"Unit '{u}': rank of each knob at each FP2 candidate (1 = most sensitive)\n"
                     f"rows ordered by median rank")
        fig.colorbar(im, ax=ax, label="rank within group")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2. rank spread
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(ks) + 2))
        ax.boxplot(R_o[::-1].T, vert=False, whis=(0, 100), widths=0.6)
        ax.set_yticks(range(1, len(ks_o) + 1))
        ax.set_yticklabels([f"{k}  (top-{TOP_K}: {100*t:.0f}%)" for k, t in zip(ks_o[::-1], top[::-1])], fontsize=8)
        ax.set_xlabel("rank within group across FP2 candidates (box 25-75%, whiskers full range)")
        ax.set_xlim(0.5, len(ks) + 0.5)
        ax.invert_xaxis()
        ax.set_title(f"Unit '{u}': how much each knob's rank moves between FP2 candidates")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 3. candidate x candidate Spearman
        fig, ax = plt.subplots(figsize=(max(7, 0.2 * len(cands) + 3), max(6, 0.2 * len(cands) + 2)))
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(cands)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"Unit '{u}': Spearman correlation of the knob ranking\nbetween candidates "
                     f"(median {np.median(S[iu]):.2f}, min {S[iu].min():.2f})", fontsize=10)
        fig.colorbar(im, ax=ax, label="Spearman rank correlation")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"wrote {OUT_PDF} and {OUT_CSV}")
