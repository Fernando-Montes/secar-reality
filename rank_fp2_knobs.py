"""
FP2 knob ranking at the E2 Hall-settings base (2026-10-24), from DigitalTwin's
eval-queue campaign sr_fp2_rank69 (69 knobs, central differences; SPEC v0.4,
dt_eval). Analysis only; no simulation.

Input: eval_queue/runs/sr_fp2_rank69/jacobian.npz (J [obs x knob] in native units,
obs0, FP2 Mahalanobis H, target) and the worker's res_000000.csv (for curvature).
Equal units (as the FP3 ranking): 1% for B_SC/dB, 1 mm for X/Y, 0.5 deg roll,
0.1 deg yaw/pitch; WF1 rhoE/rhoB/length: 1% of their value (70 mm, 70 mm, 23.65 mm).
Per knob, per equal unit: change of M12 (mm/mrad), Offset12 (mm), M16 (mm/%),
Offset16 (mm); dLoss = 2 r^T H J u (negative = moves toward the data); and the
curvature ratio |obs+ + obs- - 2 obs0| / |obs+ - obs-| (max over observables).

Run: python3 rank_fp2_knobs.py   Output: Results/fp2_rank69.csv, Results/fp2_rank69.png
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "eval_queue/runs/sr_fp2_rank69/jacobian.npz"
RES = ("/Users/fernandomontes/Research/COSY/Simulations/Agents/DigitalTwin/eval_queue/outbox/"
       "sr_fp2_rank69/res_000000.csv")
OBS = ["M12", "Offset12", "M16", "Offset16"]


def unit(name):
    if name.endswith(("_B_SC", "_dB")):
        return 0.01
    if name.endswith(("_X", "_Y")):
        return 0.001
    if name.endswith("_Roll"):
        return 0.5
    if name.endswith(("_Yaw", "_Pitch")):
        return 0.1
    if name.endswith(("_rhoE", "_rhoB")):
        return 0.07
    if name.endswith("_length"):
        return 0.02365
    raise ValueError(name)


def main():
    d = np.load(RUN, allow_pickle=True)
    J, obs0, H, target = d["J"], d["obs0"], d["H"], d["target"]
    names = [str(n) for n in d["names"]]
    r = obs0 - target
    rows = {}
    for row in csv.DictReader(l for l in open(RES) if not l.startswith("#")):
        rows[row["cand_id"]] = np.array([float(row[f"FP2_{o}"]) for o in OBS])
    out = []
    for k, n in enumerate(names):
        u = unit(n)
        dobs = J[:, k] * u
        p, m = rows[f"r0s{2 * k + 1:03d}"], rows[f"r0s{2 * k + 2:03d}"]
        curv = float(np.max(np.abs(p + m - 2 * obs0) / np.maximum(np.abs(p - m), 1e-15)))
        out.append(dict(knob=n, unit=u, dM12=dobs[0], dOff12=dobs[1], dM16=dobs[2], dOff16=dobs[3],
                        dLoss=float(2 * r @ H @ dobs), curvature=curv))
    out.sort(key=lambda o: o["dLoss"])
    with open("Results/fp2_rank69.csv", "w") as f:
        f.write("# FP2 knob ranking at the E2 Hall base (sr_fp2_rank69, DigitalTwin dt_eval, 2026-10-24); per equal unit "
                "(1% B_SC/dB, 1 mm X/Y, 0.5 deg roll, 0.1 deg yaw/pitch, 1% WF1 geometry); dLoss = 2 r^T H J u, FP2 Mahalanobis\n")
        w = csv.DictWriter(f, list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print(f"base obs {obs0.round(4)}  target {target.round(4)}  loss {float(r @ H @ r):.1f}")
    print(f"{'knob':14s} {'dM12':>8s} {'dOff12':>8s} {'dM16':>8s} {'dOff16':>8s} {'dLoss':>9s} {'curv':>5s}")
    for o in out[:15] + [None] + sorted(out, key=lambda o: -abs(o["dM12"]))[:12]:
        if o is None:
            print("--- largest |dM12| per unit ---")
            continue
        print(f"{o['knob']:14s} {o['dM12']:+8.4f} {o['dOff12']:+8.3f} {o['dM16']:+8.4f} {o['dOff16']:+8.3f} "
              f"{o['dLoss']:+9.1f} {o['curvature']:5.2f}")

    top = sorted(out, key=lambda o: -abs(o["dLoss"]))[:25][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
    y = range(len(top))
    for ax, key, lab, tgt in [(axes[0], "dLoss", "dLoss per unit (negative = toward the data)", None),
                              (axes[1], "dM12", "dM12 per unit [mm/mrad]  (needed: -0.30)", -0.30),
                              (axes[2], "dM16", "dM16 per unit [mm/%]  (needed: -0.29)", -0.29)]:
        v = [o[key] for o in top]
        ax.barh(y, v, color=["#2ca02c" if (x < 0) else "#d62728" for x in v], height=0.7)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel(lab)
        ax.grid(axis="x", ls="--", alpha=0.5)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([o["knob"] for o in top])
    fig.suptitle("FP2 knob ranking at the E2 Hall-probe base (25 largest |dLoss|), per equal unit: 1% B_SC/dB, "
                 "1 mm X/Y, 0.5° roll, 0.1° yaw/pitch, 1% WF1 geometry | simulations by DigitalTwin", fontsize=11)
    fig.tight_layout()
    fig.savefig("Results/fp2_rank69.png", dpi=120)
    print("Results/fp2_rank69.png")


if __name__ == "__main__":
    main()
