"""
Jacobian-loop driver for secar-reality campaigns on DigitalTwin's eval queue
(eval_queue/SPEC.md v0.3, 2026-10-23). Started by Fernando; one approval per
campaign covers every round (SPEC §1, §6).

A campaign is defined by a config file written by secar-reality and approved
by Fernando (eval_queue/campaigns/<campaign>.json): lattice, upstream,
evaluation, parameters (SPEC §3.1) and plan (SPEC §6). The driver writes
spec.json from it (pinned sha1s and quantities filled in), then loops:

  round r: stencil at x_r (base and x_r +- step_k)  -> J_r, obs(x_r), loss L_r
           (skipped when J is reused, SPEC §6.4)
           M candidates: weighted ridge solves        -> real losses
           best candidate: rho = (L_r - L_real)/(L_r - L_pred)
           next base = best candidate if L_real < L_r; re-linearize there unless
           rho is inside plan.rho_range (then J_r is reused)
  stop: success (n_family candidates with loss <= accept_loss), max_rounds,
        no progress (best loss improved < min_progress in 2 consecutive rounds),
        physics stop (best candidate at a trust bound), budget (max_evaluations),
        or a failed stencil point. Then DONE is written (SPEC §3.3 g2).

Solve (as archive/2026-09-21/fp2_family_mc.py, generalized): with W the
whitening of the loss metric (loss = |W (obs - target)|^2) and
Gamma = diag(1/trust_k^2) a Gaussian prior on each parameter's offset from
nominal, the step d from x_r for a target t minimizes
    |W (J d + obs_r - t)|^2 + (x_r + d)^T Gamma (x_r + d),
then x = clip(x_r + d, trust). Candidate 0 uses the measured target; the
others use targets drawn from N(target, plan.sample_cov[plane]) (fit
uncertainties of the E2 data; chosen per campaign), seeded per round.

Losses (docs/kb/measurements/datasets.md, [VERIFIED]); observable order
everywhere [M12, Offset12, M16, Offset16]:
  FP2: 4-term, targets -0.2472, 1.543 mm, -0.3842, 0.537 mm; sigmas 0.02, 0.2, 0.03, 1.5
  FP3: fp3_loss (Mahalanobis on M12/Offset12, covariance /4; M16, Offset16 sigmas 0.2, 1.5)
plan.loss_planes lists the planes whose losses are summed.

Restart-safe: every request is a deterministic function of the results
before it, so a restarted driver replays the campaign from the files: it
checks that each existing req file matches what it would write, and only
waits for missing results. Points equal in value to one already evaluated are
not sent again (SPEC f5). Round log: eval_queue/runs/<campaign>/rounds.json.

Units: parameter values in native tuneChanges units (fractions for B_SC/dB,
m for X/Y/length/rhoE/rhoB, deg for Roll/Yaw/Pitch), offsets from nominal.

Run (Fernando; the worker must be running, with an ALLOCATION agreed if
another requester is active):
    cd /Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality
    /Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3 jac_loop.py eval_queue/campaigns/<campaign>.json
Test of the loop logic without COSY or worker (synthetic observables):
    ... jac_loop.py <config> --mock
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# SR_* environment overrides exist only for testing the file protocol against a fake worker
OUTBOX = os.environ.get("SR_OUTBOX", os.path.join(HERE, "eval_queue", "outbox"))
RUNS = os.environ.get("SR_RUNS", os.path.join(HERE, "eval_queue", "runs"))
WORKER_OUTBOX = os.environ.get("SR_WORKER_OUTBOX",
                               "/Users/fernandomontes/Research/COSY/Simulations/Agents/DigitalTwin/eval_queue/outbox")
POLL_S = float(os.environ.get("SR_POLL_S", 10))
PROTOCOL = "sr-0.4"
EVALUATOR = "dt_eval"
DT = "/Users/fernandomontes/Research/COSY/Simulations/Agents/DigitalTwin"
# readout files in secar-reality (plain names) + DigitalTwin's evaluator (SPEC v0.4)
# dt_eval dte-2 reads out through dt_readout dr-1 (ti > idx: stopped at a plane = not seen), 2026-10-24
PINNED_DT = ["dt_eval.py", "deck_knobs.py", "dt_readout.py"]
PINNED = ["fp2_metrics.py", "fp3_metrics.py",
          "helper_functions.py", "misalign_blocks.py", "fast_tracer.py", "ViewerAnalysis.py", "config.py",
          "ray_functions.cpython-312-darwin.so"]
GEOMETRY = "pkl_Files/SECAR_Beamline.pkl"
OBS = ["M12", "Offset12", "M16", "Offset16"]
HEARTBEAT_WARN_S = 600

FP2_TARGETS = np.array([-0.2472, 1.543, -0.3842, 0.537])
# FP2 Mahalanobis (Fernando, 2026-10-24): sigmas 0.027, 0.36, 0.3, 1.5; off-diagonal covariances from the OLS fits
FP2_COV = np.array([[0.027 ** 2, -0.00090, 0, 0],
                    [-0.00090, 0.36 ** 2, 0, 0],
                    [0, 0, 0.3 ** 2, -0.0084],
                    [0, 0, -0.0084, 1.5 ** 2]])


def sha1_file(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def quantities(planes):
    import fp3_eval
    cols = []
    for p in planes:
        cols += [f"{p}_{o}" for o in OBS]
        for g in fp3_eval.PLANES[p]['groups'].ALL_GROUPS:
            cols += [f"{p}_meanx_g{g}", f"{p}_n_g{g}"]
    return cols


def spec_id(spec):
    s = {k: v for k, v in spec.items() if k != "spec_id"}
    return hashlib.sha1(json.dumps(s, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                        .encode("utf-8")).hexdigest()[:12]


def loss_metric(planes):
    """-> (targets, H) for the stacked observables of the loss planes; loss = r^T H r."""
    import fp3_loss
    t, blocks = [], []
    for p in planes:
        if p == "FP2":
            t.append(FP2_TARGETS)
            blocks.append(np.linalg.inv(FP2_COV))
        elif p == "FP3":
            t.append(fp3_loss.TARGETS)
            blocks.append(fp3_loss.H)
        else:
            raise ValueError(p)
    n = 4 * len(planes)
    H = np.zeros((n, n))
    for i, b in enumerate(blocks):
        H[4 * i:4 * i + 4, 4 * i:4 * i + 4] = b
    return np.concatenate(t), H


# ----------------------------------------------------------------------------- campaign files

def build_spec(cfg):
    spec = {
        "protocol_version": PROTOCOL,
        "campaign": cfg["campaign"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "produced_by": "secar-reality, secar-reality/jac_loop.py",
        "lattice": cfg["lattice"],
        "upstream": cfg.get("upstream", []),
        "engine": "default",
        "evaluator": EVALUATOR,
        "implementation": {**{f: sha1_file(os.path.join(HERE, f)) for f in PINNED},
                           **{f"DigitalTwin/{f}": sha1_file(os.path.join(DT, f)) for f in PINNED_DT}},
        "geometry_sha1": sha1_file(os.path.join(HERE, GEOMETRY)),
        "evaluation": cfg["evaluation"],
        "parameters": cfg["parameters"],
        "quantities": quantities(cfg["evaluation"]["planes"]),
        "plan": cfg["plan"],
        "keep_maps": "only on request",
    }
    spec["spec_id"] = spec_id(spec)
    return spec


def write_atomic(path, text):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


class Queue:
    """Requests through DigitalTwin's worker (SPEC §2, §5)."""

    def __init__(self, spec):
        self.spec = spec
        self.dir = os.path.join(OUTBOX, spec["campaign"])
        self.wdir = os.path.join(WORKER_OUTBOX, spec["campaign"])
        self.names = [p["name"] for p in spec["parameters"]]
        self.n_req = 0

    def start(self):
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, "spec.json")
        if os.path.exists(path):
            old = json.load(open(path))
            mine = dict(self.spec, created=old["created"])
            mine["spec_id"] = spec_id(mine)
            if mine != old:
                sys.exit(f"{path} exists and differs from this config: a changed spec needs a new campaign name")
            self.spec = old
        else:
            write_atomic(path, json.dumps(self.spec, indent=1) + "\n")
        log(f"spec {self.spec['spec_id']} in {self.dir}; waiting for the worker's spec_ack")
        ack = os.path.join(self.wdir, "spec_ack.json")
        while not os.path.exists(ack):
            time.sleep(POLL_S)
        a = json.load(open(ack))
        if a.get("spec_id") != self.spec["spec_id"]:
            sys.exit(f"spec_ack spec_id {a.get('spec_id')} != {self.spec['spec_id']}")
        if not a.get("accepted"):
            sys.exit(f"spec rejected by the worker: {a.get('reasons') or a.get('notes')}")
        log(f"spec accepted: {a.get('engine')}, {a.get('lattice_used')}")

    def evaluate(self, lines):
        """lines = [(cand_id, {name: value})] -> {cand_id: result dict}."""
        req = os.path.join(self.dir, f"req_{self.n_req:06d}.csv")
        body = io.StringIO()
        w = csv.writer(body, lineterminator="\n")
        w.writerow(["cand_id"] + self.names)
        for cid, vals in lines:
            w.writerow([cid] + [repr(float(vals[n])) for n in self.names])
        if os.path.exists(req):
            old = open(req).read()
            if old.split("\n# units", 1)[-1].split("\n", 1)[-1] != body.getvalue():
                sys.exit(f"{req} differs from the replayed request: campaign state is inconsistent")
        else:
            head = (f"# protocol_version: {PROTOCOL}\n# spec_id: {self.spec['spec_id']}\n"
                    f"# campaign: {self.spec['campaign']}\n# request: {self.n_req:06d}\n"
                    f"# created: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n# n_candidates: {len(lines)}\n"
                    f"# units: native tuneChanges units, offsets from nominal (SPEC 3.1)\n")
            write_atomic(req, head + body.getvalue())
        res = os.path.join(self.wdir, f"res_{self.n_req:06d}.csv")
        self.n_req += 1
        want = {cid for cid, _ in lines}
        got, last_beat = {}, time.time()
        while True:
            if os.path.exists(res):
                got = read_results(res)
            if want <= set(got):
                return {c: got[c] for c in want}
            st = os.path.join(self.wdir, "status.json")
            if os.path.exists(st) and time.time() - os.path.getmtime(st) > HEARTBEAT_WARN_S \
                    and time.time() - last_beat > HEARTBEAT_WARN_S:
                log("warning: worker heartbeat older than 10 min; still waiting")
                last_beat = time.time()
            time.sleep(POLL_S)

    def done(self, reason):
        write_atomic(os.path.join(self.dir, "DONE"), f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {reason}\n")


def read_results(path):
    with open(path) as f:
        text = f.read()
    text = text[:text.rfind("\n") + 1]
    rows = [l for l in text.split("\n") if l and not l.startswith("#")]
    out = {}
    for r in csv.DictReader(rows):
        if r["cand_id"] not in out:
            out[r["cand_id"]] = r
    return out


class Mock:
    """Synthetic observables for testing the loop logic (no COSY, no worker)."""

    def __init__(self, spec):
        self.spec = spec
        self.names = [p["name"] for p in spec["parameters"]]
        planes = spec["evaluation"]["planes"]
        rng = np.random.default_rng(1)
        steps = np.array([p["step"] for p in spec["parameters"]])
        targets, _ = loss_metric(planes)
        self.obs0 = targets + rng.normal(0, 1, len(targets)) * np.tile([0.3, 2.0, 0.5, 2.0], len(planes))
        self.A = rng.normal(0, 1, (len(targets), len(steps))) * np.tile([0.1, 1.0, 0.3, 1.0], len(planes))[:, None] / (
            10 * steps)
        self.n_req = 0

    def start(self):
        log("mock evaluator: synthetic observables, nothing is sent to the worker")

    def evaluate(self, lines):
        self.n_req += 1
        out = {}
        for cid, vals in lines:
            x = np.array([vals[n] for n in self.names])
            u = self.A @ x
            obs = self.obs0 + u + 0.05 * u ** 2
            rec = {"cand_id": cid, "status": "ok"}
            for i, p in enumerate(self.spec["evaluation"]["planes"]):
                rec.update({f"{p}_{o}": repr(obs[4 * i + j]) for j, o in enumerate(OBS)})
            out[cid] = rec
        return out

    def done(self, reason):
        log(f"mock: DONE ({reason})")


# ----------------------------------------------------------------------------- loop

_LOG = []


def log(s):
    line = f"{time.strftime('%H:%M:%S')} {s}"
    print(line, flush=True)
    _LOG.append(line)


class Loop:
    def __init__(self, spec, ev):
        self.spec, self.ev = spec, ev
        self.plan = spec["plan"]
        self.par = spec["parameters"]
        self.names = [p["name"] for p in self.par]
        self.steps = np.array([p["step"] for p in self.par], float)
        self.lo = np.array([p["trust"][0] for p in self.par], float)
        self.hi = np.array([p["trust"][1] for p in self.par], float)
        self.prior = np.diag(1 / np.maximum(np.abs(self.lo), np.abs(self.hi)) ** 2)
        self.lplanes = self.plan["loss_planes"]
        self.target, self.H = loss_metric(self.lplanes)
        self.W = np.linalg.cholesky(self.H).T
        self.cache = {}            # value key -> (cand_id, obs or None)
        self.n_sent = 0
        self.rounds = []
        self.family = {}           # cand_id -> loss, candidates with loss <= accept_loss
        self.rundir = os.path.join(RUNS, spec["campaign"])

    def key(self, x):
        return tuple(repr(float(v)) for v in x)

    def obs_of(self, rec):
        if rec["status"] != "ok":
            return None
        return np.array([float(rec[f"{p}_{o}"]) for p in self.lplanes for o in OBS])

    def loss(self, obs):
        if obs is None:
            return np.inf
        r = obs - self.target
        return float(r @ self.H @ r)

    def run_points(self, points):
        """points = [(cand_id, x)] -> [obs or None], reusing equal points (SPEC f5)."""
        new = [(c, x) for c, x in points if self.key(x) not in self.cache]
        seen, lines = set(), []
        for c, x in new:
            if self.key(x) not in seen:
                seen.add(self.key(x))
                lines.append((c, dict(zip(self.names, x))))
        if lines:
            if self.n_sent + len(lines) > self.plan["max_evaluations"]:
                return None
            res = self.ev.evaluate(lines)
            self.n_sent += len(lines)
            for c, vals in lines:
                x = np.array([vals[n] for n in self.names])
                self.cache[self.key(x)] = (c, self.obs_of(res[c]), res[c]["status"])
        return [self.cache[self.key(x)] for _, x in points]

    def stencil(self, r, x):
        pts = [(f"r{r}s000", x.copy())]
        for k in range(len(x)):
            for sgn, tag in ((+1, "p"), (-1, "m")):
                xk = x.copy()
                xk[k] += sgn * self.steps[k]
                pts.append((f"r{r}s{2 * k + (1 if sgn > 0 else 2):03d}", xk))
        out = self.run_points(pts)
        if out is None:
            return None
        if any(o[1] is None for o in out):
            bad = [(pts[i][0], o[2]) for i, o in enumerate(out) if o[1] is None]
            return ("failed", bad)
        obs0 = out[0][1]
        J = np.column_stack([(out[1 + 2 * k][1] - out[2 + 2 * k][1]) / (2 * self.steps[k]) for k in range(len(x))])
        return obs0, J

    def candidates(self, r, x, obs_r, J):
        rng = np.random.default_rng(self.plan["seed"] + r)
        Jw = self.W @ J
        A = Jw.T @ Jw + self.prior
        cov = np.zeros((len(self.target), len(self.target)))
        for i, p in enumerate(self.lplanes):
            cov[4 * i:4 * i + 4, 4 * i:4 * i + 4] = np.array(self.plan["sample_cov"][p])
        pts, pred = [], []
        for m in range(self.plan["M"]):
            t = self.target if m == 0 else rng.multivariate_normal(self.target, cov)
            d = np.linalg.solve(A, -Jw.T @ (self.W @ (obs_r - t)) - self.prior @ x)
            xc = np.clip(x + d, self.lo, self.hi)
            pts.append((f"r{r}c{m:03d}", xc))
            pred.append(self.loss(obs_r + J @ (xc - x)))
        return pts, pred

    def save(self, stop=None):
        os.makedirs(self.rundir, exist_ok=True)
        write_atomic(os.path.join(self.rundir, "rounds.json"),
                     json.dumps({"spec_id": self.spec["spec_id"], "parameters": self.names, "rounds": self.rounds,
                                 "family": self.family, "evaluations_sent": self.n_sent, "stop": stop},
                                indent=1) + "\n")
        write_atomic(os.path.join(self.rundir, "log.txt"), "\n".join(_LOG) + "\n")

    def finish(self, reason):
        log(f"STOP: {reason}")
        self.save(stop=reason)
        self.ev.done(reason)
        return reason

    def run(self):
        start = self.plan.get("start", {})
        x = np.array([float(start.get(n, 0.0)) for n in self.names])
        J, obs_x, slow = None, None, 0
        for r in range(self.plan["max_rounds"]):
            fresh = J is None
            if fresh:
                st = self.stencil(r, x)
                if st is None:
                    return self.finish("budget: max_evaluations reached before the stencil")
                if isinstance(st[0], str):
                    return self.finish(f"stencil point(s) failed: {st[1]}")
                obs_x, J = st
            L_r = self.loss(obs_x)
            if self.plan.get("stencil_only"):
                os.makedirs(self.rundir, exist_ok=True)
                np.savez(os.path.join(self.rundir, "jacobian.npz"), J=J, obs0=obs_x, names=self.names,
                         steps=self.steps, target=self.target, H=self.H)
                self.rounds.append({"round": r, "base": dict(zip(self.names, x.tolist())), "base_loss": L_r,
                                    "base_obs": obs_x.tolist(), "stencil_only": True})
                log(f"stencil only: base loss {L_r:.4g}; J saved in {self.rundir}/jacobian.npz")
                return self.finish("stencil_only: ranking stencil done")
            s = np.linalg.svd(self.W @ J * np.maximum(np.abs(self.lo), np.abs(self.hi)), compute_uv=False)
            pts, pred = self.candidates(r, x, obs_x, J)
            out = self.run_points(pts)
            if out is None:
                return self.finish("budget: max_evaluations reached")
            real = [self.loss(o[1]) for o in out]
            for (c, xc), L in zip(pts, real):
                cid = self.cache[self.key(xc)][0]
                if L <= self.plan["accept_loss"]:
                    self.family[cid] = L
            b = int(np.argmin(real))
            dropped = L_r - pred[b]
            rho = (L_r - real[b]) / dropped if dropped > 0 else float("nan")
            xb = pts[b][1]
            at_bound = [n for n, v, lo, hi in zip(self.names, xb, self.lo, self.hi)
                        if v <= lo + 1e-12 * max(1, abs(lo)) or v >= hi - 1e-12 * max(1, abs(hi))]
            improved = real[b] < L_r
            rec = {"round": r, "base": dict(zip(self.names, x.tolist())), "base_loss": L_r,
                   "J_singular_values_per_trust": s.tolist(), "best": self.cache[self.key(xb)][0],
                   "best_values": dict(zip(self.names, xb.tolist())), "best_loss": real[b],
                   "best_predicted_loss": pred[b], "rho": rho, "n_failed": int(sum(np.isinf(real))),
                   "family_size": len(self.family), "new_jacobian": fresh}
            self.rounds.append(rec)
            log(f"round {r}: base loss {L_r:.4g} -> best {rec['best']} loss {real[b]:.4g} "
                f"(predicted {pred[b]:.4g}, rho {rho:.3g}); family {len(self.family)}; sent {self.n_sent}")
            self.save()
            if len(self.family) >= self.plan["n_family"]:
                return self.finish(f"success: {len(self.family)} candidates with loss <= {self.plan['accept_loss']}")
            if at_bound and improved:
                return self.finish(f"physics stop: best candidate at a trust bound on {at_bound}")
            gain = (L_r - real[b]) / L_r if L_r > 0 else 0.0
            slow = slow + 1 if gain < self.plan["min_progress"] else 0
            if slow >= 2:
                return self.finish("no progress: best loss improved < "
                                   f"{self.plan['min_progress']:.0%} in two consecutive rounds")
            if improved:
                lo_rho, hi_rho = self.plan["rho_range"]
                reuse = lo_rho <= rho <= hi_rho
                x, obs_x = xb, out[b][1]
                if reuse:
                    log(f"round {r}: rho inside {self.plan['rho_range']}: J reused at the new base")
                else:
                    J = None
            # not improved: same base and J; the next round draws new targets
        return self.finish(f"rounds: max_rounds = {self.plan['max_rounds']} done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--mock", action="store_true", help="synthetic observables; nothing sent to the worker")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if not a.mock and not cfg["campaign"].startswith("sr_"):
        sys.exit("campaign names start with sr_ (SPEC 2)")
    spec = build_spec(cfg)
    if a.mock:
        spec = copy.deepcopy(spec)
        spec["campaign"] = "mock_" + spec["campaign"]
    ev = Mock(spec) if a.mock else Queue(spec)
    ev.start()
    if not a.mock:
        spec = ev.spec
    Loop(spec, ev).run()


if __name__ == "__main__":
    main()
