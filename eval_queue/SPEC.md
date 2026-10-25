# Evaluation queue spec: secar-reality → DigitalTwin (Jacobian-loop campaigns)

**Status: v0.4 (2026-10-24), approved by Fernando 2026-10-24.** v0.4 makes DigitalTwin's evaluator the one the worker
runs (§3.5); everything below §3.5 that names queue_eval now means the evaluator named in spec.json.
v0.2 approved by Fernando 2026-10-23 (sha1 6615c868…); v0.3 records
DigitalTwin's allocation points g1–g4 and Fernando's choices at g2 and g3 (given to DigitalTwin directly).
v0.1 was reviewed point by point by DigitalTwin
(`DigitalTwin/exports/secar-reality_evalqueue-spec-review_2026-10-23/review_SPEC_v0.1.md`); v0.2 adds its
points a–f, as accepted by secar-reality, and Fernando's choice at the scheduling fork (§3.3).
Changelog at the end.
Author: secar-reality (`/Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality/`).
Template: opticsFinder's `eval_queue/PROTOCOL.md` v0.3.1. Worker: DigitalTwin's `eval_worker.py`
(`DigitalTwin/docs/kb/simulation/eval-queue-worker.md`).

Lattice in the examples: `pg_June2025_corrected.fox`, sha1 `fca3af5780afba3e60dbbdea52f891cde18b45ca`
(identical in secar-reality/COSY10.2 and DigitalTwin/COSY10.2, checked 2026-10-23). Units and sign
conventions are given per parameter (§3.1) and per result column (§3.4). ECAT readout convention:
X [mm] = -x·1000.

## 1. Purpose

secar-reality fits physically meaningful parameters (magnet scale factors, dipole/WF exit kicks,
misalignments, WF1 geometry) to E2 data by an iterated linear method:

1. linearize: a central-difference Jacobian of the E2 observables around a base point (real simulations);
2. propose: ridge-regularized linear solves within each parameter's trust range give candidates;
3. judge: every candidate is simulated; its chi-square (loss) is computed from the **real** observables;
4. re-linearize at the best real candidate when the linear prediction failed, and repeat.

Steps 1 and 3 are batches of parameter sets: that is what the queue carries. Steps 2 and 4, the loss and
all decisions stay in secar-reality's driver script. No agent sits in the per-candidate path.
Fernando approves a campaign once (2026-10-23): its spec.json, including the loop rules of §6, is the
approved object; he starts the driver and the worker. Anything not in the approved spec.json needs a new
campaign.

## 2. What is reused unchanged from PROTOCOL.md v0.3.1

Folder mechanics (§2), atomic writes and partial lines, spec_id definition, pinned code copied to a frozen
folder and checked by sha1, path override, no hidden defaults, lattice taken from DigitalTwin's
`COSY10.2/` by sha1, engine-outcome statuses (`ok`, `cosy_failed:<n>maps`, `cosy_timeout`, `error: …`),
stopping/restarting/CANCEL (§5), never reusing a campaign name, maps on request (§6), status.json (§7),
cleanup (§8, deletions only with Fernando's agreement).

Requester outbox (written only by secar-reality):
`/Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality/eval_queue/outbox/<campaign>/`.
Worker outbox: `DigitalTwin/eval_queue/outbox/<campaign>/`, shared with opticsFinder's campaigns.
Campaign names start with `sr_` and must be unique across requesters; the ack rejects a name that already
exists in the worker outbox from another inbox.

**Path override (point a).** As in PROTOCOL.md, each pool process imports `config`, repoints its
directories to DigitalTwin's, and only then imports anything else, so `misalign_blocks` (which reads
`<COSY_directory>/<fox_file>.fox` at import) reads the sha1-checked deck. The worker also sets
`config.base`, `config.fox_file` and `config.matrix_file` to the stem of the deck it selected by sha1.
(`config.py` still runs `os.makedirs` on secar-reality's paths at import: a no-op on the laptop.)

## 3. Differences from PROTOCOL.md (the four points opticsFinder listed)

### 3.1 `parameters` instead of `fields` (point 1)

spec.json replaces `fields` with a list of parameters. Each value is an offset from the nominal deck
(0 = nominal), in the native unit of the frozen `helper_functions.prepareTuneChanges`, which also defines
how it is applied:

```json
"parameters": [
  {"name": "Q9_B_SC",    "tune_change": ["Q9", "B_SC"],    "unit": "fraction (0.01 = +1%)",       "applied": "deck",    "step": 0.01,    "trust": [-0.15, 0.15]},
  {"name": "B5_Exit_dB", "tune_change": ["B5 Exit", "dB"], "unit": "fraction (0.0025 = +0.25%)",  "applied": "tracing", "step": 0.0025,  "trust": [-0.01, 0.01]},
  {"name": "Q9_X",       "tune_change": ["Q9", "X"],       "unit": "m",                           "applied": "deck",    "step": 0.00025, "trust": [-0.003, 0.003]},
  {"name": "WF2_Yaw",    "tune_change": ["WF2", "Yaw"],    "unit": "deg",                         "applied": "deck",    "step": 0.05,    "trust": [-0.1, 0.1]}
]
```

(Values above illustrate the format; steps are those of `fp3_knob_ranking.KNOBS`. Trust ranges are a
physics choice per campaign, set by secar-reality and approved by Fernando with the campaign.)

- Kinds in use: `B_SC` (quad scale), `dB` (dipole/WF exit kick, applied during tracing, no compile),
  `X`, `Y` (m), `Roll`, `Yaw`, `Pitch` (deg; whole-block rewrite by `misalign_blocks.py`, incl. WF2's
  commented-out block), WF1 `length`, `rhoE`, `rhoB` (m). Which parameters a campaign may vary, and what
  their values mean physically, is decided by secar-reality, not by the worker.
- req_NNNNNN.csv columns: `cand_id`, then every parameter name in spec order, values in native units,
  full float precision. Each line is a complete, absolute parameter set (not a delta from another line),
  so any candidate can be re-evaluated alone.
- **Every parameter on every line, zeros included (f6).** A zero-valued parameter still rewrites its deck
  line (and a WF2 misalignment parameter activates the WF2 block with zeros, physically the identity).
  queue_eval therefore always passes every campaign parameter, so the base, its ± points and all
  candidates of a campaign go through the same deck-editing path.
- **Deck precision (f3).** queue_eval writes every deck value with enough digits that the value parsed
  back from the edited line matches the requested one to ≤ 1e-3·step. Today WF1 `rhoE`/`rhoB` are written
  with `.3f` (1 mm) by `helper_functions.prepareCOSYchanges`; queue_eval writes them with more digits.
- **`upstream`** (parameters fixed for the whole campaign, e.g. the FP2 solution an FP3 campaign starts
  from) uses the same object format as `parameters` plus a `value`; values are offsets, like
  `parameters`. A name present in both is rejected at ack. Upstream `deck` entries are compiled into
  every deck (and appear in `deck_edits`); upstream `tracing` entries (dB, e.g. an FP2 solution's
  WF1/B1–B4 exit kicks) are applied in every trace (v0.3.1). The worker passes only campaign parameters in
  `deck_values`/`tracing_values`; queue_eval adds the upstream entries itself.

**Ack probe (point e).** For each parameter (and each `upstream` entry), the worker:
1. checks that `applied` matches prepareTuneChanges (`dB` → `dipoleWFchanges` = `tracing`; everything else
   → `params` = `deck`);
2. builds the deck edits at `+step`, `-step` and `0`, applies them to the sha1-checked deck in memory, and
   checks that the deck text changes, that +step ≠ −step, and that every edited line is the expected
   statement for that element (`<elem>_SC :=`, `SA`/`RA`/`TA`, `WC WF1R`, `DL`);
3. checks rounding: the value parsed back from each edited line matches the requested one to ≤ 1e-3·step;
4. applies the same checks at both trust bounds, and rejects a bound that gives an unphysical deck
   (e.g. a negative drift from WF1 `length`).

These are text edits, no COSY run. Any failure gives `accepted: false` with the reasons.

### 3.2 Evaluator: observables in, loss out of the worker (point 2)

The worker returns **raw observables**, never a loss. The loss (FP2 4-term, FP3 Mahalanobis;
`docs/kb/measurements/datasets.md`), its targets, cuts and every decision stay in the driver. This keeps
the locked code to the simulation-and-readout part and lets the loss change without a new worker spec.

- **Entry points (point c):** `queue_eval.init(settings, run_dir)` once per pool process;
  `queue_eval.evaluate_group(deck_values, [(cand_id, tracing_values), …], settings) → [record, …]`:
  one compile, then one trace per candidate; `queue_eval.deck_edits(deck_values) → [[line, text], …]`:
  exactly the edits that will be compiled (upstream included, at queue_eval's precision), for the ack
  probe (§3.1 e). queue_eval is a new module in secar-reality, written after
  this spec is approved. Its readout is `fp3_eval.observables_from_maps` (validated 2026-10-04: the FP2
  OAT screen reproduced to 9e-15). It does **not** use `fp3_eval.compile_maps` (see below).
- **Private compiles (f1, f2, f4).** `fp3_eval.compile_maps` compiles under a name that depends only on the
  deck edits (`helper_functions.py:209`), in the shared `COSY10.2/`, `Matrix_Files/` and `pkl_Files/`,
  and reuses a cached base pkl. queue_eval instead:
  - compiles every deck, including the unedited base, from the sha1-checked deck text in a private
    `mkdtemp` folder under `run_dir` (cosy/cosy.bin linked, edited deck written there, COSY run there,
    fort files read, folder deleted). There is no cached base map;
  - runs COSY with `subprocess.run(..., timeout=cosy_timeout_s)` → `cosy_timeout`, checks the exit status,
    and checks the map count (8 per element up to the last plane read out) → `cosy_failed:<n>maps`.
- **Pinned files (point b):** the full local import closure, paths relative to `secar-reality/`, sha1s
  filled per campaign: `queue_eval.py`, `fp3_eval.py`, `fp2_metrics.py`, `fp3_metrics.py`, `fp3_loss.py`
  (imported by `fp3_eval`, never called by the worker), `helper_functions.py`, `misalign_blocks.py`,
  `fast_tracer.py`, `ViewerAnalysis.py` (groupsInfo), `config.py`, `ray_functions.cpython-312-darwin.so`;
  geometry `pkl_Files/SECAR_Beamline.pkl`. The worker imports only from its frozen copy, so an unpinned
  local import makes the ack fail. Site packages (`ViewerAnalysis` imports `emcee` and `corner`) must
  exist in the worker's `secar` environment; the ack probe reports where each module was loaded from.
- **Beam:** E2 groups, per-group aperture-type distributions from
  `fp2_metrics.build_initial_distribution` (seed 42, identical rays every call), `nrays_per_group` in
  spec.json (3000 in the fits). A ray counts if it reaches the plane and its X at `FP1 Slits` is inside the
  group's FP1 slit window (`groupsInfo`).
- **Planes:** `"planes": ["FP2", "FP3"]` (VD1638, `FP3 Slits & VD1783`). The group sets per plane are those
  of `fp2_metrics` / `fp3_metrics` (`ALL_GROUPS` = dE groups ∪ dalphaX groups) and are fixed by the pinned
  code. The same group label can carry different dE or dalphaX values at FP2 and FP3; the fits use each
  plane's own values.
- **Optional deck cut (FP3 campaigns).** A campaign whose last plane is FP3 may compile the deck cut
  after `FP3 Slits & VD1783` (as opticsFinder's `stage_cut_text`). It changes nothing upstream of the
  cut; spec.json states `"deck_cut": "after FP3"` or `"none"`, and §8.1 is run with the same setting.

### 3.3 Several requesters (point 3)

- One worker process, with a repeatable `--inbox`. It dispatches campaigns by `protocol_version`: `0.3*` →
  opticsFinder's evaluate path; `sr-*` → queue_eval's group path.
- **Scheduling (Fernando, 2026-10-23: option 4, split cores, with a fluid allocation).** Each requester
  has its own queue of campaigns (FIFO by spec.json `created`, one at a time per requester), so a Jacobian
  loop is never queued behind an opticsFinder campaign. The cores are shared as follows:
  - **Total:** `--workers` (default 10, as now).
  - **One requester active:** it gets all of them.
  - **The agent asks Fernando (Fernando, 2026-10-23).** Before giving Fernando the command that starts
    a campaign, the requesting agent reads the worker's status.json. If another requester has work
    queued or running, the agent asks Fernando in chat for the split. It writes his answer as
    `<own outbox>/<campaign>/ALLOCATION`: one line, e.g. `opticsFinder=7 secar-reality=3`, followed by a
    comment line with the date and "given by Fernando in chat". Each agent writes only in its own outbox.
  - **A second requester's campaign arrives while the first is running:**
    - with a valid ALLOCATION in its folder, the worker starts it with its share. The sum may not exceed
      `--workers`.
    - the running campaign shrinks at candidate boundaries: in-flight candidates finish, none is killed.
  - **Fallback, no valid ALLOCATION** (nobody asked, or the file is malformed):
    - the worker does not start the new campaign and sets its status to `awaiting_allocation`
      (status.json, worker.log);
    - it sends a macOS notification (`osascript -e 'display notification …'`) so the wait is not
      silent;
    - the running campaign keeps all its workers. Fernando then answers through the requesting agent,
      which writes the file.
  - **Active campaign (g2, Fernando: option a).** A requester's campaign is active from its first request
    until the requester writes `DONE` in its campaign folder, or until no new request has arrived for
    15 min. A Jacobian loop between rounds (driver solving) therefore stays active and keeps its share;
    the driver writes `DONE` when the loop stops for any reason (§6.5).
  - **Unused shares (g3, Fernando: option a).** While an active requester has no work in flight (e.g. its
    driver is solving), its share is lent to the other requester and taken back at candidate boundaries
    (≤ one evaluation, ~17 s) when its work arrives. The hand-back above is the special case where one
    requester is done.
  - **Preset:** an optional `--allocation opticsFinder=7,secar-reality=3` on the worker command line
    applies to every overlap and skips the question.
- **Pools (g1, DigitalTwin's implementation).** One pool cannot serve both requesters: each pool
  process imports one campaign's frozen code, and the module names (`config`, `helper_functions`, …)
  collide between requesters. The worker runs one pool per requester's active campaign, each created
  with up to `--workers` processes, and a per-requester limit on candidates in flight equal to its share.
  Changing the split changes only the limits (no pool restart); shrinking takes effect at candidate
  boundaries. Idle processes of a shrunk pool keep their memory (~0.2–0.3 GB each) until the pool is
  retired.
- **Details (g4):**
  - the requester name comes from the inbox path (`…/Agents/<name>/eval_queue/outbox`), or explicitly as
    `--inbox name=path`; the current opticsFinder command keeps working unchanged;
  - ALLOCATION keys must be exactly the active requesters' names, each ≥ 1, sum ≤ `--workers`;
  - if both active campaigns have an ALLOCATION file, the one in the more recently started campaign
    applies, and the worker logs which;
  - the fallback notification is sent once when a campaign starts waiting and repeated every 30 min;
  - status.json gains `allocation` and per-requester `in_progress`; `LOCK` lists the current split.
- One `LOCK` (Fernando's rule for heavy parallel jobs) exists while any work is running.
- The split changes the worker that opticsFinder uses. DigitalTwin tells opticsFinder; opticsFinder's
  PROTOCOL.md is not changed by this spec.

### 3.4 Result format (point 4)

All results are per-candidate scalars, so res_NNNNNN.csv suffices (no new format). spec.json lists the
columns explicitly (**`quantities`, f7**); the ack probe checks that list against each plane's
`ALL_GROUPS` in the pinned code, and nothing undeclared appears in the res files. Columns: `cand_id`,
`status`, `eval_s`, then per plane P in `planes`:

- `P_M12, P_Offset12, P_M16, P_Offset16`: M12 in mm/mrad, M16 in mm/%, offsets in mm, ECAT X sign;
- per group g of that plane, `P_meanx_g<g>` (mm) and `P_n_g<g>` (rays counted).

The per-group means let the driver recompute the fits and check them; the counts make §8 test 1 a
ray-count check. Group labels are digit strings, so the names are CSV-safe.
**No-ray error (f8):** a group with no counted ray gives the status
`error: no rays for group <g> at <P>`, raised with that text by queue_eval (not fp3_eval's message).

### 3.5 v0.4: DigitalTwin's evaluator (Fernando, 2026-10-24: simulation code is DigitalTwin's; secar-reality's is legacy)

- `protocol_version` = `"sr-0.4"`; new top-level field `"evaluator": "dt_eval"`.
- The worker runs `DigitalTwin/dt_eval.py` (same API as queue_eval: `init`, `deck_edits`, `evaluate_group`,
  `quantities`), which edits the deck with `DigitalTwin/deck_knobs.py` and reads out with secar-reality's
  pinned `fp3_eval.observables_from_maps` (unchanged readout). Documentation:
  `DigitalTwin/docs/kb/simulation/deck-knobs.md`.
- `implementation` pins secar-reality's readout files by plain name (`fp3_eval.py`, `fp2_metrics.py`,
  `fp3_metrics.py`, `fp3_loss.py`, `helper_functions.py`, `misalign_blocks.py`, `fast_tracer.py`,
  `ViewerAnalysis.py`, `config.py`, `ray_functions…so`) plus `"DigitalTwin/dt_eval.py"` and
  `"DigitalTwin/deck_knobs.py"`. `queue_eval.py` is no longer pinned. The worker resolves `<agent>/<file>` in that
  agent's folder and rejects basename collisions or an unpinned evaluator.
- Parameter kinds (deck_knobs, 12 decimals): `B_SC`; `X`, `Y` (m), `Roll`, `Yaw`, `Pitch` (deg) on any element with a
  TA/RA/SA block, as whole blocks around the deck's measured values with exit = −entry (incl. Q1–Q7, B1–B4,
  WF1; WF2's commented block uncommented with nominal 0); WF1 `length`, `rhoE`, `rhoB` together on the one WC
  line (WF1R replaced by its value 7.0; length also sets DL18, DL19 to nominal − δ/2 each, as the legacy knob);
  `dB` (tracing, via prepareTuneChanges). Every parameter on every line, as before. The §3.1 rule that two deck
  parameters may not edit the same line no longer applies to WF1's geometry.
- `plan.stencil_only: true` (driver only): one stencil, J saved, then DONE; used for rankings.

## 4. spec.json (secar-reality), example skeleton

```json
{
  "protocol_version": "sr-0.3",
  "spec_id": "<first 12 hex of sha1 of canonical JSON without spec_id, as PROTOCOL.md>",
  "campaign": "sr_fp3_jac29_a",
  "created": "...", "produced_by": "secar-reality, secar-reality/jac_loop.py",
  "lattice": {"fox": "pg_June2025_corrected", "fox_sha1": "fca3af5780afba3e60dbbdea52f891cde18b45ca",
              "misalign": 1, "wien_filters": "<as in the deck; checked at ack>"},
  "upstream": [ {"name": "...", "tune_change": [...], "unit": "...", "applied": "...", "value": 0.0} ],
  "engine": "default",
  "implementation": {"queue_eval.py": "<sha1>", "...": "..."},
  "geometry_sha1": "<sha1>",
  "evaluation": {"planes": ["FP3"], "nrays_per_group": 3000, "cosy_timeout_s": 300, "deck_cut": "none"},
  "parameters": [ ... §3.1 ... ],
  "quantities": ["FP3_M12", "FP3_Offset12", "FP3_M16", "FP3_Offset16", "FP3_meanx_g3", "FP3_n_g3", "..."],
  "plan": { ... §6 ... },
  "keep_maps": "only on request"
}
```

The `plan` block is not read by the worker; it is in the spec so that Fernando's approval and the
spec_id cover the loop rules too.

## 5. Requests in a Jacobian loop

| Request kind | Lines | Notes |
|---|---|---|
| stencil, round r | base x_r, and x_r ± step_k for each parameter k | 2K+1 lines; the base and its ±dB lines share one compile (worker groups by deck values) |
| candidates, round r | M ridge solutions around x_r | each line absolute; M set in `plan` |

`cand_id` = `r<round>s<NNN>` (stencil) or `r<round>c<NNN>` (candidates), unique in the campaign.
**Repeated points (f5):** the worker reuses a result only for a repeated `cand_id`. The driver
therefore does not send a point equal in value to one already evaluated in the campaign (e.g. the next
stencil base when it is the best candidate of the previous round); it reuses its own earlier result.

## 6. Loop rules (`plan` block; driver side; values set per campaign)

Proposed defaults, for Fernando to set or change when he approves a campaign:

1. **Round 0:** stencil at the start point x_0 (nominal, or a given earlier solution).
2. **Solve:** whiten J by the loss metric H (loss = rᵀHr); ridge solve for the residual r_0 and for M−1
   residuals drawn from the E2 fit uncertainties (as fp2_family_mc); every parameter clipped to its
   trust range. Default M = 20.
3. **Judge:** real loss of each candidate from the returned observables; linearity ratio
   ρ = (real loss drop)/(predicted loss drop) for each.
4. **Next base:** the candidate with the lowest real loss, if it improves on x_r.
   - Re-linearize (new stencil at the new base) if the best candidate's ρ is outside [0.7, 1.3].
   - Otherwise reuse J_r and only re-solve at the new base (saves 2K+1 evaluations).
5. **Stop** at the first of:
   - **success:** at least `n_family` (default 10) distinct candidates with loss ≤ 1-sigma acceptance
     (4.7196 for 4 dof);
   - **rounds:** `max_rounds` (default 5) rounds done;
   - **no progress:** best real loss improved by less than 10% in two consecutive rounds;
   - **physics stop:** the best candidate sits at a trust-range bound on any parameter. A parameter pushed
     to its bound is a symptom of a model error, not a solution: stop and report, do not widen the range;
   - **budget:** `max_evaluations` (default 600) evaluations sent.

   Whatever the reason, the driver then writes `DONE` in its campaign folder (§3.3, g2), after the last
   result has been read.
6. **Report:** each round is logged as it finishes (base, J singular values, ρ, best real loss), and the
   campaign is recorded in `docs/kb/runs.md`, failures included.

## 7. Approvals and responsibilities

- Fernando: approves this spec (each version) and each campaign's spec.json once; starts the driver and
  the worker; answers allocation requests (§3.3); agrees to deletions.
- secar-reality: parameters, trust ranges, steps, loss, loop rules, interpretation; queue_eval and the
  driver.
- DigitalTwin: the worker (per-requester queues, allocation, dispatch, ack probe), lattice versions, engine agreement; reports
  what it finds in the pinned code.

## 8. Acceptance tests before production (Fernando starts them)

1. **Engine agreement.** Reference values computed locally by secar-reality with the same spec.json and
   the same pinned queue_eval (private compiles, no cached base map):
   - FP3: the base and ± points of 3 FP2 candidates for one parameter of each kind
     (`Q9_B_SC`, `B5_Exit_dB`, `Q9_X`, `WF2_Yaw`, `Q9_Roll`). WF2_Yaw exercises the commented-out WF2
     block;
   - FP2: the base point of the same 3 candidates;
   - with the campaign's `deck_cut` setting (and once with `"after FP3"` if that option is to be used);
   - required: per-group ray counts identical; per-group mean X and the four observables to 1e-12 mm.
   - Cross-check against `Results/fp3_knob_ranking_results.csv` (2026-10-04, fp3_knob_ranking.py through
     fp3_eval: cached base map, shared compile folders, 3000 rays/group, seed 42, same lattice): expected to
     agree to ~1e-12, since the queue path changes only where the maps are compiled. A difference there,
     with the agreement above intact, points at the old path, not the queue, and is reported, not treated
     as a failure of this test.
2. **Small campaign:** one round, K = 3 parameters (one dB), M = 5, on the secar-reality pool; stop and
   restart the driver mid-request and the worker mid-request. Every candidate must be answered exactly
   once, and the dB lines must share their base compile. Run once while an opticsFinder test campaign is
   running, to check:
   - with ALLOCATION in the campaign folder: the campaign starts with its share, and the running one
     shrinks without losing a candidate;
   - without it: the `awaiting_allocation` status and the notification; writing the file then starts
     the campaign;
   - that the remaining requester gets all workers back when the other finishes.
3. **Production:** a real campaign, approved and started by Fernando.

## Changelog

- v0.4 (2026-10-24): DigitalTwin's evaluator dt_eval + deck_knobs (dte-1 fc209a36…, dk-1 138a0a1d…), validated by
  DigitalTwin (text-level vs legacy in 43 single-knob cases; a 69-knob campaign's base equals its E2 hall run
  exactly). Fields `evaluator`, `<agent>/<file>` pins; new kinds (blocks on any element, WF1 geometry together);
  driver `stencil_only`. The driver's FP2 loss is now the FP2 Mahalanobis loss (datasets.md, 2026-10-24).

- v0.3.1 (2026-10-23): upstream `tracing` entries (dB) are applied in every trace (DigitalTwin's point B;
  queue_eval d095d451…). Clarification only; no change to the worker.

- v0.3 (2026-10-23): allocation design from DigitalTwin's points g1–g4:
  - g1: one pool per active requester, per-requester in-flight limits;
  - g2 (Fernando: a): a campaign is active until `DONE` or 15 min without a new request; the driver writes
    `DONE`;
  - g3 (Fernando: a): unused shares are lent and taken back at candidate boundaries;
  - g4: requester names, ALLOCATION validity and precedence, notification repeat, status/LOCK fields.

  Also: `queue_eval.deck_edits` entry point for the ack probe (DigitalTwin's request).

- v0.2 (2026-10-23): DigitalTwin's review points accepted by secar-reality:
  - a: the worker also overrides `config.base`, `fox_file`, `matrix_file`;
  - b: the full import closure is pinned, `fp3_loss.py` included;
  - c: the worker groups candidates by deck values; entry points `queue_eval.init` and
    `queue_eval.evaluate_group`;
  - e: ack probe at ±step, 0 and both trust bounds, with line and rounding checks;
  - f1/f2/f4: private compiles in mkdtemp, always from the sha1-checked deck, with COSY timeout, exit
    status and map count;
  - f3: deck values written to ≤ 1e-3·step;
  - f5: the driver dedups repeated points by value;
  - f6: every parameter on every line;
  - f7: explicit `quantities`;
  - f8: the no-ray error text is queue_eval's own;
  - `upstream` format; `sr_` campaign prefix and cross-inbox name check; optional deck cut after FP3;
  - provenance of the §8.1 cross-check file.

  Scheduling fork, Fernando: option 4 (split cores), fluid:
  - default 10 workers to a single requester;
  - on overlap, the requesting agent asks Fernando in chat before the start and writes ALLOCATION in its
    own campaign folder; worker notification only as a fallback;
  - automatic hand-back of all workers;
  - optional preset.
- v0.1 (2026-10-23): first draft by secar-reality, for DigitalTwin's point-by-point answers and Fernando's
  review.
