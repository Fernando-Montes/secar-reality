# AgentSearch pipeline: how the agent's copy of ECAT works now

`/Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality/`
(until 2026-10-23 `/Users/fernandomontes/Research/COSY/MC/ECAT/AgentSearch/`;
see "Relocation 2026-10-23" below) is the agent's
isolated copy of ECAT (config, helper_functions, ViewerAnalysis,
ray_functions .so, COSY10.2 deck, Matrix_Files, pkl_Files, Results). The
originals in `ECAT/` are never modified. This note records how it differs
from the original and the conventions every script relies on.

### Environment
**[VERIFIED]** Run with `/Users/fernandomontes/opt/anaconda3/envs/SECAR/bin/python3`
(Python 3.12.7). The base `python3` (3.9) lacks shapely and cannot load the
cpython-312 `ray_functions` .so.
Source: session 2026-09-21 (import failure, then success under SECAR env).
Recorded: 2026-10-23

### Active lattice: pg_June2025_corrected
**[VERIFIED]** `AgentSearch/config.py`: `base = "pg_June2025_corrected"` since
2026-10-04. The user found that `pg_June2025.fox` had a spurious WF2
misalignment and wrote the corrected deck. See `cosy/fox-file-notes.md`
(2026-10-04 entry). `createCosyFile` uses `fox_file` as the template for every
tune change, so this switch affects all compiles.
Source: AgentSearch/config.py:18-21; user, 2026-10-04.
Recorded: 2026-10-23

### Bugs fixed in AgentSearch/helper_functions.py (2026-10-04)
**[VERIFIED]** Ported from the Emulator project's known-bugs note ("bug 3"):
`runECAT`'s compile-without-tuneChanges branch loaded `matrix_file` (a
config global) instead of the `foxFile` it had just compiled, and
`save_rays_flag` wrote `{fox_file}_Rays.pkl` instead of `{foxFile}`. Both now
use `foxFile`. `prepareCOSYchanges` now includes `fox_file` in the cache-name
hash, so the same knob change on different base decks never shares a name.
Source: AgentSearch/helper_functions.py (runECAT, prepareCOSYchanges);
Emulator/docs/kb/simulation/known-bugs.md §3.
Recorded: 2026-10-23

### Cache names: runECAT renames the compile
**[VERIFIED]** With any .fox-editing tuneChanges, `prepareCOSYchanges` builds
its own name `cs` + md5 hash and runECAT compiles under that name, ignoring
the `foxFile` passed in; it returns the real name as `usedFoxFile`. Cleanup
must use `usedFoxFile`. Cleaning up by the passed-in tag leaked every compile
(2026-09-21/22: 4331 leaked directories, disk full).
Source: helper_functions.py prepareCOSYchanges/runECAT; mc_quad_mistune_scan.py fix.
Recorded: 2026-10-23

### Ray tracing: fast_tracer
**[VERIFIED]** `runECAT` traces with `AgentSearch/fast_tracer.py` (vectorized,
from Emulator/emulatorAgent, extended here to apply dipole/WF `dB` knobs
exactly like `process_single_ray`). `test_fast_tracer.py`: loss index identical
for every ray and positions identical to ≤2e-14 m on the physical path,
with/without dB knobs (incl. unsorted multi-dB); ~25-50x faster.
Rays keep being propagated after an aperture loss (both tracers); those values
are not physical.
Source: AgentSearch/fast_tracer.py, test_fast_tracer.py (2026-10-04).
Recorded: 2026-10-23

### Misalignment knobs: whole-block edits (misalign_blocks.py)
**[VERIFIED]** For Q8-Q11, B5, B6, WF2, knobs `X`, `Y`, `Roll`, `Yaw`, `Pitch`
are written as one block per element (TA yaw pitch / RA roll / SA x y going in,
reversed going out). This adds yaw/pitch (not in `cosy_lines`) and fixes X and
Y on the same element overwriting each other (both edit the SA lines). Line
numbers and nominal values are read from the deck and checked at import. WF2's
block is commented out in the corrected deck (nominal zero); varying a WF2
misalignment knob un-comments it and writes zeros plus the knob. Units: X, Y in
m; Roll, Yaw, Pitch in degrees.
Also: the old `cosy_lines['WF2']` Roll/dXY exit-line numbers were swapped
(harmless while the block was inactive). Not fixed in cosy_lines; use
misalign_blocks for WF2.
Source: AgentSearch/misalign_blocks.py; prepareCOSYchanges block handling.
Recorded: 2026-10-23

### Single-pass E2 evaluation (fp3_eval.py)
**[VERIFIED]** `fp3_eval.evaluate_variants(lattice_tuneChanges, dB_variants, plane, nrays, loss_fn)`:
one compile per lattice, polygons and the seeded per-group rays built once per
process, all groups traced together, compile files deleted after use.
`plane` = 'FP2' (VD1638, fp2_metrics groups) or 'FP3' (VD1783, fp3_metrics
groups). Reproduces fp2_metrics/fp3_metrics.compute_M12_M16 exactly
(group means to 1e-15). A tuneChanges with no .fox edits uses the cached base
maps (compiled once under the base name).
Ray selection: a ray counts if inside all apertures at every location BEFORE
the readout and inside its group's FP1 slit window (-x*1000 at 'FP1 Slits').
Source: AgentSearch/fp3_eval.py; validation 2026-10-04.
Recorded: 2026-10-23

### Losses
See `measurements/datasets.md` ("E2 loss definitions", 2026-10-23).

### Script conventions (user's standing rules)
**[VERIFIED]** COSY campaigns are run by the user; scripts are checkpointed
(append per result, skip done work on restart), parallel over processes
(10 workers), delete their own compile artifacts. Ray/envelope plots include
chamber walls (`beamline_walls.py`: wall position along x at y=0 / y at x=0
from the beamline geometry polygons; `zoom_to_rays` keeps the beam visible).
Source: user instructions 2026-09-21, 2026-10-19; project memory.
Recorded: 2026-10-23

### First-order matrix elements: method caveat
**[VERIFIED, secondhand]** `compute_me17_cumulative.py` composes per-element
Jacobians evaluated at zero. The Emulator agent showed this is not exact when
maps have constant (misalignment) terms; on its lattice FP3 x-dispersion was
off 18-56%. FP3/DSSD resolution numbers from it are biased; FP2 is less
affected. The Emulator's `first_order_matrix.py` evaluates Jacobians at the
central ray and matches COSY's native ME.
Source: Emulator/docs/kb/runs.md (2026-09-26 entry).
Recorded: 2026-10-23

### Relocation 2026-10-23
**[VERIFIED]** Fernando reorganized the folders on 2026-10-23: `COSY/MC` ->
`COSY/Simulations`, `Emulator` -> `Agents`. This pipeline moved from
`MC/ECAT/AgentSearch/` to `Simulations/Agents/secar-reality/`, and the KB and
sources moved with it (formerly `ECAT/docs/kb`, `ECAT/sources`; now
`secar-reality/docs/kb`, `secar-reality/sources`). `config.ECAT_directory`
was updated to the new folder (checked by importing config with the SECAR
python). In earlier entries, read `AgentSearch/<file>` as
`secar-reality/<file>`, and read `Emulator/docs/kb/...` as
`Simulations/Agents/opticsFinder/docs/kb/...` (the same files were also
copied to `DigitalTwin/docs/kb/`). Source: user, 2026-10-23.

### queue_eval.py: evaluator for the eval queue (2026-10-23)
**[VERIFIED]** `queue_eval.py` is the code DigitalTwin's eval worker runs for secar-reality campaigns
(`eval_queue/SPEC.md` v0.3, approved by Fernando). It compiles every deck in a private mkdtemp folder from the
deck text (no shared Matrix_Files/pkl names, no cached base map), with a COSY timeout, exit-status and
map-count check (74 maps for the full deck). Deck values are written with 12 decimals:
`helper_functions.DECK_FMT`/`WFR_FMT` (defaults `.6f`/`.3f`, unchanged for every other script; 8 knob kinds
checked identical). Readout: `fp3_eval.observables_from_maps`, unchanged.
`init` refuses two deck parameters that edit the same .fox line: WF1 `length` and `rhoE`/`rhoB` all rewrite
line 593, so they cannot be varied in the same deck.
Validation, `test_queue_eval.py`, run by Fernando 2026-10-23 20:53, output in `Results/test_queue_eval.txt`:
base (zeros), Q9_B_SC +1% and WF2_Yaw +0.05 deg equal to fp3_eval's compile path (difference 0, identical
ray counts, FP2 and FP3); base with zero-valued edits vs the cached unedited map 1.1e-16; timeout gives
`cosy_timeout` with no COSY left; no compile folders left. One compile ~25 s on one process.
Source: queue_eval.py (sha1 65fc12de…), test_queue_eval.py; user, 2026-10-23.

### jac_loop.py: Jacobian-loop driver for the eval queue (2026-10-23)
**[VERIFIED]** `jac_loop.py <eval_queue/campaigns/<campaign>.json>` runs one campaign (SPEC v0.3 §5–6): stencil →
central-difference J → M weighted ridge candidates (prior = trust range on the offset from nominal, as
fp2_family_mc; targets sampled from `plan.sample_cov`) → real losses → ρ test → re-linearize or reuse J; stops
on success / max_rounds / no progress / trust bound / budget / failed stencil, then writes DONE. Losses: FP2
4-term and FP3 fp3_loss, summed over `plan.loss_planes`. Restart = deterministic replay from the req/res files;
equal points are not re-sent. Round log: `eval_queue/runs/<campaign>/rounds.json`, `log.txt`.
Tested 2026-10-23 without COSY: `--mock` (synthetic observables) and a fake worker in the scratch folder, with
the driver killed mid-request and restarted: identical requests replayed, every candidate answered once,
same result as the uninterrupted run. Not yet run against DigitalTwin's worker (SPEC §8).
First campaign config: `eval_queue/campaigns/sr_test_k3.json` (§8.2 test; mechanics only, not a fit).

### Eval queue acceptance test 1 (SPEC §8.1, engine agreement): PASSED 2026-10-23
**[VERIFIED]** `accept_8_1.py`: FP2 solutions v1:20, v1:143, v7:10 (upstream, B_SC rounded as in their original
.6f decks; v7:10 includes WF1/B1–B4 dB in tracing), parameters Q9_B_SC, B5_Exit_dB, Q9_X, WF2_Yaw, Q9_Roll at
0 and ±step, planes FP2+FP3, 3000 rays/group. Local reference (Fernando, 21:10, queue_eval d095d451…) vs
DigitalTwin's worker (eval_worker d8f2cad769eb, cosy.bin fa67d2887a10, 21:13): all 33 points identical (difference
0, identical ray counts). Also identical (difference 0) to `Results/fp3_knob_ranking_results.csv` at FP3, i.e.
zero-valued deck edits change nothing and the rounded upstream reproduces the validated solutions. Raw values
checked (independent eval_s, values differ between points). Output: `Results/acc81_compare.txt`. v7:10 FP2 M12
-0.2452 (target -0.2472). Still to do: §8.2 small campaign with stop/restart and an overlapping opticsFinder campaign.

### Eval queue acceptance test 2 (SPEC §8.2, small campaign with restarts): PASSED except overlap, 2026-10-23
**[VERIFIED]** `jac_loop.py eval_queue/campaigns/sr_test_k3.json` (Fernando, 21:16–21:18; worker d8f2cad769eb):
driver Ctrl+C'd and restarted (spec created 21:16:43, restarted run 21:17:07, same req files); worker Ctrl+C'd at
21:17:13 during the first request (pool started 21:16:54) and restarted 21:17:16. Every candidate (7 stencil, 5
candidates) answered exactly once; base and B5 ±dB shared one compile (eval_s ~10–12 s each vs ~23 s for the
others); DONE written. Still to do: the overlap part (ALLOCATION, sharing, hand-back) at Fernando's next
opticsFinder campaign (his choice (a), 2026-10-23).
