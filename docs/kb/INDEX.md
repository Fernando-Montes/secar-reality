# Knowledge base index

Bootstrapped 2026-08-07. This index is a map, not a summary — read the
linked file for detail.

**Owner**: the user-level agent `secar-reality` (`~/.claude/agents/secar-reality.md`,
since 2026-10-23; formerly the project-level OpticsAgent). Other SECAR agents
and how to reach them: "SECAR agents" in `~/.claude/CLAUDE.md`.

## Current state (2026-10-23) — read first

- Location (2026-10-23): this KB, the sources and the pipeline now live in
  `/Users/fernandomontes/Research/COSY/Simulations/Agents/secar-reality/`
  (formerly `MC/ECAT/docs/kb`, `MC/ECAT/sources`, `MC/ECAT/AgentSearch`).
  Older entries saying `AgentSearch/...` mean this folder. Details:
  `simulation/agentsearch-pipeline.md`, "Relocation 2026-10-23".
- Active lattice: `pg_June2025_corrected.fox` (spurious WF2 misalignment
  removed 2026-10-04). FP3/DSSD results computed before then are on the
  wrong lattice. See `cosy/fox-file-notes.md` and `runs.md` (2026-10-04).
- The agent's pipeline (fast_tracer, misalignment blocks, single-pass E2
  evaluation, script rules): `simulation/agentsearch-pipeline.md`.
- E2 beam and loss definitions (FP2 4-term, FP3 Mahalanobis):
  `measurements/datasets.md` (2026-10-23 section).
- Where the work stands: `runs.md` from "Entries added 2026-10-23" on —
  46 FP2 solutions; FP3 knob ranking and 29-knob Jacobian (FP3 M12 needs
  ~+10% Q9); archived Hall-probe check (wrong FP2 slope signs); WF1 EFL
  scenarios with the efl agent (WF1 field model shifts FP2 M16 notably).
- Pending: `wf1_knob_jacobian.py` (user-run) to test whether WF1's EFL is a
  new knob; the user's "more to the story" on Q6/other magnets (archived
  data via secar-pv-desk).

## simulation/ — facts about the user's Python code

- [entry-points.md](simulation/entry-points.md) — how the code is run
  (`main_new2.py`), what `runECAT()` does, how it decides whether to
  invoke COSY. Confidence: verified against code read 2026-08-07.
- [units-and-conventions.md](simulation/units-and-conventions.md) —
  units for position/angle/energy at the Python boundary. Confidence:
  verified against code read 2026-08-07, but not cross-checked against
  what COSY itself assumes internally — see `cosy/` for that side.
- [config-reference.md](simulation/config-reference.md) — what
  `config.py` defines: `cosy_lines` (element → .fox line-number map),
  `elements_dict` (beamline geometry), viewers/apertures/detectors.
  Confidence: verified against code read 2026-08-07.
- [agentsearch-pipeline.md](simulation/agentsearch-pipeline.md) — the
  agent's AgentSearch copy as of 2026-10: environment, corrected lattice,
  bug fixes, cache naming/cleanup, fast_tracer, misalignment blocks
  (yaw/pitch, WF2), single-pass E2 evaluation, script conventions, ME(1,7)
  caveat. Confidence: verified against code and validation runs.

## cosy/ — COSY Infinity file and invocation details

- [invocation.md](cosy/invocation.md) — how COSY is actually invoked
  from Python (`./cosy <file>.fox`), what output it produces, how that
  output becomes transfer matrices. Confidence: verified against code
  read 2026-08-07.
- [fox-file-notes.md](cosy/fox-file-notes.md) — structural notes on
  `COSY10.2/pg_June2025.fox`, the active lattice file. Mostly literal
  transcriptions of specific lines, now cross-checked against the
  manual (see below). Confidence: partial — only the sections read
  so far, but the parts read are manual-verified, not inferred.
- `cosy/language-reference.md` — COSYScript
  syntax reference distilled from `sources/COSYBeamMan100.pdf`
  (COSY Beam Physics Manual): coordinates, `OV`, `SR`, element
  statements (`DL`, `M5`/multipoles, `WC` Wien filter, `FR` fringe-
  field modes), misalignments (`SA`/`TA`/`RA`). **Read this before
  editing any `.fox` file content or writing a new `tuneChanges`
  entry** — it documents the degrees-vs-radians trap (see below) and
  the Wien filter "DO NOT MODIFY" field-inhomogeneity block.
  Confidence: verified against manual, ~57 of 71 manual pages read
  (sections 1-3.3.11); sections 3.3.12 onward (analysis, examples,
  COSYScript syntax table) not yet read.

**Unit trap, confirmed 2026-08-07**: "angle" means two different
things in COSY depending on context — phase-space slope coordinates
(`SR`'s `a`,`b`, and the Python `ax`/`ay`) are in **radians**;
element/misalignment angles (`RA`, `TA`, bend angles) are in
**degrees**. Both appear in the same `.fox` file. Full derivation in
`cosy/language-reference.md` and `simulation/units-and-conventions.md`.

## secar/ — physical beamline layout

- [layout.md](secar/layout.md) — element ordering and drift lengths
  along the beamline (Target → DSSD), as encoded in `config.py`.
  Confidence: verified against code, but the code itself is the only
  source consulted so far — not cross-checked against a design
  document or survey data. Treat drift lengths as "what the simulation
  currently assumes," not independently confirmed physical fact.

## measurements/

- [datasets.md](measurements/datasets.md) — two datasets are wired
  into the code: `E3_optimization/` (VD1515/VD1542 only, used to fit
  initial-distribution parameters) and `PartD_ViewerAnalysis/Group{1..19}/`
  (7 viewers × 19 groups, each group a different pepperpot-mask
  alignment / dE-E setting, per `groupsInfo` in `ViewerAnalysis.py`).
  **Resolved 2026-08-07**: `PartD_ViewerAnalysis` (via its per-viewer
  PNG figures, curated) is the dataset actually used for the completed
  optimization work — see `runs.md`. Experiment: E22503D, proton beam
  on 20Ne target. `E3_optimization/`'s role is still unconfirmed.

## runs.md — completed manual optimization work (read from user-supplied files, not derived by the agent)

**Read this before touching any lattice `tuneChanges`** — it documents
a substantial, mostly-completed sequential optimization already done
by the user (`Results/OpticsOptimization.xlsx` sheet `E2` +
`sources/OpticsComparison-MatrixElements.pdf`), matching simulated
vs. measured **M12 (dX/dalphaX) and M16 (dX/dE) slopes** at each focal
plane, one focal plane at a time (FP2→FP3→FP4). FP2 and FP3 converged
to specific `tuneChanges`; **FP4 (VD1836/VD1879) is confirmed
unfinished** by the user. Two findings worth knowing before extending
this work:
1. Nominal (untuned) ECAT gave the **opposite sign** slope from
   measurement at FP2 — not just an offset.
2. The FP3 Q9 `B_SC` +9.49% correction that matches the slopes was
   checked against an independent Hall probe field measurement, which
   **did not confirm** a matching hardware discrepancy — flagged per
   the optimization protocol as an unresolved model-error symptom, not
   a validated hardware finding. (Separately, Q9 also has a *different*,
   independently-measurement-confirmed 0.834 mm X-misalignment — see
   `secar/layout.md` — which is unrelated and does not explain the
   slope mismatch.)

**⚠️ Naming collision, not the same thing**: the code has a directory
called `E3_optimization/` (see `measurements/datasets.md`) — this is
**unrelated** to "E3," the user's name for the ¹⁴N⁶⁺ pilot-beamtime
validation work described next. Don't conflate them.

**Central open finding (2026-08-07) — E2 and E3 disagree**: E3 is a
real online pilot beamtime (¹⁴N⁶⁺ beam) that attempted to validate
E2's simulation-derived corrections against actual hardware, using an
independent method (minimizing beam spot width at the *true* focal
plane rather than matching M12/M16 slopes). **They disagree
substantially**: E2 implies a Q6 hardware correction of +2.7%; E3's
direct measurement gives +0.5%. FP3's Q9/Q11 only partially agree.
This is an unresolved, currently-open discrepancy between two
independent validation methods — see `runs.md`'s E3 section for full
detail. **Active work as of 2026-08-07**: a new joint E2+E3 fit is
underway — target/metric/knob-scope decisions recorded in `runs.md`'s
"Target established" section (top of file).

## optics/ — general beam-transport theory (Wollnik textbook)
(Local notes only: these and `cosy/language-reference.md` summarize copyrighted books/manuals and are not
in the public repository.)

Full book read 2026-08-07 (background agents), book pages 1-291
(index at 291-293), all `[VERIFIED]` with book-page citations. This is
**general theory, not SECAR-specific** — never a source for an actual
SECAR parameter, only for interpreting/diagnosing the simulation.

- `optics/transfer-matrices.md` (Ch.1, p.1-26) — transfer-matrix method, thin lens, general optical systems.
- `optics/particle-motion-em-fields.md` (Ch.2, p.27-47) — confirms Wollnik's a=p_x/p_0, b=p_y/p_0 match COSY's exactly; all angles here are radians.
- `optics/quadrupole-lenses.md` (Ch.3, p.49-88) — quad trajectories, multiplet design.
- `optics/sector-field-lenses.md` (Ch.4, p.89-137) — **most SECAR-relevant**: a sector field body focuses in x from geometry alone; **all y-focusing comes from entrance/exit edge angles** (§4.1.3), explaining COSY's `DI` ε1/ε2 params. Sign-convention match vs. COSY `[UNVERIFIED]`.
- `optics/phase-space-and-beams.md` (Ch.5, p.139-170) — Liouville's theorem, Twiss parameters, envelopes, Abbe's sine law.
- `optics/periodic-structures.md` (Ch.6, p.171-186) — Hill's equation, tune, FODO. Low relevance (SECAR is single-pass, not a ring).
- `optics/fringing-fields.md` (Ch.7, p.187-209) — cross-reference for COSY's `FR` modes / Enge functions.
- `optics/image-aberrations.md` (Ch.8, p.211-251) — aberration nomenclature (`(x|aa)`, `(x|yy)`), symplecticity.
- `optics/spectrometer-design.md` (Ch.9, p.254-289) — dispersion/resolving-power formulas. Eq.(9.4d) is a plausible (`[UNVERIFIED]`) match for the SECAR file's `MRESOL_P1`/`MRESOL_P2` procedures.

**Convention note**: no degrees-based angle appears anywhere in
Wollnik's 9 chapters — always radians. Genuine difference from COSY's
`RA`/`TA`/bend-angle degrees convention — remember the conversion when
translating between them.

## sources/

Three documents added by the user 2026-08-07:
- `COSYBeamMan100.pdf` — COSY INFINITY 10.0 Beam Physics Manual
  (Berz & Makino, MSU, Aug 2017, 71 pages). Read pages 1-36 (sections
  1-3.3.11: control commands, coordinates, all element types through
  misalignments). Distilled into `cosy/language-reference.md` and
  corrections in `cosy/fox-file-notes.md`.
- `COSYProgMan100.pdf` — COSY INFINITY 10.0 Programmer's Manual
  (same authors, 107 pages). Only pages 1-15 read (installation,
  general language internals) — this is the *general* COSYScript
  manual, not the beam-physics element reference; the latter is
  entirely in the Beam Manual above. Not yet mined for COSYScript
  control-flow syntax (`IF`/`ENDIF`, `VARIABLE`, procedures) beyond
  what's directly legible in the `.fox` file itself.
- `OpticsOfChargedParticles_Wollnik.pdf` — full book (151 PDF pages,
  two physical book pages per PDF page, rotated scan; 9 chapters,
  book pages 1-291 + index). **Fully read 2026-08-07** at the user's
  explicit request ("read all chapters now so you can make suggestions
  later" — a deliberate departure from the default on-demand reading
  protocol). Distilled into the 9 files under `optics/` above, ~124
  fact blocks total, all page-cited.
- `OpticsComparison-MatrixElements.pdf` — 24-slide presentation of the
  completed FP2/FP3 (and partial FP4) E2 optimization work. Slides
  1-14 are duplicated verbatim in the next file below.
- `OpticsComparison-update_20Ne.pdf` — 17 slides; 1-14 = same as above,
  **15-17 are new**: the E3 ¹⁴N⁶⁺ pilot-beamtime validation, which
  disagrees with E2 (see the central open finding above). Both fully
  read 2026-08-07; distilled into `runs.md`, `measurements/datasets.md`,
  `secar/layout.md`.

## Open questions for the user

- **Top priority**: reconcile the E2/E3 disagreement — see `runs.md`'s
  E3 section for the four concrete next steps the user already
  identified (re-check E2 with Q6 +0.5%, resolve the 1.4%/1.22%
  online-correction inconsistency, refit slopes with E3 knob values,
  and the FP1/FP3 "entry 689" items).
- FP4 (VD1836/VD1879): confirmed unfinished by the user. Next step per
  the presentation's own "Next steps" slide — find/confirm Q14/Q15
  correction, investigate high-order magnet effects; phenomenological
  correction as fallback. See `runs.md`.
- The Q9 `B_SC` +9.49% correction (FP3) is not corroborated by Hall
  probe measurement — worth deciding whether to keep chasing a
  physical explanation or accept it as an empirical correction. See
  `runs.md`.
- What `_cosy` means in `PartD_ViewerAnalysis` filenames, and what
  `E3_optimization/`'s relationship (if any) is to the completed work.
- ~~Is `COSY10.2/pg_June2025.fox` the correct active lattice file?~~
  Resolved 2026-10-04 for AgentSearch: `pg_June2025_corrected.fox`
  (see "Current state" above). The original ECAT `config.py` is unchanged.
- `test.pdf` at the repo root is a generated output, not a source
  document — `main_new2.py:82` writes beamline cross-section plots to
  `saveFile = 'test.pdf'`. Not a KB source.
