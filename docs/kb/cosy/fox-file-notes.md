# Notes on COSY10.2/pg_June2025.fox

This is the active lattice file (`base` in config.py:17). Notes below
are from a partial read (header + grep for structural markers) —
**not a full read of the file**. Treat as a starting map, not complete
documentation.

**[VERIFIED]** File opens with `INCLUDE 'COSY';` then `PROCEDURE RUN ;`
followed by a long block of `VARIABLE` declarations (fringe-field
coefficients `B1S1`/`B1S2`/etc. for each of 8 dipoles B1-B8, Wien
filter higher-order terms `NE1`/`NM1`/`NE2`, and various scalars used
for ray/matrix indexing and beam envelope tracking: `XX`, `AX`, `YY`,
`AY`, `DE`, `MATRIXINDEX`, `RAYINDEX`).
Source: COSY10.2/pg_June2025.fox:1-40
Recorded: 2026-08-07

**[VERIFIED]** Line 992: `OV 4 3 2 ;` — a COSY order/variable-count
statement. Exact meaning of the three arguments in this file not
independently confirmed; standard COSY Infinity usage is `OV NORD
NVAR NPHASE` (map order, number of variables, number of phase-space
variables) but this is **[UNVERIFIED]** for this specific file — needs
either the COSY manual or user confirmation.
Source: COSY10.2/pg_June2025.fox:992
Recorded: 2026-08-07

**Correction, 2026-08-07**: Confirmed against
`sources/COSYBeamMan100.pdf` p.9. `OV <order> <phase space dimension>
<number of parameters>;`. For this file: order=4, phase space
dimension=3 (full 3D — x-a, y-b, and δ_K/time-of-flight/chromatic
effects), number of parameters=2 (extra "knob" quantities beyond
phase space; which two knobs specifically has not been traced). See
`cosy/language-reference.md` for the full citation. Upgrading this
entry's status: the interpretation above is now **[VERIFIED]**, not
inferred.

**[VERIFIED]** Lines 1105-1106 carry inline comments identifying their
physical meaning directly in-file:
```
SR 0 0 0 0 0 0 (1/350) 0 6; {Relative mass dm/m = 1/350}
SR 0 0 0 0 0 -0.1 0 0 8; {Relative energy dE/E}
```
Source: COSY10.2/pg_June2025.fox:1105-1106
Recorded: 2026-08-07

**Correction, 2026-08-07**: Full `SR` syntax now confirmed against
`sources/COSYBeamMan100.pdf` p.13: `SR <x> <a> <y> <b> <T> <D> <G>
<Z> <color>;`. The two lines set `G` (δ_m = Δm/m) = 1/350 and `D`
(δ_K = ΔK/K) = -0.1 respectively — both match their inline comments
exactly. `D`, `G`, `Z` positions confirmed, not just inferred from the
comments. See `cosy/language-reference.md`.

**[VERIFIED]** Per-element field-strength scale factors referenced by
`config.py`'s `cosy_lines` (e.g. `Q1_SC:= ...`) live around lines
1010-1028, i.e. near the end of the file, separate from the geometric
element definitions (`RA`/`SA`/`DL` calls) which appear earlier
(roughly lines 296-976, matching the line numbers in `cosy_lines`).
This means the "nominal value" edits `tuneChanges` performs land in
two structurally different regions of the file depending on the
parameter: geometric misalignments (`Roll`/`dXY`) near each element's
own definition, field scale factors (`B_SC`) grouped together near the
end.
Source: config.py:40-198 (line number ranges), COSY10.2/pg_June2025.fox:1010-1028
Recorded: 2026-08-07

**[VERIFIED]** Comment at COSY10.2/pg_June2025.fox:115-116 reads "DO
NOT MODIFY UNLESS YOU HAVE RE-ANALYZED DANFYSIK DATA / OR IF YOU WANT
TO SEE THE IMPACT OF THE E- OR B-FIELD INHOMOGENEITY IN THE WF's" —
guards `PROCEDURE SETEFB` (lines 113-~150ish), which assigns the
`NE1(1..4)`, `NE2(1..4)`, `NM1(1..4)` (and presumably `NM2(1..4)`,
not yet read) array elements. These are exactly the electric/magnetic
inhomogeneity coefficient arrays passed into the `WC` (combined Wien
filter) calls for WF1 and WF2 — see `cosy/language-reference.md`'s
`WC` entry: `WC WF1R WF1R 2.365 0.11 NE1 NM1 4 ; {WF1}` at line 593.
"Danfysik" is a power-supply/magnet-measurement vendor, so this reads
as: these coefficients were fit to actual measured field data for the
Wien filters, and must not be treated as free-tunable parameters
unless that measurement has been redone. Flagging as a hard constraint
on the optimization free-parameter set (protocol section on
"Confirm the knobs") — `NE1`/`NE2`/`NM1`/`NM2` should not appear in
`tuneChanges` without the user explicitly overriding this comment.
Source: COSY10.2/pg_June2025.fox:113-135; cosy_lines does not
reference NE1/NE2/NM1/NM2 at all (config.py:40-198), so this
constraint is currently respected by construction — `tuneChanges` has
no mechanism to touch these values anyway.
Recorded: 2026-08-07

### 2026-10-04 correction: WF2 misalignment was spurious; corrected deck
**[VERIFIED]** `pg_June2025.fox` lines 805-809 and 815-819 applied
`TA -0.1 -0.1` before WF2 and `TA 0.1 0.1` after it (MISALIGN = 1), with a
comment saying they were introduced to "correct" for Q9's 0.8 mm
misalignment. The user determined this was a mistake and wrote
`AgentSearch/COSY10.2/pg_June2025_corrected.fox`, identical except that both
blocks are commented out (`{IF (MISALIGN=1); ... ENDIF;}`), so WF2 is aligned.
Everything upstream of WF2 (incl. FP2) is unchanged; FP3/DSSD optics change
(e.g. FP3 offsets of the FP2 family solutions moved by ~+3.7 mm).
`MISALIGN:=1` (line 1035) is the deck default. OV 4 3 2 (order 4).
Source: diff of the two decks; user, 2026-10-04.
Recorded: 2026-10-23

### WF1 block (lines 575-611) and its maps
**[VERIFIED]** Q7 Exit (z 11.387 m) -> DL 0.496188 (DL18) -> WF1 block (TA
0.0011 0.0016 deg, RA -0.0008 deg, SA 45/34 um; CB; `WC WF1R WF1R 2.365 0.11
NE1 NM1 4`, WF1R = 7.0 m; CB; undo; FD; FR 3) -> DL 0.398516 + DL 0.1 (DL19)
-> H3 Entrance (HEX3 entrance, z 14.747 m). ECAT applies three segment maps
there (fort.282-284 = pkl groups 32-34). A single composite map Q7 Exit ->
DL19 exit was made for efl (2026-10-17): it agrees with the three segments to
1e-9 m for |dE/E| < 2% and differs by up to 7e-5 m at |dE/E| > 15% (order-4
truncation, scales as amplitude^5). Files:
AgentSearch/Results/efl_q7exit_to_dl19/.
Recorded: 2026-10-23
