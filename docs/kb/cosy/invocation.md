# How COSY is invoked from Python

**[VERIFIED]** `runECAT()` shells out to a local COSY Infinity binary
via `run_cosy_and_save_fort(f"./cosy {cosyFile}.fox", f"{cosyFile}")`
(helper_functions.py:520 and :526) — i.e. it runs `./cosy
<name>.fox` as a subprocess (exact subprocess mechanics inside
`run_cosy_and_save_fort` not yet read) from what is presumably the
`COSY10.2` directory (`COSY_directory` in config.py:9), given `./cosy`
is a relative path to the compiled binary.

After running, `process_fort_files(f"{cosyFile}")` is called
(helper_functions.py:521, :527) — COSY Infinity's standard output
mechanism is Fortran unit files (`fort.*`), so this step is presumably
parsing those into a usable form. Not yet read.

`load_optics_data(cosyFile, pkl_directory)` (helper_functions.py:522,
:529, :532) then returns `df, grouped_data_frames` — `grouped_data_frames`
is what `process_all_rays` consumes as the per-element transfer
matrices (see `simulation/entry-points.md`). This suggests
`process_fort_files` writes intermediate results into `pkl_directory`
as pickled per-element matrix data, which `load_optics_data` reads
back. Not yet confirmed by reading `process_fort_files`/
`load_optics_data` directly.
Source: helper_functions.py:516-532
Recorded: 2026-08-07

**[UNVERIFIED]** Whether `run_cosy_and_save_fort`/`process_fort_files`
do any unit conversion between COSY's internal convention and the
meters/radians/fractional-dE convention used at the Python
`initialDistribution` boundary (see
`simulation/units-and-conventions.md`). Needs a direct read of those
two functions before trusting a unit-sensitive result.
