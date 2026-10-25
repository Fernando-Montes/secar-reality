"""
Whole-block misalignment edits for the FP2->FP3 elements (2026-10-04).

Each element's misalignment in the .fox file is a block written going in
(TA yaw pitch / RA roll / SA x y) and undone coming out in reverse
(SA -x -y / RA -roll / TA -yaw -pitch). The older per-parameter handlers in
prepareCOSYchanges edit one line pair at a time, which (a) has no yaw/pitch
support and (b) lets an 'X' and a 'Y' change on the same element overwrite
each other (both rewrite the same SA lines). Here all five values of an
element are combined and the whole block is rewritten at once.

WF2 is special: in pg_June2025_corrected.fox its blocks are commented out
({IF (MISALIGN=1); ... ENDIF;}) because the old TA +-0.1 deg misalignment was
spurious. Its nominal misalignment is therefore zero, and when any WF2
misalignment knob is used the braces are removed and the block is written
with zeros plus the knob. With no WF2 knob the template is left untouched.

Units: X, Y in meters (same as the legacy 'X'/'Y' knobs); Roll, Yaw, Pitch
in degrees (COSY RA/TA convention).

Line numbers and nominal values are read from the template at import and
checked against the expected statement on each line, so a shifted line
fails loudly here instead of silently editing the wrong line.
"""
import os
import re

from config import cosy_lines, COSY_directory, fox_file

MISALIGN_PARS = ('X', 'Y', 'Roll', 'Yaw', 'Pitch')
_STANDARD_ELEMS = ('Q8', 'Q9', 'Q10', 'Q11', 'B5', 'B6')
_NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

with open(os.path.join(COSY_directory, f"{fox_file}.fox")) as _f:
    _LINES = _f.readlines()


def _line(n):
    return _LINES[n - 1]


def _expect(n, pattern, what):
    if not re.search(pattern, _line(n)):
        raise RuntimeError(f"misalign_blocks: {fox_file}.fox line {n} should be {what}, found: {_line(n)!r}")


def _nums(n, keyword):
    m = re.search(rf'\b{keyword}\s+({_NUM})(?:\s+({_NUM}))?\s*;', _line(n))
    return [float(g) for g in m.groups() if g is not None]


def _standard_block(elem):
    ra_in, ra_out = cosy_lines[elem]['Roll']['line']
    lines = {'in': {'TA': ra_in - 1, 'RA': ra_in, 'SA': ra_in + 1},
             'out': {'SA': ra_out - 1, 'RA': ra_out, 'TA': ra_out + 1}}
    for side in ('in', 'out'):
        for kw, n in lines[side].items():
            _expect(n, rf'^\s*{kw}\s', f"an active '{kw}' statement for {elem}")
    yaw, pitch = _nums(lines['in']['TA'], 'TA')
    (roll,) = _nums(lines['in']['RA'], 'RA')
    x, y = _nums(lines['in']['SA'], 'SA')
    # the exit block must undo the entry block exactly
    assert _nums(lines['out']['TA'], 'TA') == [-yaw, -pitch], elem
    assert _nums(lines['out']['RA'], 'RA') == [-roll], elem
    assert _nums(lines['out']['SA'], 'SA') == [-x, -y], elem
    return {'lines': lines, 'wrap': None,
            'nom': {'X': x, 'Y': y, 'Roll': roll, 'Yaw': yaw, 'Pitch': pitch}}


def _wf2_block():
    # pg_June2025_corrected.fox lines 805-809 (entry) and 815-819 (exit)
    wrap = {'in': (805, 809), 'out': (815, 819)}
    lines = {'in': {'TA': 806, 'RA': 807, 'SA': 808}, 'out': {'SA': 816, 'RA': 817, 'TA': 818}}
    for side in ('in', 'out'):
        _expect(wrap[side][0], r'IF \(MISALIGN=1\);', 'the WF2 IF (MISALIGN=1) line')
        _expect(wrap[side][1], r'ENDIF;', 'the WF2 ENDIF line')
        for n in range(wrap[side][0] + 1, wrap[side][1]):
            _expect(n, r'\b(TA|RA|SA)\b', 'a WF2 TA/RA/SA line')
    _expect(812, r'WC WF2R WF2R', 'the WF2 element (WC)')
    active = not _line(805).lstrip().startswith('{')
    if active:
        raise RuntimeError("misalign_blocks: WF2 misalignment block is active in the template; "
                           "this module assumes the corrected (commented-out, zero) WF2 block")
    return {'lines': lines, 'wrap': wrap,
            'nom': {'X': 0.0, 'Y': 0.0, 'Roll': 0.0, 'Yaw': 0.0, 'Pitch': 0.0}}


MISALIGN_BLOCKS = {e: _standard_block(e) for e in _STANDARD_ELEMS}
MISALIGN_BLOCKS['WF2'] = _wf2_block()


def block_changes(elem, deltas):
    """deltas: {par: value} for any subset of MISALIGN_PARS (added to nominal).
    Returns cosyChanges entries [[line, text], ...] rewriting the whole block."""
    b = MISALIGN_BLOCKS[elem]
    v = {p: b['nom'][p] + deltas.get(p, 0.0) for p in MISALIGN_PARS}
    L = b['lines']
    out = [
        [L['in']['TA'], f" TA {v['Yaw']:.8f} {v['Pitch']:.8f}; {{Yaw, Pitch}}\n"],
        [L['in']['RA'], f" RA {v['Roll']:.8f}; {{Roll}}\n"],
        [L['in']['SA'], f" SA {v['X']:.8f} {v['Y']:.8f}; {{x, y}}\n"],
        [L['out']['SA'], f" SA {-v['X']:.8f} {-v['Y']:.8f}; {{x, y}}\n"],
        [L['out']['RA'], f" RA {-v['Roll']:.8f}; {{Roll}}\n"],
        [L['out']['TA'], f" TA {-v['Yaw']:.8f} {-v['Pitch']:.8f}; {{Yaw, Pitch}}\n"],
    ]
    if b['wrap']:
        for side in ('in', 'out'):
            start, end = b['wrap'][side]
            out.append([start, "IF (MISALIGN=1);\n"])
            out.append([end, "ENDIF;\n"])
    return out
