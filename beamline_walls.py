"""
Draw the beamline chamber walls ("beamline sizes") on x-vs-z and y-vs-z ray
plots (2026-10-19, user request: from now on, ray plots include these).

Same idea as the Emulator project's optics_plots.draw_chamber_walls: for each
segment of the beamline geometry (config.beamline_geometry, a list of
{'name', 'z_start', 'z_end', 'polygon'}), the wall position along x at y = 0
(or along y at x = 0) is drawn as a step outline over [z_start, z_end]. Here
the wall positions come from intersecting the chamber polygon with the x (or
y) axis line, so off-centre chambers are drawn at their true walls (+ and -
sides separately). Open segments (polygon None) are left blank.

These are the same polygons ECAT uses for the transmission check; note that
ECAT checks rays only at the discrete z locations of the element list, while
the walls are drawn continuously.

    from beamline_walls import draw_chamber_walls
    draw_chamber_walls(ax, 'x', scale=1000)      # axes in mm
"""
import pickle
import numpy as np
from shapely.geometry import LineString

from config import beamline_geometry

_SEGMENTS = None


def chamber_segments():
    """[(name, z0, z1, {'x': (lo, hi), 'y': (lo, hi)} in m, or None if open)], cached."""
    global _SEGMENTS
    if _SEGMENTS is None:
        with open(beamline_geometry, 'rb') as f:
            bl = pickle.load(f)
        out = []
        for s in bl:
            p = s['polygon']
            ext = None
            if p is not None:
                ext = {}
                for c, line in (('x', LineString([(-10, 0), (10, 0)])), ('y', LineString([(0, -10), (0, 10)]))):
                    seg = p.intersection(line)
                    if seg.is_empty:
                        ext[c] = None
                    else:
                        b = seg.bounds
                        ext[c] = (b[0], b[2]) if c == 'x' else (b[1], b[3])
            out.append((s['name'], s['z_start'], s['z_end'], ext))
        _SEGMENTS = out
    return _SEGMENTS


def draw_chamber_walls(ax, coord, scale=1000.0, zmax=None, color='0.55', lw=1.0, label='chamber walls'):
    """Step outline of the walls along `coord` ('x' or 'y'); scale converts m to the axis unit."""
    zz, lo, hi = [], [], []
    for name, z0, z1, ext in chamber_segments():
        if zmax is not None and z0 >= zmax:
            break
        z1 = min(z1, zmax) if zmax is not None else z1
        e = None if ext is None else ext[coord]
        if e is None:                                   # open segment: break the outline
            zz += [z0, z1]
            lo += [np.nan, np.nan]
            hi += [np.nan, np.nan]
        else:                                           # continuous step outline
            zz += [z0, z1]
            lo += [e[0] * scale] * 2
            hi += [e[1] * scale] * 2
    ax.plot(zz, hi, color=color, lw=lw, zorder=1, label=label)
    ax.plot(zz, lo, color=color, lw=lw, zorder=1)


def zoom_to_rays(ax, margin=1.3):
    """Call AFTER plotting the rays and BEFORE draw_chamber_walls: fixes the y range to the rays'
    extent times `margin`, so wide chambers don't squash the beam (walls beyond it run off the edge)."""
    lo, hi = ax.get_ylim()
    m = max(abs(lo), abs(hi)) * margin
    ax.set_ylim(-m, m)
    ax.set_autoscaley_on(False)
