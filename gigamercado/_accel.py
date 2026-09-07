"""Numba-accelerated kernels for gigamercado.

All public helpers fall back to the original pure-Python/NumPy
implementations when Numba is not installed.  The JIT kernels mirror
the cyrcantile-opencl approach: vectorized loops over coordinate
arrays with ``fastmath`` and ``parallel`` where appropriate.

``HAS_NUMBA`` is exported so callers can detect the accelerated path.
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    # Provide stub decorators so the rest of the file can be parsed
    # without Numba at import time.
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator

    prange = range

_PI = math.pi
_RE = 6378137.0
_R2D = 180.0 / _PI
_D2R = _PI / 180.0
_HALF_PI = _PI / 2.0
_QUARTER_PI = _PI / 4.0

_CPU_WORKERS = __import__("os").cpu_count() or 4


# ------------------------------------------------------------------ #
#  Mercator projection / unprojection                                    #
# ------------------------------------------------------------------ #


@njit(cache=True, fastmath=True)
def _xy_one(lng, lat):
    """ct.xy for a single point"""
    x = _RE * lng * _D2R
    y = _RE * math.log(math.tan(_QUARTER_PI + lat * _D2R * 0.5))
    return x, y


@njit(cache=True, fastmath=True, nogil=True)
def _xy_batch(lngs, lats, xs, ys, n):
    """Batch ct.xy — writes into xs, ys arrays."""
    for i in range(n):
        xs[i] = _RE * lngs[i] * _D2R
        ys[i] = _RE * math.log(math.tan(_QUARTER_PI + lats[i] * _D2R * 0.5))


@njit(cache=True, fastmath=True, nogil=True)
def _ul_batch(tx, ty, olng, olat, zoom, n):
    """Batch ct.ul — (xtile, ytile, zoom) -> (lng, lat)."""
    z2 = 2.0**zoom
    for i in range(n):
        olng[i] = tx[i] / z2 * 360.0 - 180.0
        olat[i] = math.atan(math.sinh(_PI * (1.0 - 2.0 * ty[i] / z2))) * _R2D


@njit(cache=True, fastmath=True, nogil=True)
def _unproject_batch(xs, ys, olng, olat, n):
    """Batch unproject Web Mercator -> lng/lat (fast inverse)."""
    for i in range(n):
        olng[i] = xs[i] * _R2D / _RE
        olat[i] = (_HALF_PI - 2.0 * math.atan(math.exp(-ys[i] / _RE))) * _R2D


# ------------------------------------------------------------------ #
#  Edge detection stencil kernel                                         #
# ------------------------------------------------------------------ #


@njit(cache=True, parallel=True, nogil=True)
def _edge_stencil(burn_padded, xmin, ymin, zoom, nrows, ncols, out_x, out_y, out_z):
    """Fused 8-connected morphological edge detection.

    Replaces the ``np.roll × 8 → dstack → min → XOR`` pattern with a
    single parallel pass over the padded boolean grid.

    Returns the number of edge tiles written to ``out_x/out_y/out_z``.
    """
    n_edges = 0
    # --- first pass: count edges -----------------------------------
    for r in prange(1, nrows - 1):
        for c in range(1, ncols - 1):
            center = burn_padded[r, c]
            if not center:
                continue
            # Check if any of 8 neighbours is False
            is_edge = False
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    if not burn_padded[r + dr, c + dc]:
                        is_edge = True
                        break
                if is_edge:
                    break
            if is_edge:
                n_edges += 1  # count — cannot write simultaneously

    # --- second pass: fill output ----------------------------------
    # Note: prange not re-used in second pass to avoid a race; sequential.
    idx = 0
    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            center = burn_padded[r, c]
            if not center:
                continue
            is_edge = False
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    if not burn_padded[r + dr, c + dc]:
                        is_edge = True
                        break
                if is_edge:
                    break
            if is_edge:
                out_x[idx] = r + xmin - 1
                out_y[idx] = c + ymin - 1
                out_z[idx] = zoom
                idx += 1

    return n_edges


def edge_detect(burn_padded, xmin, ymin, zoom):
    """NumPy/Numba edge detection — drop-in for the old ``np.roll`` loop.

    Returns an (N, 3) int array of [x, y, zoom] edge tiles.
    """
    nrows, ncols = burn_padded.shape
    n = nrows * ncols
    out_x = np.empty(n, dtype=np.intp)
    out_y = np.empty(n, dtype=np.intp)
    out_z = np.empty(n, dtype=np.uint8)
    n_edges = _edge_stencil(
        np.ascontiguousarray(burn_padded, dtype=np.bool_),
        np.intp(xmin),
        np.intp(ymin),
        np.uint8(zoom),
        np.intp(nrows),
        np.intp(ncols),
        out_x,
        out_y,
        out_z,
    )
    return np.column_stack((out_x[:n_edges], out_y[:n_edges], out_z[:n_edges]))


# ------------------------------------------------------------------ #
#  Fast project_geom / find_extrema                                     #
# ------------------------------------------------------------------ #


def project_geom_fast(geom):
    """Vectorised ``project_geom`` replace.

    Uses Numba-accelerated ``_xy_batch`` for polygon rings with >2 verts
    and falls back to the scalar loop for tiny geometries.
    """
    gtype = geom["type"]
    if gtype == "Point":
        coords = geom["coordinates"]
        x, y = _xy_one(coords[0], coords[1])
        return {"type": "Point", "coordinates": [x, y]}
    elif gtype == "LineString":
        coords = geom["coordinates"]
        n = len(coords)
        if n > 4:
            a = np.asarray(coords, dtype=np.float64)
            xs = np.empty(n, dtype=np.float64)
            ys = np.empty(n, dtype=np.float64)
            _xy_batch(a[:, 0], a[:, 1], xs, ys, n)
            return {
                "type": "LineString",
                "coordinates": [[float(xs[i]), float(ys[i])] for i in range(n)],
            }
        return {
            "type": "LineString",
            "coordinates": [_xy_one(c[0], c[1]) for c in coords],
        }
    elif gtype == "Polygon":
        parts = []
        for ring in geom["coordinates"]:
            n = len(ring)
            if n > 4:
                a = np.asarray(ring, dtype=np.float64)
                xs = np.empty(n, dtype=np.float64)
                ys = np.empty(n, dtype=np.float64)
                _xy_batch(a[:, 0], a[:, 1], xs, ys, n)
                parts.append([[float(xs[i]), float(ys[i])] for i in range(n)])
            else:
                parts.append([_xy_one(c[0], c[1]) for c in ring])
        return {"type": "Polygon", "coordinates": parts}


def find_extrema_fast(features):
    """Batch ``find_extrema`` that collects all coords at once.

    Avoids Python ``min``/``max``/``zip`` over per-feature tuples.
    """
    all_x = []
    all_y = []
    for feat in features:
        geom = feat["geometry"]
        coords = geom["coordinates"]
        gtype = geom["type"]
        if gtype == "Point":
            all_x.append(np.array([coords[0]]))
            all_y.append(np.array([coords[1]]))
        elif gtype == "LineString":
            a = np.asarray(coords, dtype=np.float64)
            all_x.append(a[:, 0])
            all_y.append(a[:, 1])
        elif gtype == "Polygon":
            for ring in coords:
                a = np.asarray(ring, dtype=np.float64)
                all_x.append(a[:, 0])
                all_y.append(a[:, 1])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    eps = 1.0e-10
    return (
        float(all_x.min()) + eps,
        max(float(all_y.min()) + eps, -85.0511287798066),
        float(all_x.max()) - eps,
        min(float(all_y.max()) - eps, 85.0511287798066),
    )


def tile_extrema_fast(bounds, zoom):
    """Vectorised tile_extrema using Numba-accelerated _tile_xy."""
    nw_lng, nw_lat = bounds[0], bounds[3]
    se_lng, se_lat = bounds[2], bounds[1]
    min_tx, min_ty = _tile_xy(nw_lng, nw_lat, zoom)
    max_tx, max_ty = _tile_xy(se_lng, se_lat, zoom)
    return {
        "x": {"min": min_tx, "max": max_tx + 1},
        "y": {"min": min_ty, "max": max_ty + 1},
    }


@njit(cache=True, fastmath=True)
def _tile_xy(lng, lat, zoom):
    """ct.tile (lng, lat, zoom) -> int (x, y) -- scalar, for Numba.

    Mirrors the cyrcantile.tile implementation exactly: convert to
    normalised 0-1 Mercator position, then tile via floor((pos + eps) * 2^z).

    cyrcantile._xy uses Gudermannian: y = 0.5 - 0.25*ln((1+sin)/(1-sin))/pi
    which equals: y = 0.5 - 0.5*ln(tan(pi/4 + lat_rad/2))/pi
    """
    lat_rad = lat * _D2R
    sinlat = math.sin(lat_rad)
    # Safe Gudermannian inverse -- matches cyrcantile._xy exactly
    logarg = (1.0 + sinlat) / (1.0 - sinlat)
    y01 = 0.5 - 0.25 * math.log(logarg) / _PI
    x01 = lng / 360.0 + 0.5
    z2 = 2.0**zoom
    eps = 1e-14
    if x01 <= 0.0:
        xtile = 0
    elif x01 >= 1.0:
        xtile = int(z2) - 1
    else:
        xtile = int(math.floor((x01 + eps) * z2))
    if y01 <= 0.0:
        ytile = 0
    elif y01 >= 1.0:
        ytile = int(z2) - 1
    else:
        ytile = int(math.floor((y01 + eps) * z2))
    return xtile, ytile


@njit(cache=True, fastmath=True, nogil=True)
def _tile_merc_batch(lngs, lats, ox, oy, zoom, n):
    """Batch ct.tile — writes into ox, oy arrays (single @njit call)."""
    z2 = 2.0**zoom
    eps = 1e-14
    for i in range(n):
        lat_rad = lats[i] * _D2R
        sinlat = math.sin(lat_rad)
        logarg = (1.0 + sinlat) / (1.0 - sinlat)
        y01 = 0.5 - 0.25 * math.log(logarg) / _PI
        x01 = lngs[i] / 360.0 + 0.5
        if x01 <= 0.0:
            ox[i] = 0
        elif x01 >= 1.0:
            ox[i] = int(z2) - 1
        else:
            ox[i] = int(math.floor((x01 + eps) * z2))
        if y01 <= 0.0:
            oy[i] = 0
        elif y01 >= 1.0:
            oy[i] = int(z2) - 1
        else:
            oy[i] = int(math.floor((y01 + eps) * z2))


# ------------------------------------------------------------------ #
#  Unprojecter                                                         #
# ------------------------------------------------------------------ #


def unproject_feature_fast(feature):
    """Vectorised unproject -- replaces Unprojecter.unproject().

    projects each coordinate ring from Web Mercator -> lng/lat using
    the Numba-accelerated ``_unproject_batch``.
    """
    new_coords = []
    for ring in feature["coordinates"]:
        n = len(ring)
        a = np.asarray(ring, dtype=np.float64)
        if n > 4:
            olng = np.empty(n, dtype=np.float64)
            olat = np.empty(n, dtype=np.float64)
            _unproject_batch(a[:, 0], a[:, 1], olng, olat, n)
            new_coords.append([[float(olng[i]), float(olat[i])] for i in range(n)])
        else:
            # falls back to scalar loop for tiny rings
            new_coords.append(
                [
                    [
                        float(a[i, 0] * _R2D / _RE),
                        float(
                            (_HALF_PI - 2.0 * math.atan(math.exp(-a[i, 1] / _RE)))
                            * _R2D
                        ),
                    ]
                    for i in range(n)
                ]
            )
    feature["coordinates"] = new_coords
    return feature


# ------------------------------------------------------------------ #
#  Parallel tile_parser                                                  #
# ------------------------------------------------------------------ #


def tile_parser_fast(tiles, parsenames=False):
    """Parallel tile_parser -- threads JSON/regex decode across cores.

    Replicates the original ``tile_parser + parseString`` behaviour
    exactly: ``split("-") -> [x, y, z]``; ``tile.append(tile.pop(0))``
    reorders to ``[y, z, x]``.
    """
    if parsenames:
        tMatch = re.compile(r"[\d]+-[\d]+-[\d]+")

        def _parse_chunk(lines):
            rows = []
            matcher = tMatch
            for t in lines:
                nums = [int(r) for r in matcher.match(t).group().split("-")]
                nums.append(nums.pop(0))  # [x,y,z] -> [y,z,x]
                rows.append(nums)
            return rows

        if len(tiles) > 1000:
            chunk_size = max(1, len(tiles) // _CPU_WORKERS)
            chunks = [
                tiles[i : i + chunk_size] for i in range(0, len(tiles), chunk_size)
            ]
            with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as pool:
                results = pool.map(_parse_chunk, chunks)
            rows = []
            for r in results:
                rows.extend(r)
            return np.array(rows, dtype=np.intp)
        else:
            return np.array(_parse_chunk(list(tiles)), dtype=np.intp)
    else:
        if len(tiles) > 1000:
            chunk_size = max(1, len(tiles) // _CPU_WORKERS)

            def _json_chunk(lines):
                return [json.loads(t) for t in lines]

            chunks = [
                tiles[i : i + chunk_size] for i in range(0, len(tiles), chunk_size)
            ]
            with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as pool:
                results = pool.map(_json_chunk, chunks)
            rows = []
            for r in results:
                rows.extend(r)
            return np.array(rows, dtype=np.intp)
        return np.array([json.loads(t) for t in tiles], dtype=np.intp)
