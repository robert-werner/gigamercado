import cyrcantile as ct
import numpy as np
from affine import Affine
from cyrcantile import Tile
from rasterio import features

from gigamercado._accel import (
    HAS_NUMBA,
    _xy_one,
    find_extrema_fast,
    tile_extrema_fast,
)
from gigamercado._dispatch import HAS_GPU
from gigamercado._dispatch import xy_batch as _gpu_xy_batch


def project_geom(geom):
    """Project a GeoJSON geometry from lng/lat to Web Mercator.

    Uses GPU batch projection for geometries with >4 vertices (when a GPU
    backend is available), Numba otherwise, falling back to cyrcantile.xy.
    """
    gtype = geom["type"]
    if gtype == "Point":
        coords = geom["coordinates"]
        if HAS_GPU or HAS_NUMBA:
            x, y = _xy_one(coords[0], coords[1])
        else:
            x, y = ct.xy(*coords)
        return {"type": "Point", "coordinates": [float(x), float(y)]}

    elif gtype == "LineString":
        coords = geom["coordinates"]
        n = len(coords)
        if n > 4 and (HAS_GPU or HAS_NUMBA):
            a = np.asarray(coords, dtype=np.float64)
            ox, oy = _gpu_xy_batch(a[:, 0], a[:, 1])
            return {
                "type": "LineString",
                "coordinates": [[float(ox[i]), float(oy[i])] for i in range(n)],
            }
        return {
            "type": "LineString",
            "coordinates": [ct.xy(*c) for c in coords],
        }

    elif gtype == "Polygon":
        parts = []
        for ring in geom["coordinates"]:
            n = len(ring)
            if n > 4 and (HAS_GPU or HAS_NUMBA):
                a = np.asarray(ring, dtype=np.float64)
                ox, oy = _gpu_xy_batch(a[:, 0], a[:, 1])
                parts.append([[float(ox[i]), float(oy[i])] for i in range(n)])
            else:
                parts.append([ct.xy(*c) for c in ring])
        return {"type": "Polygon", "coordinates": parts}


def _feature_extrema(geometry):
    if geometry["type"] == "Polygon":
        x, y = zip(*[c for part in geometry["coordinates"] for c in part])
    elif geometry["type"] == "LineString":
        x, y = zip(*[c for c in geometry["coordinates"]])
    elif geometry["type"] == "Point":
        x, y = geometry["coordinates"]
        return x, y, x, y

    return min(x), min(y), max(x), max(y)


def find_extrema(features):
    if HAS_NUMBA:
        return find_extrema_fast(features)
    epsilon = 1.0e-10
    min_x, min_y, max_x, max_y = zip(
        *[_feature_extrema(f["geometry"]) for f in features]
    )

    return (
        min(min_x) + epsilon,
        max(min(min_y) + epsilon, -85.0511287798066),
        max(max_x) - epsilon,
        min(max(max_y) - epsilon, 85.0511287798066),
    )


def tile_extrema(bounds, zoom):
    if HAS_NUMBA:
        return tile_extrema_fast(bounds, zoom)
    minimumTile = ct.tile(bounds[0], bounds[3], zoom)
    maximumTile = ct.tile(bounds[2], bounds[1], zoom)

    return {
        "x": {"min": minimumTile.x, "max": maximumTile.x + 1},
        "y": {"min": minimumTile.y, "max": maximumTile.y + 1},
    }


def make_transform(tilerange, zoom):
    ulx, uly = ct.xy(*ct.ul(Tile(tilerange["x"]["min"], tilerange["y"]["min"], zoom)))
    lrx, lry = ct.xy(*ct.ul(Tile(tilerange["x"]["max"], tilerange["y"]["max"], zoom)))
    xcell = (lrx - ulx) / float(tilerange["x"]["max"] - tilerange["x"]["min"])
    ycell = (uly - lry) / float(tilerange["y"]["max"] - tilerange["y"]["min"])
    return Affine(xcell, 0, ulx, 0, -ycell, uly)


def burn(polys, zoom):
    bounds = find_extrema(polys)

    tilerange = tile_extrema(bounds, zoom)
    afftrans = make_transform(tilerange, zoom)

    burn = features.rasterize(
        ((project_geom(geom["geometry"]), 255) for geom in polys),
        out_shape=(
            (
                tilerange["y"]["max"] - tilerange["y"]["min"],
                tilerange["x"]["max"] - tilerange["x"]["min"],
            )
        ),
        transform=afftrans,
        all_touched=True,
    )

    xys = np.fliplr(np.dstack(np.where(burn))[0])

    xys[:, 0] += tilerange["x"]["min"]
    xys[:, 1] += tilerange["y"]["min"]

    return np.append(xys, np.zeros((xys.shape[0], 1), dtype=np.uint8) + zoom, axis=1)
