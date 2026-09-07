import cyrcantile as ct
import numpy as np
from cyrcantile import Tile
from rasterio import Affine, features

from gigamercado import super_utils as sutils


def union(inputtiles, parsenames):

    tiles = sutils.tile_parser(inputtiles, parsenames)

    xmin, xmax, ymin, ymax = sutils.get_range(tiles)

    zoom = sutils.get_zoom(tiles)

    # make an array of shape (xrange + 3, yrange + 3)
    burn = sutils.burnXYZs(tiles, xmin, xmax, ymin, ymax, 0)

    nw = ct.xy(*ct.ul(Tile(xmin, ymin, zoom)))

    se = ct.xy(*ct.ul(Tile(xmax + 1, ymax + 1, zoom)))

    aff = Affine(
        ((se[0] - nw[0]) / float(xmax - xmin + 1)),
        0.0,
        nw[0],
        0.0,
        -((nw[1] - se[1]) / float(ymax - ymin + 1)),
        nw[1],
    )

    unprojecter = sutils.Unprojecter()

    unionedTiles = [
        {
            "geometry": unprojecter.unproject(feature),
            "properties": {},
            "type": "Feature",
        }
        for feature, shapes in features.shapes(
            np.asarray(np.flipud(np.rot90(burn)).astype(np.uint8), order="C"),
            transform=aff,
        )
        if shapes == 1
    ]

    return unionedTiles


if __name__ == "__main__":
    union()
