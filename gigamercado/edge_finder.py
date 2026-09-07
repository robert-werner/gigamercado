import numpy as np

from gigamercado import super_utils as sutils
from gigamercado._accel import HAS_NUMBA
from gigamercado._dispatch import HAS_GPU, edge_stencil


def findedges(inputtiles, parsenames):
    tiles = sutils.tile_parser(inputtiles, parsenames)

    xmin, xmax, ymin, ymax = sutils.get_range(tiles)

    zoom = sutils.get_zoom(tiles)

    # make an array of shape (xrange + 3, yrange + 3)
    burn = sutils.burnXYZs(tiles, xmin, xmax, ymin, ymax)

    if HAS_GPU or HAS_NUMBA:
        return edge_stencil(burn, xmin, ymin, zoom)

    # Create the indices for rolling (original NumPy fallback)
    idxs = sutils.get_idx()

    # Using the indices to roll + stack the array, find the minimum along
    # the rolled / stacked axis
    xys_edge = (
        np.min(
            np.dstack([np.roll(np.roll(burn, i[0], 0), i[1], 1) for i in idxs]),
            axis=2,
        )
        ^ burn
    )

    # Set missed non-tiles to False
    xys_edge[not burn] = False

    # Recreate the tile xyzs, and add the min vals
    xys_edge = np.dstack(np.where(xys_edge))[0]
    xys_edge[:, 0] += xmin - 1
    xys_edge[:, 1] += ymin - 1

    # Return the edge array
    return np.append(
        xys_edge, np.zeros((xys_edge.shape[0], 1), dtype=np.uint8) + zoom, axis=1
    )


if __name__ == "__main__":
    findedges()
