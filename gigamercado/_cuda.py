"""CUDA backend for gigamercado (numba.cuda).

``HAS_CUDA`` is ``True`` only when numba is installed and an NVIDIA GPU
is reachable at import time.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import cuda

    _HAS_DRIVER = bool(cuda.is_available())
except Exception:
    cuda = None
    _HAS_DRIVER = False

HAS_CUDA = cuda is not None and _HAS_DRIVER

_PI = math.pi
_RE = 6378137.0
_R2D = 180.0 / _PI
_D2R = _PI / 180.0
_HALF_PI = _PI / 2.0
_QUARTER_PI = _PI / 4.0
_THREADS = 256


# ------------------------------------------------------------------ #
#  Device kernels
# ------------------------------------------------------------------ #


@cuda.jit
def _xy_kernel(lng, lat, ox, oy, n):
    i = cuda.grid(1)
    if i >= n:
        return
    ox[i] = _RE * lng[i] * _D2R
    oy[i] = _RE * math.log(math.tan(_QUARTER_PI + lat[i] * _D2R * 0.5))


@cuda.jit
def _lnglat_kernel(x, y, olng, olat, n):
    i = cuda.grid(1)
    if i >= n:
        return
    olng[i] = (x[i] / _RE) * _R2D
    olat[i] = (2.0 * math.atan(math.exp(y[i] / _RE)) - _HALF_PI) * _R2D


@cuda.jit
def _tile_kernel(lng, lat, ox, oy, zoom, n):
    i = cuda.grid(1)
    if i >= n:
        return
    sinlat = math.sin(lat[i] * _D2R)
    x01 = lng[i] / 360.0 + 0.5
    logarg = (1.0 + sinlat) / (1.0 - sinlat)
    y01 = 0.5 - 0.25 * math.log(logarg) / _PI
    z2 = 2.0**zoom
    eps = 1e-14
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


@cuda.jit
def _edge_stencil_kernel(
    burn, ncols, xmin, ymin, zoom, out_x, out_y, out_cnt, nrows_total, ncols_total
):
    # 2D grid: gx=col interior, gy=row interior
    gx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    gy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    r = gy + 1
    c = gx + 1
    if r >= nrows_total - 1 or c >= ncols_total - 1:
        return
    idx = r * ncols_total + c
    if not burn[idx]:
        return
    is_edge = False
    for dr in range(-1, 2):
        if is_edge:
            break
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nb = (r + dr) * ncols_total + (c + dc)
            if not burn[nb]:
                is_edge = True
                break
    if is_edge:
        pos = cuda.atomic.add(out_cnt, 0, 1)
        out_x[pos] = c + xmin - 1
        out_y[pos] = r + ymin - 1


# ------------------------------------------------------------------ #
#  Backend class
# ------------------------------------------------------------------ #


class CudaBackend:
    """Singleton CUDA backend for gigamercado GPU operations."""

    _instance: "CudaBackend | None" = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_CUDA:
            raise RuntimeError("numba.cuda is not available")
        cuda.select_device(0)

    def _blocks(self, n):
        return (n + _THREADS - 1) // _THREADS

    def xy_batch(self, lngs, lats):
        n = len(lngs)
        d_l = cuda.to_device(np.ascontiguousarray(lngs, np.float64))
        d_a = cuda.to_device(np.ascontiguousarray(lats, np.float64))
        d_ox = cuda.device_array(n, np.float64)
        d_oy = cuda.device_array(n, np.float64)
        _xy_kernel[self._blocks(n), _THREADS](d_l, d_a, d_ox, d_oy, np.uint64(n))
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def lnglat_batch(self, xs, ys):
        n = len(xs)
        d_x = cuda.to_device(np.ascontiguousarray(xs, np.float64))
        d_y = cuda.to_device(np.ascontiguousarray(ys, np.float64))
        d_ol = cuda.device_array(n, np.float64)
        d_oa = cuda.device_array(n, np.float64)
        _lnglat_kernel[self._blocks(n), _THREADS](d_x, d_y, d_ol, d_oa, np.uint64(n))
        return d_ol.copy_to_host(), d_oa.copy_to_host()

    def tile_merc_batch(self, lngs, lats, zoom):
        n = len(lngs)
        d_l = cuda.to_device(np.ascontiguousarray(lngs, np.float64))
        d_a = cuda.to_device(np.ascontiguousarray(lats, np.float64))
        d_ox = cuda.device_array(n, np.int32)
        d_oy = cuda.device_array(n, np.int32)
        _tile_kernel[self._blocks(n), _THREADS](
            d_l, d_a, d_ox, d_oy, np.int32(zoom), np.uint64(n)
        )
        return d_ox.copy_to_host(), d_oy.copy_to_host()

    def edge_stencil(self, burn_padded, xmin, ymin, zoom):
        nrows, ncols = burn_padded.shape
        gncols = ncols - 2
        gnrows = nrows - 2
        if gncols <= 0 or gnrows <= 0:
            return np.empty((0, 3), np.int32)
        burn = np.ascontiguousarray(burn_padded, np.bool_)
        d_burn = cuda.to_device(burn.ravel())
        max_out = nrows * ncols
        d_ox = cuda.device_array(max_out, np.int32)
        d_oy = cuda.device_array(max_out, np.int32)
        d_cnt = cuda.to_device(np.zeros(1, np.uint32))
        bpg_x = (gncols + 15) // 16
        bpg_y = (gnrows + 15) // 16
        _edge_stencil_kernel[(bpg_x, bpg_y), (16, 16)](
            d_burn,
            np.int32(ncols),
            np.int32(xmin),
            np.int32(ymin),
            np.uint8(zoom),
            d_ox,
            d_oy,
            d_cnt,
            np.int32(nrows),
            np.int32(ncols),
        )
        n_edges = int(d_cnt.copy_to_host()[0])
        if n_edges == 0:
            return np.empty((0, 3), np.int32)
        ox = d_ox.copy_to_host()[:n_edges]
        oy = d_oy.copy_to_host()[:n_edges]
        return np.column_stack((ox, oy, np.full(n_edges, zoom, np.uint8)))


def get_backend():
    """Return the singleton CudaBackend or None if unavailable."""
    if not HAS_CUDA:
        return None
    try:
        return CudaBackend()
    except Exception:
        return None
