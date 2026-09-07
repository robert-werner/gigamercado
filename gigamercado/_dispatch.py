"""Backend dispatcher for gigamercado GPU acceleration.

Tries backends in priority order: CUDA > Vulkan > OpenCL > Numba CPU.
Each backend exposes the same 4-method interface:

    .xy_batch(lngs, lats) -> (ox, oy)
    .lnglat_batch(xs, ys) -> (olng, olat)
    .tile_merc_batch(lngs, lats, zoom) -> (ox, oy)
    .edge_stencil(burn_padded, xmin, ymin, zoom) -> (N,3) int array

HAS_GPU is True when any GPU backend is available.
backend_name is "cuda" | "vulkan" | "opencl" | "numba" | "none".
"""

from __future__ import annotations

import numpy as np

# Try backends in priority order
_gpu = None
_backend_name = "none"

try:
    from gigamercado._cuda import HAS_CUDA
    from gigamercado._cuda import get_backend as _cuda_backend

    if HAS_CUDA:
        _gpu = _cuda_backend()
        _backend_name = "cuda"
except Exception:
    HAS_CUDA = False

if _gpu is None:
    try:
        from gigamercado._vulkan import HAS_VULKAN
        from gigamercado._vulkan import get_backend as _vk_backend

        if HAS_VULKAN:
            _gpu = _vk_backend()
            _backend_name = "vulkan"
    except Exception:
        HAS_VULKAN = False

if _gpu is None:
    try:
        from gigamercado._opencl import HAS_OPENCL
        from gigamercado._opencl import get_backend as _ocl_backend

        if HAS_OPENCL:
            _gpu = _ocl_backend()
            _backend_name = "opencl"
    except Exception:
        HAS_OPENCL = False

if _gpu is None:
    try:
        from gigamercado._accel import HAS_NUMBA

        if HAS_NUMBA:
            _backend_name = "numba"
    except Exception:
        HAS_NUMBA = False

HAS_GPU = _gpu is not None
backend_name = _backend_name


# ------------------------------------------------------------------ #
#  Unified batch API
# ------------------------------------------------------------------ #


def xy_batch(lngs, lats):
    """Batch ct.xy. Works on GPU or CPU."""
    if _gpu is not None:
        return _gpu.xy_batch(lngs, lats)
    from gigamercado._accel import _xy_batch

    lngs = np.ascontiguousarray(lngs, np.float64)
    lats = np.ascontiguousarray(lats, np.float64)
    n = len(lngs)
    ox = np.empty(n, np.float64)
    oy = np.empty(n, np.float64)
    _xy_batch(lngs, lats, ox, oy, n)
    return ox, oy


def lnglat_batch(xs, ys):
    """Batch inverse Mercator. Works on GPU or CPU."""
    if _gpu is not None:
        return _gpu.lnglat_batch(xs, ys)
    from gigamercado._accel import _unproject_batch

    xs = np.ascontiguousarray(xs, np.float64)
    ys = np.ascontiguousarray(ys, np.float64)
    n = len(xs)
    olng = np.empty(n, np.float64)
    olat = np.empty(n, np.float64)
    _unproject_batch(xs, ys, olng, olat, n)
    return olng, olat


def tile_merc_batch(lngs, lats, zoom):
    """Batch ct.tile via GPU or CPU."""
    if _gpu is not None:
        return _gpu.tile_merc_batch(lngs, lats, zoom)
    from gigamercado._accel import _tile_merc_batch

    lngs = np.ascontiguousarray(lngs, np.float64)
    lats = np.ascontiguousarray(lats, np.float64)
    n = len(lngs)
    ox = np.empty(n, np.int32)
    oy = np.empty(n, np.int32)
    _tile_merc_batch(lngs, lats, ox, oy, zoom, n)
    return ox, oy


def edge_stencil(burn_padded, xmin, ymin, zoom):
    """Morphological edge detection. Works on GPU or CPU."""
    if _gpu is not None:
        return _gpu.edge_stencil(burn_padded, xmin, ymin, zoom)
    from gigamercado._accel import edge_detect as _cpu_edge

    return _cpu_edge(burn_padded, xmin, ymin, zoom)
