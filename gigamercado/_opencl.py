"""OpenCL backend for gigamercado.

Provides GPU-accelerated Mercator projection/inverse-projection and
morphological edge detection via ``pyopencl``.

``HAS_OPENCL`` is ``True`` only when pyopencl is installed and a
suitable OpenCL device is reachable.
"""

from __future__ import annotations

import pathlib

import numpy as np

try:
    import pyopencl as cl

    HAS_OPENCL = True
except Exception:
    cl = None
    HAS_OPENCL = False

_KERNEL_PATH = pathlib.Path(__file__).resolve().parent / "_kernels.cl"


class OpenCLBackend:
    """Singleton OpenCL backend for gigamercado GPU operations."""

    _instance: "OpenCLBackend | None" = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_OPENCL:
            raise RuntimeError("pyopencl is not available")
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)
        source = _KERNEL_PATH.read_text()
        self.program = cl.Program(self.ctx, source).build()
        self.mf = cl.mem_flags

    # -- helpers ---------------------------------------------------

    def _buf(self, arr, flags):
        return cl.Buffer(self.ctx, flags | cl.mem_flags.COPY_HOST_PTR, arr.nbytes, arr)

    def _wb(self, arr):
        return cl.Buffer(self.ctx, self.mf.WRITE_ONLY, arr.nbytes)

    def _readback(self, host, devbuf):
        cl.enqueue_copy(self.queue, host, devbuf).wait()

    # -- public batch operations -----------------------------------

    def xy_batch(self, lngs, lats):
        """Batch ct.xy -- lng,lat arrays to Web Mercator x,y."""
        n = len(lngs)
        lngs = np.ascontiguousarray(lngs, np.float64)
        lats = np.ascontiguousarray(lats, np.float64)
        ox = np.empty(n, np.float64)
        oy = np.empty(n, np.float64)
        bl = self._buf(lngs, self.mf.READ_ONLY)
        ba = self._buf(lats, self.mf.READ_ONLY)
        bxo = self._wb(ox)
        byo = self._wb(oy)
        self.program.xy_batch(self.queue, (n,), None, bl, ba, bxo, byo, np.uint64(n))
        self._readback(ox, bxo)
        self._readback(oy, byo)
        bl.release()
        ba.release()
        bxo.release()
        byo.release()
        self.queue.finish()
        return ox, oy

    def lnglat_batch(self, xs, ys):
        """Batch inverse Mercator -- Web Mercator x,y arrays to lng,lat."""
        n = len(xs)
        xs = np.ascontiguousarray(xs, np.float64)
        ys = np.ascontiguousarray(ys, np.float64)
        olng = np.empty(n, np.float64)
        olat = np.empty(n, np.float64)
        bx = self._buf(xs, self.mf.READ_ONLY)
        by = self._buf(ys, self.mf.READ_ONLY)
        blo = self._wb(olng)
        bla = self._wb(olat)
        self.program.lnglat_batch(
            self.queue, (n,), None, bx, by, blo, bla, np.uint64(n)
        )
        self._readback(olng, blo)
        self._readback(olat, bla)
        bx.release()
        by.release()
        blo.release()
        bla.release()
        self.queue.finish()
        return olng, olat

    def tile_merc_batch(self, lngs, lats, zoom):
        """Batch ct.tile(lng, lat, zoom) via GPU Gudermannian."""
        n = len(lngs)
        lngs = np.ascontiguousarray(lngs, np.float64)
        lats = np.ascontiguousarray(lats, np.float64)
        ox = np.empty(n, np.int32)
        oy = np.empty(n, np.int32)
        bl = self._buf(lngs, self.mf.READ_ONLY)
        ba = self._buf(lats, self.mf.READ_ONLY)
        bxo = self._wb(ox)
        byo = self._wb(oy)
        self.program.tile_merc_batch(
            self.queue, (n,), None, bl, ba, bxo, byo, np.int32(zoom), np.uint64(n)
        )
        self._readback(ox, bxo)
        self._readback(oy, byo)
        bl.release()
        ba.release()
        bxo.release()
        byo.release()
        self.queue.finish()
        return ox, oy

    def edge_stencil(self, burn_padded, xmin, ymin, zoom):
        """GPU 2D morphological edge detection.

        Input: burn_padded -- 2D uint8 or bool array (padded by 1 on each side).
        Returns (N, 3) int array of [x, y, zoom] edge tiles.
        """
        burn = np.ascontiguousarray(burn_padded, np.uint8)
        nrows, ncols = burn.shape
        # Work dimensions cover the non-padding region
        gncols = ncols - 2  # interior columns
        gnrows = nrows - 2  # interior rows
        if gncols <= 0 or gnrows <= 0:
            return np.empty((0, 3), np.int32)
        max_out = nrows * ncols  # worst case
        out_x = np.empty(max_out, np.int32)
        out_y = np.empty(max_out, np.int32)
        out_count = np.zeros(1, np.uint32)

        bb = self._buf(burn, self.mf.READ_ONLY)
        bx = self._wb(out_x)
        by = self._wb(out_y)
        bc = cl.Buffer(
            self.ctx,
            self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            out_count.nbytes,
            out_count,
        )

        self.program.edge_stencil(
            self.queue,
            (gncols, gnrows),
            None,
            bb,
            bx,
            by,
            bc,
            np.int32(ncols),
            np.int32(xmin),
            np.int32(ymin),
            np.uint8(zoom),
        )

        n_edges = np.zeros(1, np.uint32)
        self._readback(n_edges, bc)
        n_edges = int(n_edges[0])

        if n_edges > 0:
            self._readback(out_x, bx)
            self._readback(out_y, by)

        bb.release()
        bx.release()
        by.release()
        bc.release()
        self.queue.finish()

        if n_edges == 0:
            return np.empty((0, 3), np.int32)
        return np.column_stack(
            (
                out_x[:n_edges],
                out_y[:n_edges],
                np.full(n_edges, zoom, np.uint8),
            )
        )


def get_backend():
    """Return the singleton OpenCLBackend or None if unavailable."""
    if not HAS_OPENCL:
        return None
    try:
        return OpenCLBackend()
    except Exception:
        return None
