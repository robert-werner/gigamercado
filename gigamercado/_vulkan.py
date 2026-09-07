"""Vulkan (wgpu/WGSL) backend for gigamercado.

``HAS_VULKAN`` is ``True`` only when wgpu is installed and a Vulkan
capable device is reachable.
"""

from __future__ import annotations

import numpy as np

try:
    import wgpu
    import wgpu.backends.auto  # noqa: F401

    _adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    _device = _adapter.request_device() if _adapter else None
    HAS_VULKAN = _device is not None
except Exception:
    wgpu = None
    _device = None
    HAS_VULKAN = False

# -- WGSL kernels ------------------------------------------------ #

_WGSL = {}

# NOTE: WGSL storage buffers use f32 (wgpu requires FLOAT64 capability for
# f64).  We upload f64 inputs as f32, run f32 on GPU, read back f64.

_WGSL["xy"] = """struct UBO { n: u32 };
@group(0) @binding(0) var<storage, read>       lng  : array<f32>;
@group(0) @binding(1) var<storage, read>       lat  : array<f32>;
@group(0) @binding(2) var<storage, read_write> ox   : array<f32>;
@group(0) @binding(3) var<storage, read_write> oy   : array<f32>;
@group(0) @binding(20) var<uniform>            u    : UBO;
const PI = 3.141592741012573;
const RE = 6378137.0;
const QP = 0.785398185253143;
const D2R = 0.017453292384744;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    ox[i] = f32(RE) * lng[i] * D2R;
    oy[i] = f32(RE) * log(tan(QP + lat[i] * D2R * 0.5));
}"""

_WGSL["lnglat"] = """struct UBO { n: u32 };
@group(0) @binding(0) var<storage, read>       x    : array<f32>;
@group(0) @binding(1) var<storage, read>       y    : array<f32>;
@group(0) @binding(2) var<storage, read_write> olng : array<f32>;
@group(0) @binding(3) var<storage, read_write> olat : array<f32>;
@group(0) @binding(20) var<uniform>            u    : UBO;
const HP = 1.570796370506287;
const R2D = 57.295780181884766;
const RE = 6378137.0;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    olng[i] = (x[i] / f32(RE)) * R2D;
    olat[i] = (2.0 * atan(exp(y[i] / f32(RE))) - HP) * R2D;
}"""

_WGSL["tile"] = """struct UBO { zoom: i32, n: u32 };
@group(0) @binding(0) var<storage, read>       lng : array<f32>;
@group(0) @binding(1) var<storage, read>       lat : array<f32>;
@group(0) @binding(2) var<storage, read_write> ox  : array<i32>;
@group(0) @binding(3) var<storage, read_write> oy  : array<i32>;
@group(0) @binding(20) var<uniform>            u   : UBO;
const PI  = 3.141592741012573;
const D2R = 0.017453292384744;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n) { return; }
    let z2 = exp2(f32(u.zoom));
    let l = lng[i] / 360.0 + 0.5;
    let sinlat = sin(lat[i] * D2R);
    let sinlat_safe = clamp(sinlat, -0.99999994, 0.99999994);
    let yf = 0.5 - 0.25 * log((1.0 + sinlat_safe) / (1.0 - sinlat_safe)) / PI;
    let eps = 1e-7;
    if (l <= 0.0)      { ox[i] = 0; }
    else if (l >= 1.0) { ox[i] = i32(z2) - 1; }
    else               { ox[i] = i32(floor((l + eps) * z2)); }
    if (yf <= 0.0)      { oy[i] = 0; }
    else if (yf >= 1.0) { oy[i] = i32(z2) - 1; }
    else                { oy[i] = i32(floor((yf + eps) * z2)); }
}"""

# Edge stencil: 2D dispatched, one work-item per interior cell.
# Uses an atomic counter to append edge tiles to output arrays.
_WGSL["edge"] = """struct UBO { ncols: i32, xmin: i32, ymin: i32, nrows: i32 };
@group(0) @binding(0) var<storage, read>       burn : array<u32>;   // packed bits
@group(0) @binding(1) var<storage, read_write> outX : array<i32>;
@group(0) @binding(2) var<storage, read_write> outY : array<i32>;
@group(0) @binding(3) var<storage, read_write> cnt  : array<atomic<u32>>;
@group(0) @binding(20) var<uniform>            u    : UBO;
const QP = 0.78539816339744830962;

fn burn_at(r: i32, c: i32) -> bool {
    let idx = u32(r * u.ncols + c);
    return (burn[idx >> 5u] >> (idx & 31u)) & 1u != 0u;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gx = i32(gid.x);  // column in interior
    let gy = i32(gid.y);  // row    in interior
    if (gx >= u.ncols - 2 || gy >= u.nrows - 2) { return; }
    let r = gy + 1;
    let c = gx + 1;
    if (!burn_at(r, c)) { return; }
    var is_edge = false;
    for (var dr = -1; dr <= 1 && !is_edge; dr++) {
        for (var dc = -1; dc <= 1 && !is_edge; dc++) {
            if (dr == 0 && dc == 0) { continue; }
            if (!burn_at(r + dr, c + dc)) { is_edge = true; }
        }
    }
    if (is_edge) {
        let idx = atomicAdd(&cnt[0], 1u);
        outX[idx] = c + u.xmin - 1;
        outY[idx] = r + u.ymin - 1;
    }
}"""

# Dispatch: 4096 workgroups of 256 threads for 1D, 16x16 workgroups for 2D.
_ROW_WORKGROUPS = 4096
_ROW_THREADS = _ROW_WORKGROUPS * 256


class VulkanBackend:
    """Singleton Vulkan/wgpu backend for gigamercado GPU operations."""

    _instance: "VulkanBackend | None" = None

    def __new__(cls):
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._init()
            cls._instance = obj
        return cls._instance

    def _init(self):
        if not HAS_VULKAN:
            raise RuntimeError("wgpu is not available")
        self.device = _device
        self._pipes = {}

    def _pipe(self, key):
        pipe = self._pipes.get(key)
        if pipe is not None:
            return pipe
        wgsl = _WGSL[key]
        shader = self.device.create_shader_module(code=wgsl)
        pipe = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )
        self._pipes[key] = pipe
        return pipe

    def _upload(self, arr):
        return self.device.create_buffer_with_data(
            data=arr, usage=wgpu.BufferUsage.STORAGE
        )

    def _out(self, nbytes):
        return self.device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

    def _uni(self, values):
        data = np.zeros(8, dtype=np.int32)
        data[: len(values)] = values
        return self.device.create_buffer_with_data(
            data=data, usage=wgpu.BufferUsage.UNIFORM
        )

    def _read(self, buf, dtype, count):
        raw = self.device.queue.read_buffer(buf)
        return np.frombuffer(raw, dtype=dtype, count=count).copy()

    def _dispatch_1d(self, key, buffers, n):
        pipe = self._pipe(key)
        entries = []
        for binding, buf in buffers:
            resource = {
                "buffer": buf,
                "offset": 0,
                "size": buf.size,
            }
            entries.append({"binding": binding, "resource": resource})
        bg = self.device.create_bind_group(
            layout=pipe.get_bind_group_layout(0), entries=entries
        )
        rows = (n + _ROW_THREADS - 1) // _ROW_THREADS
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipe)
        p.set_bind_group(0, bg)
        p.dispatch_workgroups(_ROW_WORKGROUPS, rows, 1)
        p.end()
        self.device.queue.submit([enc.finish()])

    def _dispatch_2d(self, key, buffers, gx, gy):
        pipe = self._pipe(key)
        entries = []
        for binding, buf in buffers:
            resource = {
                "buffer": buf,
                "offset": 0,
                "size": buf.size,
            }
            entries.append({"binding": binding, "resource": resource})
        bg = self.device.create_bind_group(
            layout=pipe.get_bind_group_layout(0), entries=entries
        )
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(pipe)
        p.set_bind_group(0, bg)
        p.dispatch_workgroups((gx + 15) // 16, (gy + 15) // 16, 1)
        p.end()
        self.device.queue.submit([enc.finish()])

    def xy_batch(self, lngs, lats):
        n = len(lngs)
        f32 = np.float32
        bl = self._upload(np.ascontiguousarray(lngs, f32))
        ba = self._upload(np.ascontiguousarray(lats, f32))
        bx = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([n])
        self._dispatch_1d(
            "xy", [(0, bl), (1, ba), (2, bx), (3, by), (20, bu)], n
        )
        return (
            self._read(bx, np.float32, n).astype(np.float64),
            self._read(by, np.float32, n).astype(np.float64),
        )

    def lnglat_batch(self, xs, ys):
        n = len(xs)
        f32 = np.float32
        bx = self._upload(np.ascontiguousarray(xs, f32))
        by = self._upload(np.ascontiguousarray(ys, f32))
        bo = self._out(n * 4)
        bl = self._out(n * 4)
        bu = self._uni([n])
        self._dispatch_1d(
            "lnglat", [(0, bx), (1, by), (2, bo), (3, bl), (20, bu)], n
        )
        return (
            self._read(bo, np.float32, n).astype(np.float64),
            self._read(bl, np.float32, n).astype(np.float64),
        )

    def tile_merc_batch(self, lngs, lats, zoom):
        n = len(lngs)
        f32 = np.float32
        bl = self._upload(np.ascontiguousarray(lngs, f32))
        ba = self._upload(np.ascontiguousarray(lats, f32))
        bx = self._out(n * 4)
        by = self._out(n * 4)
        bu = self._uni([zoom, n])
        self._dispatch_1d(
            "tile", [(0, bl), (1, ba), (2, bx), (3, by), (20, bu)], n
        )
        return self._read(bx, np.int32, n), self._read(by, np.int32, n)

    def edge_stencil(self, burn_padded, xmin, ymin, zoom):
        nrows, ncols = burn_padded.shape
        gncols = ncols - 2
        gnrows = nrows - 2
        if gncols <= 0 or gnrows <= 0:
            return np.empty((0, 3), np.int32)
        # Pack bool grid into u32 bitmask for storage buffer
        flat = burn_padded.ravel().astype(np.uint8)
        bits = np.packbits(flat, bitorder="little")  # uint8 array
        # Align to 4 bytes (u32)
        padded_len = ((len(bits) + 3) // 4) * 4
        bits_padded = np.zeros(padded_len, np.uint8)
        bits_padded[: len(bits)] = bits
        bits_u32 = bits_padded.view(np.uint32)

        max_out = nrows * ncols
        bb = self._upload(bits_u32)
        bx = self._out(max_out * 4)
        by = self._out(max_out * 4)
        bc = self._out(4)  # atomic counter
        # Zero the counter
        self.device.queue.write_buffer(bc, 0, np.zeros(1, np.uint32).tobytes())
        bu = self._uni([ncols, xmin, ymin, nrows, 0, 0, 0, zoom])

        self._dispatch_2d(
            "edge",
            [(0, bb), (1, bx), (2, by), (3, bc), (20, bu)],
            gncols,
            gnrows,
        )

        n_edges = self._read(bc, np.uint32, 1)[0]
        if n_edges == 0:
            return np.empty((0, 3), np.int32)
        ox = self._read(bx, np.int32, n_edges)
        oy = self._read(by, np.int32, n_edges)
        return np.column_stack((ox, oy, np.full(n_edges, zoom, np.uint8)))


def get_backend():
    """Return the singleton VulkanBackend or None if unavailable."""
    if not HAS_VULKAN:
        return None
    try:
        return VulkanBackend()
    except Exception:
        return None
