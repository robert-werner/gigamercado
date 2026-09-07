"""Benchmark all gigamercado backends: CUDA, Vulkan, OpenCL, Numba CPU, pure NumPy.

Mirrors the cyrcantile-opencl main.py structure: one section per back‑end,
4 operations benchmarked per section (xy, lnglat, tile_merc, edge_stencil).

Run from repo root:
    python benchmarks/main.py
    python benchmarks/main.py --operation xy        # single operation
    python benchmarks/main.py --size 4000000         # custom point count
    python benchmarks/main.py --grid 2048            # edge grid size
    python benchmarks/main.py --repeat 5             # repetitions
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np


# ── helpers ────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {msg}")
    print(f"{'─' * 64}")


def timer(fn: Callable, n: int, repeat: int = 3, warmup: int = 1) -> dict:
    """Run *fn* once for warmup, then *repeat* times.  Return stats."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    total = time.perf_counter() - t0
    avg_ms = total / repeat * 1000
    return {"count": repeat, "avg_ms": avg_ms}


# ── pure‑NumPy fallback kernels (no Numba, no GPU) ────────────────

PI = np.pi
RE = 6378137.0
R2D = 180.0 / PI
D2R = PI / 180.0
HALF_PI = PI / 2.0
QPI = PI / 4.0


def pura_xy(lngs, lats):
    x = RE * lngs * D2R
    y = RE * np.log(np.tan(QPI + lats * D2R * 0.5))
    return x, y


def pura_lnglat(xs, ys):
    lng = (xs / RE) * R2D
    lat = (2.0 * np.arctan(np.exp(ys / RE)) - HALF_PI) * R2D
    return lng, lat


def pura_tile(lngs, lats, zoom):
    z2 = 2.0 ** zoom
    sinlat_safe = np.clip(np.sin(lats * D2R), -0.999999999, 0.999999999)
    y01 = 0.5 - 0.25 * np.log((1.0 + sinlat_safe) / (1.0 - sinlat_safe)) / PI
    x01 = lngs / 360.0 + 0.5
    eps = 1e-14
    ox = np.clip(np.floor((x01 + eps) * z2), 0, int(z2) - 1).astype(np.int32)
    oy = np.clip(np.floor((y01 + eps) * z2), 0, int(z2) - 1).astype(np.int32)
    return ox, oy


def pura_edge(burn, xmin, ymin, zoom):
    idxs = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    rolled = [np.roll(np.roll(burn, i[0], 0), i[1], 1) for i in idxs]
    xys_edge = np.min(np.dstack(rolled), axis=2) ^ burn
    xys_edge[burn == 0] = False
    coords = np.dstack(np.where(xys_edge))[0]
    coords[:, 0] += xmin - 1
    coords[:, 1] += ymin - 1
    if len(coords) == 0:
        return np.empty((0, 3), dtype=np.int32)
    return np.column_stack((coords, np.full(len(coords), zoom, dtype=np.uint8)))


# ── test data generators ──────────────────────────────────────────

def make_points(n: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    lngs = rng.uniform(-180, 180, n).astype(np.float64)
    lats = rng.uniform(-85, 85, n).astype(np.float64)
    return lngs, lats


def make_merc_xy(n: int, seed: int = 42):
    """Web Mercator coordinates (±20 037 508 m)."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(-20037508, 20037508, n).astype(np.float64)
    ys = rng.uniform(-20037508, 20037508, n).astype(np.float64)
    return xs, ys


def make_edge_grid(
    grid_side: int = 1024, pad: int = 1, seed: int = 42
) -> tuple:
    """Return (burn_padded, xmin, ymin, zoom) with ~30% fill."""
    rng = np.random.RandomState(seed)
    inner = rng.random((grid_side, grid_side)) < 0.3
    burn = np.zeros((grid_side + 2 * pad, grid_side + 2 * pad), dtype=np.uint8)
    burn[pad:-pad, pad:-pad] = inner
    return burn, -512, -512, 12


# ── availability probes ───────────────────────────────────────────

_HAS_CUDA: bool | None = None
_HAS_VULKAN: bool | None = None
_HAS_OPENCL: bool | None = None


def has_cuda() -> bool:
    global _HAS_CUDA
    if _HAS_CUDA is not None:
        return _HAS_CUDA
    try:
        from gigamercado._cuda import HAS_CUDA, get_backend

        _HAS_CUDA = HAS_CUDA and get_backend() is not None
    except Exception:
        _HAS_CUDA = False
    return _HAS_CUDA


def has_vulkan() -> bool:
    global _HAS_VULKAN
    if _HAS_VULKAN is not None:
        return _HAS_VULKAN
    try:
        from gigamercado._vulkan import HAS_VULKAN, get_backend

        _HAS_VULKAN = HAS_VULKAN and get_backend() is not None
    except Exception:
        _HAS_VULKAN = False
    return _HAS_VULKAN


def has_opencl() -> bool:
    global _HAS_OPENCL
    if _HAS_OPENCL is not None:
        return _HAS_OPENCL
    try:
        from gigamercado._opencl import HAS_OPENCL, get_backend

        _HAS_OPENCL = HAS_OPENCL and get_backend() is not None
    except Exception:
        _HAS_OPENCL = False
    return _HAS_OPENCL


def has_numba() -> bool:
    try:
        from gigamercado._accel import HAS_NUMBA

        return HAS_NUMBA
    except Exception:
        return False


def _get_cuda():
    from gigamercado._cuda import get_backend

    return get_backend()


def _get_vulkan():
    from gigamercado._vulkan import get_backend

    return get_backend()


def _get_opencl():
    from gigamercado._opencl import get_backend

    return get_backend()


def _numba_xy(lngs, lats):
    from gigamercado._accel import _xy_batch

    n = len(lngs)
    ox = np.empty(n, np.float64)
    oy = np.empty(n, np.float64)
    _xy_batch(lngs, lats, ox, oy, n)
    return ox, oy


def _numba_lnglat(xs, ys):
    from gigamercado._accel import _unproject_batch

    n = len(xs)
    olng = np.empty(n, np.float64)
    olat = np.empty(n, np.float64)
    _unproject_batch(xs, ys, olng, olat, n)
    return olng, olat


def _numba_tile(lngs, lats, zoom):
    from gigamercado._accel import _tile_xy

    n = len(lngs)
    ox = np.empty(n, np.int32)
    oy = np.empty(n, np.int32)
    for i in range(n):
        ox[i], oy[i] = _tile_xy(float(lngs[i]), float(lats[i]), zoom)
    return ox, oy


def _numba_edge(burn, xmin, ymin, zoom):
    from gigamercado._accel import edge_detect

    return edge_detect(burn, xmin, ymin, zoom)


# ── benchmark runners ─────────────────────────────────────────────

def bench_xy(lngs, lats, repeat: int):
    banner(f"xy_batch  ({len(lngs):,} points)")
    ref = pura_xy(lngs, lats)
    results: dict[str, float] = {}

    # pure NumPy
    t = timer(lambda: pura_xy(lngs, lats), len(lngs), repeat)
    results["NumPy"] = t["avg_ms"]
    print(f"  NumPy       {t['avg_ms']:10.2f} ms")

    # Numba
    if has_numba():
        t = timer(lambda: _numba_xy(lngs, lats), len(lngs), repeat)
        results["Numba"] = t["avg_ms"]
        print(f"  Numba       {t['avg_ms']:10.2f} ms")

    # OpenCL
    if has_opencl():
        try:
            ocl = _get_opencl()
            t = timer(lambda: ocl.xy_batch(lngs, lats), len(lngs), repeat)
            results["OpenCL"] = t["avg_ms"]
            print(f"  OpenCL      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  OpenCL      (failed: {e})")

    # Vulkan
    if has_vulkan():
        try:
            vk = _get_vulkan()
            vk.xy_batch(lngs[:4096], lats[:4096])
            t = timer(lambda: vk.xy_batch(lngs, lats), len(lngs), repeat)
            results["Vulkan"] = t["avg_ms"]
            print(f"  Vulkan      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  Vulkan      (failed: {e})")

    # CUDA
    if has_cuda():
        try:
            cu = _get_cuda()
            cu.xy_batch(lngs[:4096], lats[:4096])
            t = timer(lambda: cu.xy_batch(lngs, lats), len(lngs), repeat)
            results["CUDA"] = t["avg_ms"]
            print(f"  CUDA        {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  CUDA        (failed: {e})")

    # correctness check
    for name, fn in [
        ("Numba", lambda: _numba_xy(lngs, lats)),
        ("OpenCL", lambda: _get_opencl().xy_batch(lngs, lats) if has_opencl() else None),
        ("Vulkan", lambda: _get_vulkan().xy_batch(lngs, lats) if has_vulkan() else None),
        ("CUDA", lambda: _get_cuda().xy_batch(lngs, lats) if has_cuda() else None),
    ]:
        try:
            out = fn()
            if out is not None:
                match = np.allclose(ref[0], out[0], atol=1e-3) and np.allclose(
                    ref[1], out[1], atol=1e-3
                )
                if not match:
                    print(f"  ⚠ {name} != NumPy (max Δ: "
                          f"{np.max(np.abs(ref[0]-out[0])):.6f}, "
                          f"{np.max(np.abs(ref[1]-out[1])):.6f})")
        except Exception:
            pass

    return results


def bench_lnglat(xs, ys, repeat: int):
    banner(f"lnglat_batch  ({len(xs):,} points)")
    ref = pura_lnglat(xs, ys)
    results: dict[str, float] = {}

    t = timer(lambda: pura_lnglat(xs, ys), len(xs), repeat)
    results["NumPy"] = t["avg_ms"]
    print(f"  NumPy       {t['avg_ms']:10.2f} ms")

    if has_numba():
        t = timer(lambda: _numba_lnglat(xs, ys), len(xs), repeat)
        results["Numba"] = t["avg_ms"]
        print(f"  Numba       {t['avg_ms']:10.2f} ms")

    if has_opencl():
        try:
            ocl = _get_opencl()
            t = timer(lambda: ocl.lnglat_batch(xs, ys), len(xs), repeat)
            results["OpenCL"] = t["avg_ms"]
            print(f"  OpenCL      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  OpenCL      (failed: {e})")

    if has_vulkan():
        try:
            vk = _get_vulkan()
            vk.lnglat_batch(xs[:4096], ys[:4096])
            t = timer(lambda: vk.lnglat_batch(xs, ys), len(xs), repeat)
            results["Vulkan"] = t["avg_ms"]
            print(f"  Vulkan      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  Vulkan      (failed: {e})")

    if has_cuda():
        try:
            cu = _get_cuda()
            cu.lnglat_batch(xs[:4096], ys[:4096])
            t = timer(lambda: cu.lnglat_batch(xs, ys), len(xs), repeat)
            results["CUDA"] = t["avg_ms"]
            print(f"  CUDA        {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  CUDA        (failed: {e})")

    return results


def bench_tile(lngs, lats, zoom: int, repeat: int):
    banner(f"tile_merc_batch  ({len(lngs):,} pts, z={zoom})")
    ref = pura_tile(lngs, lats, zoom)
    results: dict[str, float] = {}

    t = timer(lambda: pura_tile(lngs, lats, zoom), len(lngs), repeat)
    results["NumPy"] = t["avg_ms"]
    print(f"  NumPy       {t['avg_ms']:10.2f} ms")

    if has_numba():
        t = timer(lambda: _numba_tile(lngs, lats, zoom), len(lngs), repeat)
        results["Numba"] = t["avg_ms"]
        print(f"  Numba       {t['avg_ms']:10.2f} ms")

    if has_opencl():
        try:
            ocl = _get_opencl()
            t = timer(lambda: ocl.tile_merc_batch(lngs, lats, zoom), len(lngs), repeat)
            results["OpenCL"] = t["avg_ms"]
            print(f"  OpenCL      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  OpenCL      (failed: {e})")

    if has_vulkan():
        try:
            vk = _get_vulkan()
            vk.tile_merc_batch(lngs[:4096], lats[:4096], zoom)
            t = timer(
                lambda: vk.tile_merc_batch(lngs, lats, zoom), len(lngs), repeat
            )
            results["Vulkan"] = t["avg_ms"]
            print(f"  Vulkan      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  Vulkan      (failed: {e})")

    if has_cuda():
        try:
            cu = _get_cuda()
            cu.tile_merc_batch(lngs[:4096], lats[:4096], zoom)
            t = timer(
                lambda: cu.tile_merc_batch(lngs, lats, zoom), len(lngs), repeat
            )
            results["CUDA"] = t["avg_ms"]
            print(f"  CUDA        {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  CUDA        (failed: {e})")

    return results


def bench_edge(burn, xmin, ymin, zoom, repeat: int):
    grid_side = burn.shape[0]
    banner(f"edge_stencil  ({grid_side}×{grid_side} grid)")
    ref = pura_edge(burn, xmin, ymin, zoom)
    results: dict[str, float] = {}

    t = timer(lambda: pura_edge(burn, xmin, ymin, zoom), grid_side, repeat)
    results["NumPy"] = t["avg_ms"]
    print(f"  NumPy       {t['avg_ms']:10.2f} ms")

    if has_numba():
        t = timer(lambda: _numba_edge(burn, xmin, ymin, zoom), grid_side, repeat)
        results["Numba"] = t["avg_ms"]
        print(f"  Numba       {t['avg_ms']:10.2f} ms")

    if has_opencl():
        try:
            ocl = _get_opencl()
            t = timer(
                lambda: ocl.edge_stencil(burn, xmin, ymin, zoom), grid_side, repeat
            )
            results["OpenCL"] = t["avg_ms"]
            print(f"  OpenCL      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  OpenCL      (failed: {e})")

    if has_vulkan():
        try:
            vk = _get_vulkan()
            vk.edge_stencil(burn, xmin, ymin, zoom)
            t = timer(
                lambda: vk.edge_stencil(burn, xmin, ymin, zoom), grid_side, repeat
            )
            results["Vulkan"] = t["avg_ms"]
            print(f"  Vulkan      {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  Vulkan      (failed: {e})")

    if has_cuda():
        try:
            cu = _get_cuda()
            cu.edge_stencil(burn, xmin, ymin, zoom)
            t = timer(
                lambda: cu.edge_stencil(burn, xmin, ymin, zoom), grid_side, repeat
            )
            results["CUDA"] = t["avg_ms"]
            print(f"  CUDA        {t['avg_ms']:10.2f} ms")
        except Exception as e:
            print(f"  CUDA        (failed: {e})")

    # correctness check
    ref_n = len(ref)
    for name, fn in [
        ("Numba", lambda: _numba_edge(burn, xmin, ymin, zoom)),
        (
            "OpenCL",
            lambda: (
                _get_opencl().edge_stencil(burn, xmin, ymin, zoom)
                if has_opencl()
                else None
            ),
        ),
        (
            "CUDA",
            lambda: (
                _get_cuda().edge_stencil(burn, xmin, ymin, zoom)
                if has_cuda()
                else None
            ),
        ),
    ]:
        try:
            out = fn()
            if out is not None and len(out) > 0:
                if len(out) != ref_n:
                    print(f"  ⚠ {name} edge count {len(out)} != NumPy {ref_n}")
        except Exception:
            pass

    return results


# ── speedup table printer ──────────────────────────────────────────

def print_speedup_table(all_results: dict[str, dict[str, float]]) -> None:
    banner("SPEEDUP SUMMARY (vs NumPy baseline)")
    backends = ["NumPy", "Numba", "OpenCL", "Vulkan", "CUDA"]
    header = f"{'Operation':<20}" + "".join(f"{b:>10}" for b in backends)
    print(header)
    print("─" * len(header))

    for op, res in all_results.items():
        base = res.get("NumPy", 1.0)
        cols = ""
        for b in backends:
            if b == "NumPy":
                cols += f"{'1.00':>10}"
            elif b in res:
                speedup = base / res[b]
                cols += f"{speedup:>9.1f}x"
            else:
                cols += f"{'—':>10}"
        print(f"{op:<20}{cols}")


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gigamercado GPU benchmark")
    parser.add_argument(
        "--size",
        type=int,
        default=8_000_000,
        help="Number of random points (default 8 000 000)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=1024,
        help="Edge grid side length (default 1024)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Repetitions per benchmark (default 3)"
    )
    parser.add_argument(
        "--operation",
        choices=["xy", "lnglat", "tile", "edge", "all"],
        default="all",
        help="Run only one operation",
    )
    args = parser.parse_args()

    n = args.size
    grid = args.grid
    repeat = args.repeat

    banner("gigamercado benchmark  —  backend availability")
    print(f"  CUDA   : {has_cuda()}")
    print(f"  Vulkan : {has_vulkan()}")
    print(f"  OpenCL : {has_opencl()}")
    print(f"  Numba  : {has_numba()}")
    print(f"  NumPy  : always")
    print(f"\n  points = {n:,}   grid = {grid}×{grid}   repeat = {repeat}")

    # Generate shared test data once
    lngs, lats = make_points(n)
    merc_xs, merc_ys = make_merc_xy(n)
    burn, xmin, ymin, zoom = make_edge_grid(grid, pad=1)

    # Warm up Numba JIT (first invocation compiles the kernels)
    if has_numba():
        _ = _numba_xy(lngs[:4096], lats[:4096])
        _ = _numba_lnglat(merc_xs[:4096], merc_ys[:4096])
        _ = _numba_tile(lngs[:4096], lats[:4096], 12)

    all_results: dict[str, dict[str, float]] = {}
    ops = {
        "xy": lambda: bench_xy(lngs, lats, repeat),
        "lnglat": lambda: bench_lnglat(merc_xs, merc_ys, repeat),
        "tile": lambda: bench_tile(lngs, lats, 12, repeat),
        "edge": lambda: bench_edge(burn, xmin, ymin, zoom, repeat),
    }

    if args.operation == "all":
        for name, fn in ops.items():
            all_results[name] = fn()
        print_speedup_table(all_results)
    else:
        ops[args.operation]()

    print("\n  Done ✓\n")


if __name__ == "__main__":
    main()
