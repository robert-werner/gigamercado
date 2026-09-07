[![PyPI](https://img.shields.io/pypi/v/gigamercado.svg)](https://pypi.org/project/gigamercado/) [![Tests](https://github.com/mapbox/gigamercado/actions/workflows/tests.yml/badge.svg)](https://github.com/mapbox/gigamercado/actions/workflows/tests.yml)

# gigamercado

`gigamercado` extends the functionality of [`mercantile`](https://github.com/mapbox/mercantile) with additional commands


## Quickstart

```sh
pip install gigamercado
```

### Numba acceleration (optional)

For large workloads, install Numba for a **3-tier accelerated path** (mirrors
the [cyrcantile-opencl](https://github.com/...) optimisation strategy):

```sh
pip install gigamercado[fast]   # adds numba>=0.60
```

`HAS_NUMBA` (`gigamercado.HAS_NUMBA`) is `True` when the JIT path is active.
The accelerated code auto-degrades to NumPy/Python when Numba is absent.

| Function | Hot path | Numba acceleration |
|---|---|---|
| `burntiles.project_geom()` | Per-vertex `mercantile.xy()` loop | Vectorised `@njit` Mercator kernel |
| `burntiles.find_extrema()` | Python `zip`/`min`/`max` per feature | NumPy batched min/max on concatenated arrays |
| `burntiles.tile_extrema()` | `mercantile.tile()` calls | `@njit` `_tile_xy` scalar kernel |
| `edge_finder.findedges()` | 8x `np.roll` + `dstack` + `min` | Fused `@njit(parallel=True)` 3x3 stencil (eliminates 8 full-array copies) |
| `super_utils.Unprojecter.unproject()` | Per-vertex `arctan(exp())` loop | Vectorised `@njit` inverse-Mercator kernel |
| `super_utils.tile_parser()` | Python `json.loads` / regex per line | `ThreadPoolExecutor` parallel decode |

### Usage

```sh
Usage: gigamercado [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  burn   Burn a stream of GeoJSON into a output...
  edges  For a stream of [<x>, <y>, <z>] tiles, return...
  union  Returns the unioned shape of a stream of...
```

#### `gigamercado burn`

```
<{geojson} stream> | gigamercado burn <zoom> | <[x, y, z] stream>
```

Takes an input stream of GeoJSON and returns a stream of intersecting `[x, y, z]`s for a given zoom.

![image](https://cloud.githubusercontent.com/assets/5084513/14003508/94bc0994-f110-11e5-8e99-e9aadf07bf8d.png)

```sh
cat data/ellada.geojson | gigamercado burn 10 | mercantile shapes | fio collect
```

![image](https://cloud.githubusercontent.com/assets/5084513/14003559/d5427ba6-f110-11e5-80d5-a2aba6433e77.png)

#### `gigamercado edges`
```
<[x, y, z] stream> | gigamercado edges | <[x, y, z] stream>
```
Outputs a stream of `[x, y, z]`s representing the edge tiles of an input stream of `[x, y, z]`s. Edge tile = any tile that is either directly adjacent to a tile that does not exist, or diagonal to an empty tile.

```
cat data/ellada.geojson | gigamercado burn 10 | gigamercado edges | mercantile shapes | fio collect | geojsonio
```

![image](https://cloud.githubusercontent.com/assets/5084513/14003587/01e8e370-f111-11e5-8df4-ac3ae07bbf92.png)


#### `gigamercado union`

```
<[x, y, z] stream> | gigamercado union | <{geojson} stream>
```

Outputs a stream of unioned GeoJSON from an input stream of `[x, y, z]`s. Like `mercantile shapes` but as an overall footprint instead of individual shapes for each tile.

```
cat data/ellada.geojson | gigamercado burn 10 | gigamercado union | fio collect | geojsonio
```

![image](https://cloud.githubusercontent.com/assets/5084513/14003622/365af88c-f111-11e5-8712-28f42253e270.png)


#### `getting crazy`

```
cat data/ellada.geojson | gigamercado burn 12 | gigamercado edges | gigamercado union | fio collect | geojsonio

```

![image](https://cloud.githubusercontent.com/assets/5084513/14003951/ccfecf3c-f113-11e5-943b-94bd6eca1536.png)


## Contributing

### Developing

```sh
git clone git@github.com:mapbox/gigamercado.git
cd gigamercado
uv venv
source .venv/bin/activate
uv sync --all-groups
```

### Releasing

1. Within your PR, update `pyproject.toml` with the new version number
2. Once the PR is merged, pull `main` and verify the tests

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

3. Tag with the new version number and push

```sh
# ie
$ git checkout main
$ git pull
$ git tag 1.2.3
$ git push origin main 1.2.3
```

4. GitHub Actions builds the release, runs `uvx twine check`, and publishes to PyPI
using the `pypi` environment token
