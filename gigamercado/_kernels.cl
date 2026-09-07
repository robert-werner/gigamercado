# supermercado/_kernels.cl
# OpenCL kernels for batch Mercator projection, inverse projection,
# and morphological edge detection (2D grid stencil).

#define PI       3.14159265358979323846
#define HALF_PI  1.57079632679489661923
#define QUARTER_PI 0.78539816339744830962
#define R2D      57.29577951308232
#define D2R      0.017453292519943295
#define RE       6378137.0

/* ---- 1. Geographic -> Web Mercator (batch) -------------------- */

__kernel void xy_batch(
    __global const double *lng,
    __global const double *lat,
    __global double *ox,
    __global double *oy,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;
    double l = lng[i] * D2R;
    double p = lat[i] * D2R;
    ox[i] = RE * l;
    oy[i] = RE * log(tan(QUARTER_PI + p * 0.5));
}

/* ---- 2. Web Mercator -> Geographic (batch) -------------------- */

__kernel void lnglat_batch(
    __global const double *x,
    __global const double *y,
    __global double *olng,
    __global double *olat,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;
    olng[i] = (x[i] / RE) * R2D;
    olat[i] = (2.0 * atan(exp(y[i] / RE)) - HALF_PI) * R2D;
}

/* ---- 3. Tile (x,y,z) -> Web Mercator tile coordinates --------- */

__kernel void tile_merc_batch(
    __global const double *lng,
    __global const double *lat,
    __global int *ox,
    __global int *oy,
    const int zoom,
    const ulong n
) {
    size_t i = get_global_id(0);
    if (i >= n) return;

    /* Gudermannian inverse (matches cyrcantile._xy exactly). */
    double l = lng[i] / 360.0 + 0.5;
    double sinlat = sin(lat[i] * D2R);
    double yf = 0.5 - 0.25 * log((1.0 + sinlat) / (1.0 - sinlat)) / PI;
    double z2 = exp2((double)zoom);
    double eps = 1e-14;

    if (l <= 0.0)      ox[i] = 0;
    else if (l >= 1.0) ox[i] = (int)z2 - 1;
    else               ox[i] = (int)floor((l + eps) * z2);

    if (yf <= 0.0)      oy[i] = 0;
    else if (yf >= 1.0) oy[i] = (int)z2 - 1;
    else                oy[i] = (int)floor((yf + eps) * z2);
}

/* ---- 4. Morphological edge detection (2D stencil) ------------- */
/*
 *  Input:  burn_padded[nrows * ncols]  -- unsigned char / bool grid
 *  Output: out_x[out_max], out_y[out_max]  -- flattened edge-tile coords
 *          out_count[0] = number of edges written
 *
 *  Each work-item examines one interior cell. If the cell is True and
 *  has at least one False 8-connected neighbour it is an edge tile.
 *  Using a global atomic to append to the output arrays.
 *
 *  The 2D global size should be (ncols-2, nrows-2) so that idx 0 maps
 *  to (1,1) in the padded grid (i.e. the first non-padding cell).
 */

__kernel void edge_stencil(
    __global const uchar *burn_padded,
    __global int  *out_x,
    __global int  *out_y,
    __global uint *out_count,
    const int ncols,
    const int xmin,
    const int ymin,
    const uchar zoom
) {
    /* 2D work-item: gx = column, gy = row (both 0-indexed inside the
       non-padding region).  Actual grid coordinates = (gy+1, gx+1). */
    int gx = get_global_id(0);  // column in [0, ncols-3]
    int gy = get_global_id(1);  // row    in [0, nrows-3]
    int r  = gy + 1;            // actual row in padded grid
    int c  = gx + 1;            // actual col in padded grid

    size_t rc = (size_t)r * (size_t)ncols + (size_t)c;
    uchar center = burn_padded[rc];
    if (!center) return;

    /* Check 8-connected neighbours */
    int is_edge = 0;
    for (int dr = -1; dr <= 1 && !is_edge; dr++) {
        for (int dc = -1; dc <= 1 && !is_edge; dc++) {
            if (dr == 0 && dc == 0) continue;
            size_t nb = (size_t)(r + dr) * (size_t)ncols + (size_t)(c + dc);
            if (!burn_padded[nb]) is_edge = 1;
        }
    }

    if (is_edge) {
        uint idx = atomic_add(out_count, 1u);
        out_x[idx] = c + xmin - 1;   // padded col -> tile x
        out_y[idx] = r + ymin - 1;   // padded row -> tile y
    }
}
