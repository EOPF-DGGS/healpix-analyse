# `alm_latlon` — Part 3: API reference

> **Module** `healpix_analyse.alm_latlon`  
> **Parts** [1 · Quickstart](alm_latlon_1_quickstart.md) · [2 · Mathematics](alm_latlon_2_mathematics.md) · [3 · API reference](#)

---

## Overview

| Function | Purpose |
|----------|---------|
| [`build_rings_from_latlon`](#build_rings_from_latlon) | Sort a flat pixel list into iso-latitude rings |
| [`compute_weights`](#compute_weights) | Compute per-pixel quadrature weights |
| [`map2alm_latlon`](#map2alm_latlon) | Map → spherical harmonic coefficients |
| [`alm2map_latlon`](#alm2map_latlon) | Spherical harmonic coefficients → map |
| [`anafast_latlon`](#anafast_latlon) | Map → angular power spectrum |
| [`grid_summary`](#grid_summary) | Print a diagnostic summary of the grid |

---

## `build_rings_from_latlon`

```python
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    lat, lon,
    atol=1e-10,
    convention="colatitude_rad"
)
```

Group a flat list of $N$ pixels into iso-latitude rings, sorted by colatitude.

This is the **mandatory first step** before any transform call.  The function
sorts pixels by colatitude (then longitude) and identifies contiguous groups
whose colatitude values agree within `atol`.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `lat` | array-like `[N]` | — | Latitude or colatitude of each pixel. Unit and sign depend on `convention`. |
| `lon` | array-like `[N]` | — | Longitude of each pixel. Unit depends on `convention`. |
| `atol` | `float` | `1e-10` | Angular tolerance in **radians** (after conversion) to merge two pixels into the same ring. Increase to `1e-6` when coordinates come from single-precision data or are read from a GRIB/NetCDF file with limited decimal places. |
| `convention` | `str` | `"colatitude_rad"` | Coordinate convention (see table below). |

#### Coordinate conventions

| `convention` | `lat` | `lon` |
|---|---|---|
| `"colatitude_rad"` *(default)* | colatitude $\theta$, radians, $0 \to \pi$ | longitude $\phi$, radians, $0 \to 2\pi$ |
| `"colatitude_deg"` | colatitude $\theta$, degrees, $0° \to 180°$ | longitude $\phi$, degrees, $0° \to 360°$ |
| `"geographic_rad"` | geographic latitude, radians, $-\pi/2 \to +\pi/2$ | longitude, radians, $-\pi \to +\pi$ or $0 \to 2\pi$ |
| `"geographic_deg"` | geographic latitude, degrees, $-90° \to +90°$ | longitude, degrees, $-180° \to +180°$ or $0° \to 360°$ |

All conventions convert internally to colatitude + longitude in radians before
any further processing.

### Returns

| Name | Type | Description |
|------|------|-------------|
| `ring_theta` | `np.ndarray [R]`, float64 | Colatitude $\theta_r$ in **radians** of each ring, sorted in ascending order (North pole first). |
| `ring_phi_list` | `list` of `np.ndarray [N_r]` | Longitudes $\phi_j$ in **radians** for each ring. |
| `ring_counts` | `np.ndarray [R]`, int64 | Number of pixels $N_r$ in each ring. |
| `sort_idx` | `np.ndarray [N]`, int64 | Permutation index. Apply as `im_sorted = im[sort_idx]` to put a flat map into ring order before passing it to any transform function. |

### Raises

`ValueError` — if `convention` is not one of the four accepted strings.

### Notes

- Pixels within the same ring do **not** need to be pre-sorted in longitude;
  `build_rings_from_latlon` sorts them internally.
- Pixels at different colatitudes that differ by less than `atol` are merged
  into the same ring.  If two rings are spuriously merged, decrease `atol` or
  ensure the input coordinates are in `float64`.

---

## `compute_weights`

```python
weights = compute_weights(
    ring_theta,
    ring_phi_list,
    ring_counts,
    quadrature="trapeze"
)
```

Compute per-pixel quadrature weights in steradians such that:

$$
\int f \, d\Omega \approx \sum_{i} f_i \, w_i
$$

The weights are the product of a colatitude weight $w_r^\theta$ (ring-level) and
a longitude weight $w_j^\phi$ (pixel-level within the ring).

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ring_theta` | `np.ndarray [R]` | — | Colatitudes in radians (output of `build_rings_from_latlon`). |
| `ring_phi_list` | `list` of `np.ndarray` | — | Longitudes in radians per ring. |
| `ring_counts` | `np.ndarray [R]` | — | Number of pixels per ring. |
| `quadrature` | `str` | `"trapeze"` | Quadrature rule for the colatitude integral. See table below. |

#### Quadrature rules

| `quadrature` | Colatitude weight $w_r^\theta$ | Accuracy | Best for |
|---|---|---|---|
| `"trapeze"` | $\sin(\theta_r) \cdot \Delta\theta_r$ (trapezoidal) | $O(\Delta\theta^2)$ | Regular grids, HEALPix |
| `"gauss_legendre"` | Gauss-Legendre weight $w_r^{\text{GL}}$ | Exact for $\ell \leq 2R-1$ | **ERA5, ECMWF, IFS, ARPEGE** |
| `"equal_area"` | $4\pi / N_{\text{total}}$ (uniform) | Exact for equal-area grids | HEALPix |

For the longitude direction, the weights are:
- **Uniform ring:** $w_j^\phi = 2\pi / N_r$ (exact rectangle rule).
- **Irregular ring:** wrapped trapezoidal rule on the sorted longitudes.

### Returns

| Name | Type | Description |
|------|------|-------------|
| `weights` | `np.ndarray [N_total]`, float64 | Per-pixel weights in steradians, in ring order (same order as a map reindexed by `sort_idx`). |

### Raises

- `ValueError` — if `quadrature` is not one of the three accepted strings.
- `UserWarning` — when `quadrature="gauss_legendre"` and the provided
  colatitudes deviate from the expected Gauss-Legendre nodes by more than
  $10^{-6}$ radians.

---

## `map2alm_latlon`

```python
alm = map2alm_latlon(
    im,
    ring_theta,
    ring_phi_list,
    ring_counts,
    lmax=24,
    weights=None,
    quadrature="trapeze",
    limit_range=1e10,
)
```

Compute spherical harmonic coefficients $a_{\ell m}$ from a map.

The transform follows the orthonormal convention (same as `healpy.map2alm`):

$$
a_{\ell m} = \int f(\theta, \phi) \, Y_{\ell m}^*(\theta, \phi) \, d\Omega
$$

The algorithm performs two sequential steps:
1. **Longitude DFT** (per ring): FFT + phase shift for uniform rings, direct DFT for irregular rings.
2. **Legendre projection** (over rings): for each $m$, projects the ring Fourier
   coefficients onto normalised associated Legendre polynomials.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `im` | array-like or `torch.Tensor`, shape `[..., N_total]` | — | Map values in ring order. **Apply `sort_idx`** from `build_rings_from_latlon` first. A leading batch dimension `[B, N_total]` is supported. |
| `ring_theta` | `np.ndarray [R]` | — | Colatitudes in radians. |
| `ring_phi_list` | `list` of `np.ndarray` or flat `np.ndarray [N_total]` | — | Longitudes in radians per ring. |
| `ring_counts` | `np.ndarray [R]` | — | Number of pixels per ring. |
| `lmax` | `int` | `24` | Maximum multipole $\ell_{\max}$. |
| `weights` | `np.ndarray [N_total]`, `None`, or `"uniform"` | `None` | Per-pixel quadrature weights. `None`: computed automatically. `"uniform"`: $w_i = 1/N_{\text{total}}$. Any array: used as-is (ignores `quadrature`). |
| `quadrature` | `str` | `"trapeze"` | Quadrature rule passed to `compute_weights`. Ignored when `weights` is not `None`. |
| `limit_range` | `float` | `1e10` | Overflow guard in the Legendre recurrence. |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `alm` | `torch.Tensor`, shape `[..., n_alm]`, `complex128` | Spherical harmonic coefficients. $n_{\text{alm}} = ({\ell_{\max}+1})(\ell_{\max}+2)/2$. |

#### Memory layout

Coefficients are stored in order of increasing $m$, then increasing $\ell$ within each $m$-block:

```
Index:   0          1          ...    lmax      lmax+1      lmax+2     ...
Coeff:   a_{0,0}    a_{1,0}    ...    a_{L,0}   a_{1,1}     a_{2,1}   ...
          m=0 block (L+1 entries)      m=1 block (L entries)
```

This layout matches `healpy.map2alm` exactly.

To extract $a_{\ell m}$:
```python
def alm_index(ell, m, lmax):
    offset = sum(lmax - mp + 1 for mp in range(m))
    return offset + (ell - m)

a_32 = alm[alm_index(3, 2, lmax)]   # a_{3,2}
```

### Notes

- The output is `complex128` regardless of the input dtype.
- A 1-D input `im` is treated as an unbatched map; the batch dimension is removed
  from the output automatically.
- `lmax` should not exceed $3 \times n_{\text{side}} - 1$ for HEALPix grids.

---

## `alm2map_latlon`

```python
im_out = alm2map_latlon(
    alm,
    ring_theta,
    ring_phi_list,
    ring_counts,
    lmax=24,
    limit_range=1e10,
)
```

Reconstruct a map from spherical harmonic coefficients (synthesis / inverse transform).

$$
f(\theta_r, \phi_j) = \text{Re}\!\left[
\sum_{m=0}^{\ell_{\max}} \tilde{F}_r(m) \, e^{im\phi_j}
\right], \quad
\tilde{F}_r(m) = \sum_{\ell=m}^{\ell_{\max}} a_{\ell m} \, Y_{\ell m}(\theta_r, 0)
$$

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `alm` | `torch.Tensor`, shape `[..., n_alm]`, complex | — | Coefficients in the layout produced by `map2alm_latlon`. |
| `ring_theta` | `np.ndarray [R]` | — | Colatitudes in radians of the **target** grid. |
| `ring_phi_list` | `list` of `np.ndarray` or flat `np.ndarray [N_total]` | — | Longitudes in radians of the target grid. |
| `ring_counts` | `np.ndarray [R]` | — | Number of pixels per ring of the target grid. |
| `lmax` | `int` | `24` | Maximum multipole used when computing the alm. |
| `limit_range` | `float` | `1e10` | Overflow guard in the Legendre recurrence. |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `im_out` | `torch.Tensor`, shape `[..., N_total]`, `float64` | Reconstructed map values in ring order. |

### Notes

- The target grid (defined by `ring_theta`, `ring_phi_list`, `ring_counts`) can
  differ from the analysis grid.  This allows, for example, analysing on an ERA5
  grid and synthesising on a finer regular grid.
- The synthesis uses a direct DFT in longitude ($O(N_r \cdot \ell_{\max})$ per ring).
  There is no FFT optimisation for the synthesis step in the current version.
- The output is a real tensor (imaginary parts from numerical noise are discarded).

---

## `anafast_latlon`

```python
cl = anafast_latlon(
    im,
    ring_theta,
    ring_phi_list,
    ring_counts,
    lmax=24,
    weights=None,
    quadrature="trapeze",
    limit_range=1e10,
)
```

Estimate the angular power spectrum $C_\ell$ from a map.

Internally calls `map2alm_latlon`, then accumulates:

$$
C_\ell = \frac{1}{2\ell+1}
\left[ |a_{\ell 0}|^2 + 2 \sum_{m=1}^{\ell} |a_{\ell m}|^2 \right]
$$

### Parameters

All parameters are identical to `map2alm_latlon`.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `im` | array-like or `torch.Tensor`, shape `[..., N_total]` | — | Map in ring order (apply `sort_idx` first). Batch dimension supported. |
| `ring_theta` | `np.ndarray [R]` | — | Colatitudes in radians. |
| `ring_phi_list` | `list` of `np.ndarray` or flat `np.ndarray [N_total]` | — | Longitudes in radians per ring. |
| `ring_counts` | `np.ndarray [R]` | — | Number of pixels per ring. |
| `lmax` | `int` | `24` | Maximum multipole. |
| `weights` | `np.ndarray [N_total]`, `None`, or `"uniform"` | `None` | Quadrature weights (same semantics as `map2alm_latlon`). |
| `quadrature` | `str` | `"trapeze"` | Quadrature rule (same semantics as `map2alm_latlon`). |
| `limit_range` | `float` | `1e10` | Overflow guard in the Legendre recurrence. |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `cl` | `torch.Tensor`, shape `[..., lmax+1]`, `float64` | Angular power spectrum. `cl[..., ell]` is $C_\ell$. |

### Notes

- `cl[0]` ($C_0$) is the square of the map mean, divided by 1.
- `cl[1]` ($C_1$) is the dipole power.
- For a Gaussian random field with theoretical spectrum $C_\ell^{\text{th}}$, the
  expected value of `cl[ell]` is $C_\ell^{\text{th}}$ (unbiased estimator, up to
  quadrature accuracy).
- Batch dimensions are preserved: `cl_batch.shape == im_batch.shape[:-1] + (lmax+1,)`.

---

## `grid_summary`

```python
info = grid_summary(ring_theta, ring_phi_list, ring_counts, lmax=24)
```

Print a human-readable diagnostic summary of the grid geometry and estimated
computational cost.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ring_theta` | `np.ndarray [R]` | — | Colatitudes in radians. |
| `ring_phi_list` | `list` of `np.ndarray` | — | Longitudes in radians per ring. |
| `ring_counts` | `np.ndarray [R]` | — | Number of pixels per ring. |
| `lmax` | `int` | `24` | Maximum multipole for the cost estimate. |

### Returns

| Name | Type | Description |
|------|------|-------------|
| `info` | `dict` | Dictionary with the same fields printed to stdout. |

#### `info` keys

| Key | Type | Description |
|-----|------|-------------|
| `"N_total"` | `int` | Total number of pixels. |
| `"n_rings"` | `int` | Number of iso-latitude rings $R$. |
| `"n_uniform"` | `int` | Number of rings whose longitudes are uniformly spaced (FFT path). |
| `"n_alm"` | `int` | Number of alm coefficients for the given `lmax`. |
| `"theta_min_deg"` | `float` | Colatitude of the northernmost ring, in degrees. |
| `"theta_max_deg"` | `float` | Colatitude of the southernmost ring, in degrees. |

### Example output

```
=== Grid summary  (lmax=192) ===
  Total pixels      : 786432
  Latitude rings    : 511
  Uniform rings     : 511/511  (FFT acceleration)
  θ range           : [0.110°, 179.890°]
  Pixels/ring       : min=4, max=512, mean=1538.0
  n_alm             : 18721
  Cost estimate     : 5.94e+06 ops  (FFT: 5.94e+06, DFT: 0.00e+00)
```

---

## Internal functions (not public API)

These functions are implementation details.  Their signatures may change between
versions.

| Function | Description |
|----------|-------------|
| `_compute_legendre_m(x, m, lmax, limit_range)` | Evaluate normalised associated Legendre polynomials $\tilde{P}_{\ell m}(\cos\theta)$ for $\ell = m \ldots \ell_{\max}$ via three-term recurrence in log-space. Returns shape `[lmax-m+1, R]`, scaled by $\sqrt{4\pi}$. |
| `_double_factorial_log(n)` | Return $\log(n!!)$. Used to seed the Legendre recurrence. |
| `_rfft2fft(v)` | Reconstruct the full $N$-point complex spectrum from `torch.fft.rfft` output (real input only). |
| `_check_uniform_phi(phi, tol)` | Return `(True, phi0)` if longitudes are uniformly spaced within relative tolerance `tol`, otherwise `(False, 0.0)`. |
| `_parse_phi_list(ring_phi_list, ring_counts)` | Accept `ring_phi_list` as a list of arrays or a single flat array, and return a list of float64 arrays. |
| `comp_tf_latlon(im, ring_phi_list, ring_counts, pixel_weights, mmax, cdtype)` | Per-ring weighted longitude DFT. Dispatches to FFT+phase-shift (uniform rings) or direct DFT (irregular rings). Returns shape `[..., R, mmax+1]`. |

---

## Error handling

| Situation | Behaviour |
|-----------|-----------|
| Unknown `convention` in `build_rings_from_latlon` | `ValueError` |
| Unknown `quadrature` in `compute_weights` | `ValueError` |
| Colatitudes do not match GL nodes with `quadrature="gauss_legendre"` | `UserWarning` (computation continues) |
| `im.ndim == 1` | Treated as `[1, N]`, batch dim removed from output |
| `alm.ndim == 1` in `alm2map_latlon` | Same: batch dim added then removed |

---

## Type reference

```python
# Input maps accept any of:
ArrayLike = Union[np.ndarray, torch.Tensor, List[float]]

# ring_phi_list accepts either:
List[np.ndarray]    # one array per ring  (output of build_rings_from_latlon)
np.ndarray          # flat [N_total] array (split internally by ring_counts)

# All returns:
ring_theta    : np.ndarray  float64  [R]
ring_counts   : np.ndarray  int64    [R]
sort_idx      : np.ndarray  int64    [N]
weights       : np.ndarray  float64  [N_total]
alm           : torch.Tensor  complex128  [..., n_alm]
cl            : torch.Tensor  float64     [..., lmax+1]
im_out        : torch.Tensor  float64     [..., N_total]
```
