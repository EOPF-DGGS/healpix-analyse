# `alm_latlon` — Part 1: Quickstart

> **Module** `healpix_analyse.alm_latlon`  
> **Dependencies** `numpy`, `torch` only — no HEALPix, no FOSCAT  
> **Parts** [1 · Quickstart](#) · [2 · Mathematics](alm_latlon_2_mathematics.md) · [3 · API reference](alm_latlon_3_api.md)

---

## Installation

`alm_latlon` ships as a single file inside the `healpix_analyse` package.
No extra dependency beyond `numpy` and `torch` is required.

```python
from healpix_analyse.alm_latlon import (
    build_rings_from_latlon,
    compute_weights,
    map2alm_latlon,
    alm2map_latlon,
    anafast_latlon,
    grid_summary,
)
```

---

## The four-line workflow

Every use of `alm_latlon` follows the same pattern regardless of the grid type:

```
1.  build_rings_from_latlon   →  ring_theta, ring_phi_list, ring_counts, sort_idx
2.  im_sorted = im[sort_idx]  →  reorder your map into ring order
3.  map2alm_latlon  /  anafast_latlon  →  alm  /  Cl
4.  alm2map_latlon  (optional)  →  reconstructed map
```

---

## Example 1 — HEALPix pixels (validation against healpy)

The simplest way to check correctness is to use HEALPix pixel coordinates.

```python
import numpy as np
import healpy as hp
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

nside = 64
npix  = 12 * nside**2
lmax  = 3 * nside   # same default as healpy

# Simulate a random map
np.random.seed(0)
im = np.random.randn(npix)

# Step 1 — build rings from HEALPix pixel coordinates (ring order)
theta, phi = hp.pix2ang(nside, np.arange(npix))   # colatitude, longitude in radians
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi,
    convention="colatitude_rad"   # default, no conversion needed
)

# Step 2 — compute Cl
# HEALPix is an equal-area grid → use quadrature="equal_area"
cl = anafast_latlon(
    im[sort_idx],
    ring_theta, ring_phi_list, ring_counts,
    lmax=lmax,
    quadrature="equal_area",
)

# Reference
cl_hp = hp.anafast(im, lmax=lmax)

# Both Cl arrays should agree to within numerical precision
import matplotlib.pyplot as plt
ell = np.arange(lmax + 1)
plt.loglog(ell[2:], cl.numpy()[2:],  label="anafast_latlon")
plt.loglog(ell[2:], cl_hp[2:],       label="healpy.anafast", ls="--")
plt.xlabel("ℓ"); plt.ylabel("Cℓ"); plt.legend(); plt.show()
```

---

## Example 2 — ERA5 / ECMWF Gaussian grid

ERA5 data comes on a **Gaussian grid** where latitudes are the zeros of a
Legendre polynomial.  The `"gauss_legendre"` quadrature rule must be used —
any other rule introduces oscillations at high ℓ (see Part 2 for the reason).

```python
import numpy as np
import xarray as xr
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

# Load an ERA5 field (e.g. surface temperature, single level)
ds  = xr.open_dataset("era5_t2m.nc")
t2m = ds["t2m"].values.ravel()          # shape [N]
lat = ds["latitude"].values             # 1-D, geographic degrees [-90, +90]
lon = ds["longitude"].values            # 1-D, degrees [0, 360]

# Broadcast to pixel arrays if the field is on a 2-D (lat × lon) grid
lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
lat_flat = lat2d.ravel()
lon_flat = lon2d.ravel()

# Step 1 — build rings
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    lat_flat, lon_flat,
    convention="geographic_deg"   # ERA5 uses geographic degrees
)

# Step 2 — compute Cl with Gauss-Legendre quadrature
lmax = 2 * len(ring_theta)   # safe upper bound for a Gaussian grid
cl = anafast_latlon(
    t2m[sort_idx],
    ring_theta, ring_phi_list, ring_counts,
    lmax=lmax,
    quadrature="gauss_legendre",   # mandatory for Gaussian grids
)

ell = np.arange(lmax + 1)
import matplotlib.pyplot as plt
plt.loglog(ell[2:], cl.numpy()[2:])
plt.xlabel("ℓ"); plt.ylabel("Cℓ  [K²]"); plt.title("ERA5 T2m power spectrum")
plt.show()
```

---

## Example 3 — Regular lat/lon grid

A grid with equally-spaced latitudes and longitudes (e.g. from a global
atmospheric model output on a regular 0.5° grid).

```python
import numpy as np
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

# Build a regular 1° global grid
dlat, dlon = 1.0, 1.0
lat_1d = np.arange(-90 + dlat/2,  90, dlat)   # geographic degrees, N→S
lon_1d = np.arange(0,            360, dlon)   # degrees, 0→360

lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
lat_flat = lat2d.ravel()
lon_flat = lon2d.ravel()

np.random.seed(1)
im = np.random.randn(len(lat_flat))

ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    lat_flat, lon_flat,
    convention="geographic_deg"
)

# Trapezoidal rule is fine for a regular grid
lmax = 180
cl = anafast_latlon(
    im[sort_idx],
    ring_theta, ring_phi_list, ring_counts,
    lmax=lmax,
    quadrature="trapeze",
)
print(cl.shape)   # torch.Size([181])
```

---

## Example 4 — Getting the alm coefficients

When you need the individual alm coefficients (e.g. for filtering, cross-spectra,
or further processing):

```python
from healpix_analyse.alm_latlon import map2alm_latlon
import numpy as np, healpy as hp

nside = 32
im    = np.random.randn(12 * nside**2)
lmax  = 3 * nside

theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

alm = map2alm_latlon(
    im[sort_idx],
    ring_theta, ring_phi_list, ring_counts,
    lmax=lmax,
    quadrature="equal_area",
)

# alm is a complex128 torch.Tensor of shape [n_alm]
# n_alm = sum(lmax - m + 1 for m in range(lmax+1))
print(alm.shape)        # torch.Size([n_alm])
print(alm.dtype)        # torch.complex128
print(alm[:3])          # a_00, a_10, a_20  (m=0 block)
```

---

## Example 5 — Round-trip analysis / synthesis

`alm2map_latlon` reconstructs the map from its alm coefficients.  The residual
measures the truncation error introduced by the finite `lmax`.

```python
from healpix_analyse.alm_latlon import map2alm_latlon, alm2map_latlon
import numpy as np, healpy as hp

nside = 32
lmax  = 3 * nside
im    = np.random.randn(12 * nside**2)

theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

im_sorted = im[sort_idx]

# Analysis
alm = map2alm_latlon(
    im_sorted, ring_theta, ring_phi_list, ring_counts,
    lmax=lmax, quadrature="equal_area"
)

# Synthesis (same grid)
im_rec = alm2map_latlon(
    alm, ring_theta, ring_phi_list, ring_counts, lmax=lmax
)

residual = (im_rec.numpy() - im_sorted)
print(f"Max |residual| : {np.abs(residual).max():.2e}")
print(f"RMS  residual  : {np.std(residual):.2e}")
```

---

## Example 6 — Batch maps

All transform functions support a leading batch dimension, so you can process
many maps in a single call:

```python
import numpy as np, healpy as hp
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

nside  = 64
B      = 50    # number of maps
lmax   = 3 * nside
npix   = 12 * nside**2

# Batch of B independent maps, shape [B, npix]
im_batch = np.random.randn(B, npix)

theta, phi = hp.pix2ang(nside, np.arange(npix))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

# Apply sort_idx along the pixel axis
cl_batch = anafast_latlon(
    im_batch[:, sort_idx],
    ring_theta, ring_phi_list, ring_counts,
    lmax=lmax,
    quadrature="equal_area",
)
print(cl_batch.shape)   # torch.Size([50, lmax+1])

# Mean spectrum across realisations
cl_mean = cl_batch.mean(dim=0)
```

---

## Example 7 — Inspecting the grid

`grid_summary` prints a diagnostic before committing to a long computation:

```python
from healpix_analyse.alm_latlon import build_rings_from_latlon, grid_summary
import numpy as np, healpy as hp

nside = 128
theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2))
ring_theta, ring_phi_list, ring_counts, _ = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

info = grid_summary(ring_theta, ring_phi_list, ring_counts, lmax=3*nside)
```

Example output:
```
=== Grid summary  (lmax=384) ===
  Total pixels      : 196608
  Latitude rings    : 511
  Uniform rings     : 511/511  (FFT acceleration)
  θ range           : [0.110°, 179.890°]
  Pixels/ring       : min=4, max=512, mean=384.8
  n_alm             : 74305
  Cost estimate     : 1.50e+06 ops  (FFT: 1.50e+06, DFT: 0.00e+00)
```

---

## Choosing the right `quadrature`

| Grid type | `quadrature` | Notes |
|-----------|-------------|-------|
| HEALPix (any `nside`) | `"equal_area"` | All pixels have the same area. |
| Regular lat/lon (equidistant θ) | `"trapeze"` | Trapezoidal rule works well. |
| **ERA5, ECMWF, IFS, ARPEGE, GFS** | `"gauss_legendre"` | Mandatory — eliminates Cℓ oscillations at high ℓ. |
| Custom explicit weights | pass `weights=` array | `quadrature` is ignored. |

> **Symptom of wrong quadrature:** regular oscillations in Cℓ that grow
> towards high ℓ.  If you see this pattern, switch to `"gauss_legendre"`.

---

## Coordinate convention quick reference

`build_rings_from_latlon` accepts four input conventions via the `convention` argument:

| `convention` | `lat` | `lon` |
|---|---|---|
| `"colatitude_rad"` *(default)* | colatitude θ, radians, 0 → π | longitude φ, radians, 0 → 2π |
| `"colatitude_deg"` | colatitude θ, degrees, 0° → 180° | longitude φ, degrees, 0° → 360° |
| `"geographic_rad"` | geographic latitude, radians, −π/2 → +π/2 | longitude, radians, −π → +π or 0 → 2π |
| `"geographic_deg"` | geographic latitude, degrees, −90° → +90° | longitude, degrees, −180° → +180° or 0° → 360° |

All conventions convert internally to colatitude + longitude in radians.
