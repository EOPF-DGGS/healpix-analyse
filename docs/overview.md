# Overview

**healpix-analyse** provides a set of tools for analysing signals defined on
HEALPix spherical grids, with a focus on Earth Observation (EO) data.

All operators are implemented in PyTorch and are fully differentiable through
`torch.autograd`, making them suitable for deep learning pipelines as well as
classical data analysis.

## Package structure

| Module | Description |
|---|---|
| `healpix_analyse.alm` | Local complex spherical harmonic coefficients (`AlmCoeffs`, `AlmTransform`) |
| `healpix_analyse.alm_latlon` | SHT for arbitrary iso-latitude grids (ERA5, regular lat/lon, HEALPix) |
| `healpix_analyse.healpix_sht` | Ring-based full-sky SHT optimised for HEALPix (spin-0, spin-1, spin-2) |
| `healpix_analyse.convol` | Gauge-equivariant spherical convolution (`HealPixConv`) |
| `healpix_analyse.down` | HEALPix resolution reduction — smooth or max-pool (`HealPixDown`) |
| `healpix_analyse.up` | HEALPix resolution increase — adjoint of smooth downsampling (`HealPixUp`) |
| `healpix_analyse.powerspectra` | Isotropic 1D power spectrum on HEALPix patches |
| `healpix_analyse.powerspectra_lonlat` | Power spectrum on irregular lon/lat grids |
| `healpix_analyse.healpix_interp` | Bilinear interpolation on HEALPix (NESTED) |
| `healpix_analyse.make_rectangle` | Build rectangular HEALPix patches from bounding boxes |
| `healpix_analyse.resample` | Resample HEALPix data onto regular lat/lon grids |

## Design principles

**Differentiable by default.** All hot-path operations (`torch.fft`, `einsum`,
`index_select`, sparse matrix-vector products) are supported by PyTorch autograd.
Geometry tables (Legendre polynomials, interpolation weights, phase matrices) are
precomputed once and stored as non-gradient buffers.

**Numpy and Torch interoperability.** Every operator accepts both `np.ndarray`
and `torch.Tensor` inputs and returns the same type. Shape `[N]` (single map)
and `[B, N]` (batch of maps) are both supported throughout.

**Full-sky and partial-sky.** Operators like `HealPixDown`, `HealPixUp`, and
`HealPixConv` work on the full sphere or on arbitrary partial-sky patches defined
by a set of NESTED pixel indices (`cell_ids`).

**Consistent mathematical conventions.** All SHT modules follow the standard
orthonormal convention (identical to `healpy`):

$$a_{\ell m} = \int f(\theta, \varphi)\, Y_{\ell m}^*(\theta, \varphi)\, d\Omega$$

$$C_\ell = \frac{1}{2\ell+1} \left[ |a_{\ell 0}|^2 + 2 \sum_{m=1}^{\ell} |a_{\ell m}|^2 \right]$$

## Quick example

```python
import numpy as np
import healpy as hp
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

nside = 64
npix  = 12 * nside**2
lmax  = 3 * nside

# Random test map
im = np.random.randn(npix)

# Build ring structure from HEALPix coordinates
theta, phi = hp.pix2ang(nside, np.arange(npix))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

# Compute angular power spectrum
cl = anafast_latlon(
    im[sort_idx], ring_theta, ring_phi_list, ring_counts,
    lmax=lmax, quadrature="equal_area",
)

print(cl.shape)   # torch.Size([193])
```

## Relationship to `healpix-geo`

`healpix-analyse` builds on top of
[healpix-geo](https://healpix-geo.readthedocs.io/) for pixel coordinate
conversions and ellipsoidal geometry. Where `healpix-geo` focuses on
**where** pixels are, `healpix-analyse` focuses on **what you do** with
the signal values stored in those pixels.
