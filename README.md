# healpix-analyse

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://eopf-dggs.github.io/healpix-analyse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

A Python toolkit for analysing signals defined on HEALPix spherical grids,
with a focus on Earth Observation data. All operators are implemented in PyTorch
and are fully differentiable through `torch.autograd`.

**[Read the full documentation](https://eopf-dggs.github.io/healpix-analyse/)**

---

## Features

- **Spherical harmonic transforms** — local ALM coefficients, ring-based full-sky SHT (spin-0, 1, 2), power spectra
- **Gauge-equivariant convolution** — `HealPixConv` with configurable kernel size, gauge types, and number of gauges
- **Multi-resolution operators** — `HealPixDown` (smooth / max-pool) and `HealPixUp` (adjoint upsampling), NESTED ordering
- **Differentiable by default** — all hot-path operations are autograd-compatible
- **NumPy and Torch interoperability** — accepts both array types, returns the same type

## Package map

```
healpix_analyse/
├── alm.py               # Local complex spherical harmonic coefficients
├── alm_latlon.py         # SHT for arbitrary iso-latitude grids
├── healpix_sht.py        # Ring-based full-sky SHT for HEALPix
├── convol.py             # Gauge-equivariant spherical convolution (HealPixConv)
├── down.py               # Resolution reduction (HealPixDown)
├── up.py                 # Resolution increase (HealPixUp)
├── powerspectra.py        # Isotropic power spectrum on HEALPix patches
├── powerspectra_lonlat.py # Power spectrum on irregular lon/lat grids
├── healpix_interp.py      # Bilinear interpolation on HEALPix (NESTED)
├── make_rectangle.py      # Rectangular HEALPix patches from bounding boxes
├── resample.py            # Resample HEALPix onto regular lat/lon grids
└── ps.py                  # Power spectrum utilities
```

---

## Quick start

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

---

## Installation

```bash
pip install git+https://github.com/EOPF-DGGS/healpix-analyse.git
```

### From source (development)

```bash
git clone git@github.com:EOPF-DGGS/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

## Documentation

Full documentation is available at **[eopf-dggs.github.io/healpix-analyse](https://eopf-dggs.github.io/healpix-analyse/)**.

To build locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

## Relationship to healpix-geo and healpix-ai

- [healpix-geo](https://healpix-geo.readthedocs.io/) — HEALPix geometry: pixel coordinates, ellipsoids, coverage queries
- **healpix-analyse** — signal analysis: SHT, convolutions, power spectra, multi-resolution operators
- [healpix-ai](https://iaocea.github.io/healpix-ai/) — deep learning: autoencoders, U-Nets, forecasters built on top of `healpix-analyse`

## License

Apache 2.0 — see [LICENSE](LICENSE).
