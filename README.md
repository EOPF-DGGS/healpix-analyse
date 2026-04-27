# healpix-analyse

**healpix-analyse** is a Python toolkit for analysing signals defined on [HEALPix](https://healpix.sourceforge.io/) spherical grids, with a focus on Earth Observation (EO) data.

All operators are implemented in [PyTorch](https://pytorch.org/) and are fully differentiable through `torch.autograd`, making them suitable for deep learning pipelines as well as classical data analysis.

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://eopf-dggs.github.io/healpix-analyse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

---

## Features

- **Spherical Harmonic Transforms** — ring-based full-sky SHT (`healpix_sht`) and SHT for arbitrary iso-latitude grids including ERA5, regular lat/lon, and HEALPix (`alm_latlon`)
- **Power spectra** — isotropic 1D angular power spectrum on HEALPix patches and lon/lat grids
- **Gauge-equivariant convolution** — learned spherical convolution on full-sky or partial-sky HEALPix maps (`HealPixConv`)
- **Multi-resolution operators** — smooth Gaussian downsampling (`HealPixDown`) and its adjoint upsampling (`HealPixUp`), usable in U-Net-style architectures
- **Flat-sky localized SHT** — memory-efficient approximation for ultra-high resolution patches (`LocalizedFlatSkyAlm`)
- **Numpy & Torch interoperability** — every operator accepts both `np.ndarray` and `torch.Tensor`, with shapes `[N]` and `[B, N]`

---

## Installation

```bash
pip install healpix-analyse
```

### From source

```bash
git clone git@github.com:EOPF-DGGS/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

### With Pixi (recommended for development)

```bash
pixi install
```

---

## Quick start

### Angular power spectrum on a HEALPix map

```python
import numpy as np
import healpy as hp
from healpix_analyse.alm_latlon import build_rings_from_latlon, anafast_latlon

nside = 64
npix  = 12 * nside**2
lmax  = 3 * nside

im = np.random.randn(npix)

theta, phi = hp.pix2ang(nside, np.arange(npix))
ring_theta, ring_phi_list, ring_counts, sort_idx = build_rings_from_latlon(
    theta, phi, convention="colatitude_rad"
)

cl = anafast_latlon(
    im[sort_idx], ring_theta, ring_phi_list, ring_counts,
    lmax=lmax, quadrature="equal_area",
)
# cl is a torch.Tensor of shape [lmax+1]
```

### HEALPix downsampling in a neural network

```python
import torch
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown

nside = 64
conv  = HealPixConv(nside, in_channels=1, out_channels=32, use_norm=True)
down  = HealPixDown(nside, mode="smooth")

x = torch.randn(4, 1, 12 * nside**2)   # [batch, channels, pixels]
features, _  = conv(x)                  # [4, 32, N]
coarse,   _  = down(features)           # [4, 32, N/4]
```

### Full-sky SHT with spin support (CMB polarisation)

```python
from healpix_analyse.healpix_sht import HEALPixSHT
import torch

nside = 128
sht   = HEALPixSHT(nside, lmax=3*nside, device="cpu")

im    = torch.randn(12 * nside**2)
alm   = sht.map2alm(im)          # analysis
im_rec = sht.alm2map(alm)        # synthesis
cl    = sht.anafast(im)           # power spectrum
```

---

## Package overview

| Module | Description |
|---|---|
| `healpix_analyse.alm_latlon` | SHT for arbitrary iso-latitude grids (ERA5, lat/lon, HEALPix) |
| `healpix_analyse.healpix_sht` | Ring-based optimised full-sky SHT (spin-0, spin-1, spin-2) |
| `healpix_analyse.convol` | Gauge-equivariant spherical convolution (`HealPixConv`) |
| `healpix_analyse.down` | HEALPix resolution reduction (`HealPixDown`) |
| `healpix_analyse.up` | HEALPix resolution increase (`HealPixUp`) |
| `healpix_analyse.powerspectra` | Isotropic 1D power spectrum on HEALPix patches |
| `healpix_analyse.powerspectra_lonlat` | Power spectrum on lon/lat grids |
| `healpix_analyse.healpix_interp` | Bilinear interpolation on HEALPix (NESTED) |
| `healpix_analyse.make_rectangle` | Rectangular HEALPix patches from bounding boxes |
| `healpix_analyse.resample` | Resample HEALPix data onto regular lat/lon grids |

---

## Documentation

Full documentation is available at **https://eopf-dggs.github.io/healpix-analyse/**

---

## Dependencies

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [NumPy](https://numpy.org/) ≥ 2.0
- [healpix-geo](https://healpix-geo.readthedocs.io/)

---

## Authors

- Jean-Marc Delouis
- Tina Odaka

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
