# Installation

## Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/) (CPU or GPU)
- [healpix-geo](https://healpix-geo.readthedocs.io/)

## Install from GitHub

```bash
pip install git+https://github.com/EOPF-DGGS/healpix-analyse.git
```

## Install from source (development)

```bash
git clone git@github.com:EOPF-DGGS/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

To also install documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Using Pixi (recommended for development)

This project uses [Pixi](https://pixi.sh/) for reproducible environments:

```bash
pixi install
pixi run python -c "import healpix_analyse"
```

## Optional dependencies

Some modules require additional packages:

| Feature | Package |
|---|---|
| HEALPix pixel queries (`query_disc`, etc.) | `healpy` |
| Coordinate transformations | `pyproj` |
| Gaussian-grid resampling | `scipy` |
| Jupyter notebooks | `matplotlib`, `jupyter` |

## Verify the installation

```python
import healpix_analyse
print("healpix-analyse installed successfully")
```
