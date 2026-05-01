# healpix-analyse: Spherical Analysis on HEALPix

`healpix-analyse` is a Python toolkit for analysing signals defined on HEALPix spherical grids,
with a focus on Earth Observation (EO) data. All operators are implemented in PyTorch and are
fully differentiable through `torch.autograd`.

## Why healpix-analyse?

Where [healpix-geo](https://healpix-geo.readthedocs.io/) focuses on **where** pixels are,
`healpix-analyse` focuses on **what you do** with the signal values stored in those pixels:
spherical harmonic transforms, power spectra, gauge-equivariant convolutions, and multi-resolution
up/downsampling operators.

## Install

::::{tab-set}

:::{tab-item} pip (from GitHub)

```bash
pip install git+https://github.com/EOPF-DGGS/healpix-analyse.git
```

:::

:::{tab-item} From source

```bash
git clone git@github.com:EOPF-DGGS/healpix-analyse.git
cd healpix-analyse
pip install -e .
```

:::

:::{tab-item} pixi

```bash
pixi install
```

:::

::::

## Start

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Overview
:link: overview
:link-type: doc

Package structure, design principles and quick example.
:::

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Requirements, install options and verification.
:::

:::{grid-item-card} API Reference
:link: autoapi/index
:link-type: doc

Auto-generated documentation of all classes and functions.
:::

:::{grid-item-card} Changelog
:link: changelog
:link-type: doc

Version history and release notes.
:::

::::

## Spherical harmonics

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Quickstart
:link: alm_latlon_1_quickstart
:link-type: doc

Get started with spherical harmonic transforms on arbitrary grids.
:::

:::{grid-item-card} Mathematics
:link: alm_latlon_2_mathematics
:link-type: doc

Conventions, quadrature rules, and mathematical details.
:::

:::{grid-item-card} API details
:link: alm_latlon_3_api
:link-type: doc

Full API reference for `alm_latlon`.
:::

::::

## Convolution & multi-resolution

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} HealPixConv
:link: convol_doc
:link-type: doc

Gauge-equivariant spherical convolution on HEALPix.
:::

:::{grid-item-card} HealPixDown
:link: down
:link-type: doc

Resolution reduction: smooth or max-pool downsampling.
:::

:::{grid-item-card} HealPixUp
:link: up
:link-type: doc

Resolution increase: adjoint of smooth downsampling.
:::

::::

## Resources

- {doc}`healpix_sht` - Ring-based full-sky SHT optimised for HEALPix
- {doc}`overview` - Design principles and package map
- {doc}`autoapi/index` - Full API reference

```{toctree}
---
maxdepth: 1
caption: Getting Started
hidden: true
---
installation
overview
```

```{toctree}
---
maxdepth: 2
caption: Spherical harmonics
hidden: true
---
alm_latlon_1_quickstart
alm_latlon_2_mathematics
alm_latlon_3_api
healpix_sht
```

```{toctree}
---
maxdepth: 2
caption: Convolution & multi-resolution
hidden: true
---
convol_doc
down
up
```

```{toctree}
---
maxdepth: 1
caption: API Reference
hidden: true
---
autoapi/index
```

```{toctree}
---
maxdepth: 1
caption: About
hidden: true
---
changelog
license
```
