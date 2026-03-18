# `HealPixDown` — HEALPix resolution reduction

**Module** `healpix_analyse.down`  
**Class** `HealPixDown`  
**Inherits** `torch.nn.Module`

---

## Overview

`HealPixDown` reduces the HEALPix resolution by a factor of 2:

$$n_{\text{side,out}} = n_{\text{side,in}} \mathbin{/} 2, \qquad N_{\text{out}} = N_{\text{in}} \mathbin{/} 4$$

Two strategies are available:

- **`"smooth"`** — Gaussian-weighted average over a disc.  Implemented as
  a sparse linear operator `M @ x` (differentiable, usable in autograd).
- **`"maxpool"`** — Non-linear maximum over the 4 direct NESTED children
  of each coarse pixel.  Faster, no gradient.

Both modes work on the full sphere or on a partial-sky subset defined by
`cell_ids`.

Input shapes `[N]` and `[B, N]` are both accepted; numpy arrays and torch
tensors are both accepted, and the return type mirrors the input type.

---

## Constructor

```python
HealPixDown(
    nside_in,
    mode        = "smooth",
    radius_deg  = None,
    sigma_deg   = None,
    weight_norm = "l1",
    cell_ids    = None,
    level       = None,
    device      = None,
    dtype       = torch.float32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside_in` | `int` | — | Input HEALPix resolution. Must be a power of 2 and ≥ 2. |
| `mode` | `str` | `"smooth"` | Downsampling strategy: `"smooth"` or `"maxpool"`. |
| `radius_deg` | `float` or `None` | auto | Angular radius of the Gaussian disc (degrees). Default ≈ 3 × pixel size. Used only with `"smooth"`. |
| `sigma_deg` | `float` or `None` | auto | Gaussian sigma (degrees). Default = `radius_deg / 2`. Used only with `"smooth"`. |
| `weight_norm` | `str` | `"l1"` | Per-output-pixel weight normalisation in `"smooth"` mode: `"l1"` (sum=1, preserves constants), `"l2"` (sum of squares=1, preserves energy), `"none"` (raw Gaussian). |
| `cell_ids` | array-like or `None` | `None` | Input pixel indices (NESTED ordering) for partial-sky operation. `None` = full sphere. |
| `level` | `int` or `None` | `None` | HEALPix level such that `nside_in = 2**level`. Required when `cell_ids` is provided. |
| `device` | device or `None` | auto | Torch device. Defaults to CUDA if available, else CPU. |
| `dtype` | `torch.dtype` | `float32` | Dtype for the sparse matrix values. |

---

## Forward

```python
y, cell_ids_out = down(x)
```

| Argument | Type | Shape | Description |
|---|---|---|---|
| `x` | numpy or torch | `[N]` or `[B, N]` | Input map(s) at fine resolution `nside_in`. `N` must equal `len(cell_ids)` when partial sky, or `12 * nside_in²` otherwise. |

| Return | Type | Shape | Description |
|---|---|---|---|
| `y` | same as `x` | `[N_out]` or `[B, N_out]` | Downsampled map(s) at coarse resolution `nside_out = nside_in // 2`. |
| `cell_ids_out` | `np.ndarray` | `[N_out]` | NESTED pixel indices of the output pixels at `nside_out`. |

---

## Modes

### `"smooth"` — Gaussian-weighted average

The output for coarse pixel `p_out` is:

$$y_{p_{\text{out}}} = \sum_{p_{\text{in}} \in \text{disc}} M_{p_{\text{out}}, p_{\text{in}}} \cdot x_{p_{\text{in}}}$$

where the weights are:

$$M_{p_{\text{out}}, p_{\text{in}}} \propto \exp\!\left(-\frac{\gamma^2}{2\sigma^2}\right) \cdot \mathbb{1}[\gamma \leq r]$$

and `γ` is the haversine angular distance between the centers of the
fine and coarse pixels.

The sparse matrix `M` is built once at construction and stored as a
buffer — `forward` only applies it.

**`weight_norm` options:**

| Value | Effect |
|---|---|
| `"l1"` *(default)* | Sum of weights = 1. A constant input map returns a constant output. Recommended for standard downsampling. |
| `"l2"` | Sum of squares = 1. Preserves local signal energy rather than amplitude. |
| `"none"` | Raw Gaussian weights (sum < 1 near boundaries). |

### `"maxpool"` — Maximum over NESTED children

In NESTED ordering, the 4 children of coarse pixel `p` are always
`4p, 4p+1, 4p+2, 4p+3`.  The output is:

$$y_{p_{\text{out}}} = \max(x_{4p},\; x_{4p+1},\; x_{4p+2},\; x_{4p+3})$$

This is non-differentiable and does not support partial sky smoothly
(missing children are replaced by the first available child).

---

## Partial-sky operation

When `cell_ids` is provided, the operator works on a subset of the
sphere.  The coarse output cell ids are automatically derived as the
unique parents of the input pixels in NESTED ordering:

```
cell_ids_out = unique(cell_ids_in // 4)
```

Some coarse pixels at the patch boundary may have fewer than 4 children
available.  In `"smooth"` mode, the Gaussian weights of absent fine
pixels are set to zero and the remaining weights are renormalized.

---

## Examples

### Full sphere

```python
import numpy as np
from healpix_analyse.down import HealPixDown

nside = 64
down  = HealPixDown(nside_in=nside)                      # "smooth", l1

# Single map
x = np.random.randn(12 * nside**2)                      # [N]
y, ids_out = down(x)
print(y.shape)       # (12288,) = 12 * 32²

# Batch
x_batch = np.random.randn(8, 12 * nside**2)             # [B, N]
y_batch, _ = down(x_batch)
print(y_batch.shape) # (8, 12288)
```

### Partial sky

```python
import healpy as hp
import numpy as np
from healpix_analyse.down import HealPixDown

nside = 64;  level = 6   # nside = 2**6
patch = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0.), np.radians(20.), nest=True)

down = HealPixDown(nside_in=nside, cell_ids=patch, level=level)

x_patch = np.random.randn(len(patch))
y_patch, coarse_ids = down(x_patch)
print(f"Fine   pixels: {len(patch)}")
print(f"Coarse pixels: {len(coarse_ids)}")    # ≈ len(patch) / 4
```

### Torch tensor, GPU

```python
import torch
from healpix_analyse.down import HealPixDown

nside = 128
down  = HealPixDown(nside_in=nside, device="cuda", dtype=torch.float32)

x = torch.randn(4, 12 * nside**2, device="cuda")        # [B, N]
y, ids_out = down(x)
print(y.shape, y.device)   # (4, 49152) cuda:0
```

### Maxpool mode

```python
from healpix_analyse.down import HealPixDown
import numpy as np

down_max = HealPixDown(nside_in=64, mode="maxpool")
x = np.random.randn(12 * 64**2)
y, _ = down_max(x)
```

### Custom Gaussian scale

```python
# Wider disc, sharper Gaussian — more blurring
down_wide = HealPixDown(
    nside_in=64,
    mode="smooth",
    radius_deg=2.0,
    sigma_deg=0.5,
    weight_norm="l1",
)
```

---

## Use in a U-Net encoder

```python
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown
import torch

nside = 64
conv1 = HealPixConv(nside,    in_channels=1,  out_channels=32, use_norm=True)
down1 = HealPixDown(nside,    mode="smooth")
conv2 = HealPixConv(nside//2, in_channels=32, out_channels=64, use_norm=True)
down2 = HealPixDown(nside//2, mode="smooth")

x = torch.randn(4, 1, 12 * nside**2)        # [B, 1, N]
f1 = conv1(x)                               # [B, 32, N]
f1_down, _ = down1(f1)                      # [B, 32, N/4]
f2 = conv2(f1_down)                         # [B, 64, N/4]
f2_down, _ = down2(f2)                      # [B, 64, N/16]
```

---

## Properties

| Attribute | Description |
|---|---|
| `nside_in` | Fine input resolution. |
| `nside_out` | Coarse output resolution (`nside_in // 2`). |
| `N_in` | Number of input pixels (full sphere or `len(cell_ids)`). |
| `N_out` | Number of output pixels. |
| `mode` | `"smooth"` or `"maxpool"`. |
| `partial` | `True` when `cell_ids` was provided. |
| `cell_ids_out` | Output pixel indices (coarse NESTED ids). |

---

## Pairing with `HealPixUp`

`HealPixDown(mode="smooth")` and `HealPixUp` are designed as a matched
pair.  The upsampling operator is the **adjoint (transpose)** of the
smooth downsampling matrix, with the same `weight_norm`, `radius_deg`,
and `sigma_deg`.  Use the same values in both constructors:

```python
from healpix_analyse.down import HealPixDown
from healpix_analyse.up   import HealPixUp

nside = 64
down = HealPixDown(nside_in=nside, weight_norm="l1")
up   = HealPixUp(nside_in=nside//2, weight_norm="l1", up_norm="col_l1")

x = torch.randn(4, 12 * nside**2)
y, _  = down(x)          # [4, N/4]
x_up, _ = up(y)          # [4, N]  ← approximate round-trip
```
