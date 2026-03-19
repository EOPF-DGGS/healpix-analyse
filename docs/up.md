# `HealPixUp` — HEALPix resolution increase

**Module** `healpix_analyse.up`  
**Class** `HealPixUp`  
**Inherits** `torch.nn.Module`

---

## Overview

`HealPixUp` increases the HEALPix resolution by a factor of 2:

$$n_{\text{side,out}} = n_{\text{side,in}} \times 2, \qquad N_{\text{out}} = N_{\text{in}} \times 4$$

The operation is defined as the **adjoint (transpose)** of the smooth
downsampling matrix from `HealPixDown`:

$$x_{\text{up}} = M^T \, x$$

where `M` is the `(N_coarse × N_fine)` Gaussian-weighted downsampling
matrix.  An optional diagonal normalisation corrects the amplitude after
the transpose.

Both modes work on the full sphere or on a partial-sky subset defined by
`cell_ids`.

Input shapes `[N]` and `[B, N]` are both accepted; numpy arrays and torch
tensors are both accepted, and the return type mirrors the input type.

---

## Constructor

```python
HealPixUp(
    nside_in,
    radius_deg  = None,
    sigma_deg   = None,
    weight_norm = "l1",
    up_norm     = "col_l1",
    eps         = 1e-12,
    cell_ids    = None,
    level       = None,
    device      = None,
    dtype       = torch.float32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside_in` | `int` | — | **Coarse** input resolution. Must be a power of 2 and ≥ 1. Output resolution will be `nside_in * 2`. |
| `radius_deg` | `float` or `None` | auto | Angular radius used to build the internal downsampling matrix (same semantics as `HealPixDown`). |
| `sigma_deg` | `float` or `None` | auto | Gaussian sigma (degrees). Default = `radius_deg / 2`. |
| `weight_norm` | `str` | `"l1"` | Normalisation used when building `M` (`"l1"`, `"l2"`, `"none"`). **Must match** the paired `HealPixDown`. |
| `up_norm` | `str` | `"col_l1"` | Diagonal correction applied after the transpose (see table below). |
| `eps` | `float` | `1e-12` | Floor for the denominator in normalisation, avoids division by zero. |
| `cell_ids` | array-like or `None` | `None` | **Coarse** pixel indices (NESTED) for partial-sky operation. `None` = full sphere. |
| `level` | `int` or `None` | `None` | HEALPix level such that `nside_in = 2**level`. Required when `cell_ids` is provided. |
| `device` | device or `None` | auto | Torch device. Defaults to CUDA if available, else CPU. |
| `dtype` | `torch.dtype` | `float32` | Dtype for the sparse matrix values. |

---

## Forward

```python
y, cell_ids_out = up(x)
```

| Argument | Type | Shape | Description |
|---|---|---|---|
| `x` | numpy or torch | `[N]` or `[B, N]` | Input map(s) at **coarse** resolution `nside_in`. `N` must equal `len(cell_ids)` when partial sky, or `12 * nside_in²` otherwise. |

| Return | Type | Shape | Description |
|---|---|---|---|
| `y` | same as `x` | `[N_out]` or `[B, N_out]` | Upsampled map(s) at fine resolution `nside_out = nside_in * 2`. |
| `cell_ids_out` | `np.ndarray` | `[N_out]` | NESTED pixel indices of the output fine pixels at `nside_out`. |

---

## The transpose operation

The paired `HealPixDown` with `mode="smooth"` computes:

$$y_{p_\text{out}} = \sum_{p_\text{in}} M_{p_\text{out}, p_\text{in}} \cdot x_{p_\text{in}}$$

`HealPixUp` computes:

$$x_{\text{up},\, p_\text{in}} = \sum_{p_\text{out}} M_{p_\text{out}, p_\text{in}} \cdot y_{p_\text{out}}$$

which is exactly `M^T y`.  `M^T` alone does not preserve the amplitude
of constant fields because a fine pixel that contributes to multiple coarse
pixels accumulates multiple counts.  The `up_norm` options correct for this.

---

## `up_norm` options

| `up_norm` | Formula | Effect |
|---|---|---|
| `"col_l1"` *(default)* | $x_{\text{up}} \mathbin{/} \text{col\_sum}$ where $\text{col\_sum}[i] = \sum_k M_{k,i}$ | Preserves constant fields when `weight_norm="l1"` was used in `HealPixDown`. **Recommended** for U-Net skip connections. |
| `"diag_l2"` | $x_{\text{up}} \mathbin{/} \text{diag}(M^T M)$ where $\text{diag}[i] = \sum_k M_{k,i}^2$ | Minimises the local L² approximation error. Use when `weight_norm="l2"`. |
| `"adjoint"` | $x_{\text{up}} = M^T y$ (no correction) | Raw adjoint. Amplitude is not corrected. Useful for debugging or when further processing follows. |

---

## Partial-sky operation

When `cell_ids` (coarse) is provided, the fine output pixels are the
4 NESTED children of each coarse input pixel:

```
cell_ids_out = unique(cell_ids_in[:, None] * 4 + [0, 1, 2, 3])
```

The internal downsampling matrix is built on the fine side
(`cell_ids_out`) and then transposed.

---

## Examples

### Full sphere

```python
import numpy as np
from healpix_analyse.up import HealPixUp

nside = 32   # coarse resolution
up    = HealPixUp(nside_in=nside)            # col_l1 normalisation

# Single map
x = np.random.randn(12 * nside**2)          # [N_coarse]
y, ids_fine = up(x)
print(y.shape)       # (49152,) = 12 * 64²

# Batch
x_batch = np.random.randn(8, 12 * nside**2) # [B, N_coarse]
y_batch, _ = up(x_batch)
print(y_batch.shape) # (8, 49152)
```

### Partial sky

```python
import healpy as hp
import numpy as np
from healpix_analyse.up import HealPixUp

nside_coarse = 32;  level_coarse = 5   # nside = 2**5
patch_coarse = hp.query_disc(
    nside_coarse, hp.ang2vec(np.pi/2, 0.), np.radians(20.), nest=True
)

up = HealPixUp(nside_in=nside_coarse, cell_ids=patch_coarse, level=level_coarse)

x_coarse = np.random.randn(len(patch_coarse))
y_fine, fine_ids = up(x_coarse)
print(f"Coarse pixels: {len(patch_coarse)}")
print(f"Fine   pixels: {len(fine_ids)}")    # ≈ 4 * len(patch_coarse)
```

### Torch tensor, GPU

```python
import torch
from healpix_analyse.up import HealPixUp

nside = 32
up = HealPixUp(nside_in=nside, device="cuda", dtype=torch.float32)

x = torch.randn(4, 12 * nside**2, device="cuda")   # [B, N_coarse]
y, _ = up(x)
print(y.shape, y.device)   # (4, 49152) cuda:0
```

### Different normalisation modes

```python
from healpix_analyse.up import HealPixUp
import numpy as np

nside = 32
x     = np.random.randn(12 * nside**2).astype(np.float32)

# Adjoint only (no amplitude correction)
up_adj  = HealPixUp(nside_in=nside, up_norm="adjoint")
y_adj, _ = up_adj(x)

# Least-squares diagonal preconditioner
up_l2   = HealPixUp(nside_in=nside, weight_norm="l2", up_norm="diag_l2")
y_l2, _ = up_l2(x)
```

---

## Use in a U-Net decoder

```python
from healpix_analyse.convol import HealPixConv
from healpix_analyse.up     import HealPixUp
import torch

nside = 64

# These match the encoder (down.py example)
up1  = HealPixUp(nside//2, up_norm="col_l1")
dec1 = HealPixConv(nside, 64 + 32, 32, kernel_sz=3, use_norm=True)

up2  = HealPixUp(nside//4, up_norm="col_l1")
dec2 = HealPixConv(nside//2, 128 + 64, 64, kernel_sz=3, use_norm=True)

# s1, s2 are skip connections from the encoder
xu2, _ = up2(bottleneck)                           # [B, 128, N/4]
xd2    = dec2(torch.cat([xu2, s2], dim=1))         # [B,  64, N/4]
xu1, _ = up1(xd2)                                  # [B,  64, N]
xd1    = dec1(torch.cat([xu1, s1], dim=1))         # [B,  32, N]
```

---

## Properties

| Attribute | Description |
|---|---|
| `nside_in` | Coarse input resolution. |
| `nside_out` | Fine output resolution (`nside_in * 2`). |
| `N_in` | Number of input pixels (full sphere or `len(cell_ids)`). |
| `N_out` | Number of output pixels (`4 * N_in` approximately). |
| `up_norm` | Active normalisation mode. |
| `partial` | `True` when `cell_ids` was provided. |
| `cell_ids_out` | Output fine pixel indices (NESTED ids at `nside_out`). |

---

## Round-trip accuracy

A `down → up` round-trip does not perfectly reconstruct the input — it
is an approximation limited by the Gaussian kernel width and the
normalisation mode.  To evaluate the round-trip quality:

```python
from healpix_analyse.down import HealPixDown
from healpix_analyse.up   import HealPixUp
import numpy as np

nside = 64
down = HealPixDown(nside_in=nside, weight_norm="l1")
up   = HealPixUp(nside_in=nside//2, weight_norm="l1", up_norm="col_l1")

x = np.random.randn(12 * nside**2).astype(np.float32)

y,    _ = down(x)       # coarse
x_rec, _ = up(y)        # back to fine

residual = x_rec - x
print(f"Max |residual|: {np.abs(residual).max():.3f}")
print(f"RMS  residual : {np.std(residual):.3f}")
```

For a white-noise input at `nside=64`, typical values are:
- `col_l1`  → RMS ≈ 0.5–0.7 (constant fields perfectly preserved)
- `diag_l2` → RMS ≈ 0.4–0.6 (better for smooth signals)
- `adjoint` → RMS ≈ 0.8–1.2 (no correction)
