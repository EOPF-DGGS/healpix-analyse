# `HealPixConv` — Gauge-equivariant spherical convolution

**Module** `healpix_analyse.convol`  
**Class** `HealPixConv`  
**Inherits** `torch.nn.Module`

---

## Overview

`HealPixConv` applies a spatial convolution to a HEALPix map in a
**gauge-equivariant** way.  The core idea is that the kernel is defined
once as a regular `kernel_sz × kernel_sz` grid at the North Pole, then
**rotated** to each target pixel by a gauge rotation matrix.  This gives
every pixel a geometrically consistent kernel orientation, independent of
HEALPix topology.

### Three precomputed stages

**A — Geometry**

A `P = kernel_sz²` grid of unit vectors is placed at the North Pole,
with angular spacing equal to one HEALPix pixel size (`hp.nside2resol`).
For each target pixel `k` at colatitude `θ_k` and longitude `φ_k`, the
full rotation matrix is:

$$R_{\text{total}}(k, g) = R_{\text{gauge}}(\alpha_g) \cdot R_z(\varphi_k) \cdot R_y(\theta_k)$$

The gauge angle for orientation `g` is:

$$\alpha_g = \alpha_{\text{base}}(k) + \text{sign}(k) \cdot g \cdot \frac{\pi}{G}$$

where `alpha_base` and `sign` depend on `gauge_type` (see table below).

**B — Binding**

For each of the `K×G×P` rotated stencil points,
`healpy.get_interp_weights` returns 4 bilinear-interpolation neighbours
and weights.  For partial-sky patches, out-of-domain neighbours are
zeroed and remaining weights renormalized to 1.  Stencil points with
zero total weight fall back to the center pixel.

**C — Convolution**

$$y_{b,\, g \cdot C_{out}+o,\, k} = \text{bias}_{g C_{out}+o} + \sum_{c=0}^{C_{in}-1} \sum_{p=0}^{P-1} W_{g,c,o,p} \cdot x^{\text{interp}}_{b,c,k,p}$$

---

## Gauge types

| `gauge_type` | `alpha_base` | `sign` on `g_shifts` | Singularities | Best for |
|---|---|---|---|---|
| `"phi"` | 0 | +1 everywhere | 2 geographic poles | NWP, Earth observation |
| `"cosmo"` | `−φ` (North) / `+φ` (South) | +1 North / −1 South | Equatorial discontinuity | CMB, cosmology (Delouis et al. 2022) |
| `"projected_ref"` | `atan2(r·eφ, r·eθ)` | +1 everywhere | 2 antipodal points (chosen via `ref_direction`) | Any domain — optimal smoothness |

### `"projected_ref"` gauge — hairy ball theorem

The hairy ball theorem guarantees at least **2 singularities** on any
continuous tangent-vector field on S².  The `"projected_ref"` gauge lets
you **choose** where they fall.

A fixed 3-D unit vector **r** is projected onto the local tangent plane
of each pixel:

$$\vec{v}_{\text{proj}} = \hat{r} - (\hat{r} \cdot \hat{n})\,\hat{n}, \qquad \alpha_{\text{base}} = \text{atan2}(\vec{v}_{\text{proj}} \cdot \hat{e}_\varphi,\; \vec{v}_{\text{proj}} \cdot \hat{e}_\theta)$$

Singularities appear only at the two antipodal pixels where
$\hat{r} \parallel \hat{n}$.

| `ref_direction` | Singularity locations |
|---|---|
| `[0, 0, -1]` | Geographic poles (same as `"phi"`) |
| `[1, 0, 0]` *(default)* | Equator at φ = 0° and φ = 180° |
| `[0, 1, 0]` | Equator at φ = 90° and φ = 270° |
| Any unit vector | The pixel nearest to **r** and its antipode |

---

## Input / output shapes

| Input | Output | Notes |
|---|---|---|
| `[N]` | `[N]` | Only when `G = C_out = 1` |
| `[N]` | `[G*C_out, N]` | When `G > 1` or `C_out > 1` |
| `[B, N]` | `[B, G*C_out, N]` | Single-channel batch |
| `[B, C_in, N]` | `[B, G*C_out, N]` | Multi-channel batch |

The return type mirrors the input type (numpy → numpy, torch → torch).

---

## Constructor

```python
HealPixConv(
    nside,
    in_channels,
    out_channels,
    kernel_sz     = 3,
    n_gauges      = 1,
    gauge_type    = "phi",
    ref_direction = None,
    cell_ids      = None,
    level         = None,
    nest          = True,
    use_norm      = False,
    device        = None,
    dtype         = torch.float32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside` | `int` | — | HEALPix resolution (power of 2). |
| `in_channels` | `int` | — | Number of input channels `C_in`. |
| `out_channels` | `int` | — | Output channels per gauge `C_out`. Total = `G × C_out`. |
| `kernel_sz` | `int` | `3` | Odd integer ≥ 1. `P = kernel_sz²` stencil points. |
| `n_gauges` | `int` | `1` | Number of gauge orientations `G`. |
| `gauge_type` | `str` | `"phi"` | `"phi"`, `"cosmo"`, or `"projected_ref"`. |
| `ref_direction` | array-like or `None` | `[1,0,0]` | Reference vector for `"projected_ref"` gauge. Ignored otherwise. |
| `cell_ids` | array-like or `None` | `None` | Pixel indices (NESTED) for partial sky. `None` = full sphere. |
| `level` | `int` or `None` | `None` | `nside = 2**level`. Required with `cell_ids`. |
| `nest` | `bool` | `True` | NESTED pixel ordering. |
| `use_norm` | `bool` | `False` | GroupNorm + ReLU after convolution. |
| `device` | device or `None` | auto | Torch device. |
| `dtype` | `torch.dtype` | `float32` | Dtype for kernel parameters. |

---

## Learnable attributes

| Attribute | Shape | Description |
|---|---|---|
| `weight` | `[G, C_in, C_out, P]` | Spatial + channel kernel. Kaiming uniform init. |
| `bias` | `[G * C_out]` | Per-output-channel bias. Zero init. |

---

## `set_kernel` — fixed filters

```python
conv.set_kernel(W, bias=None, requires_grad=False)
```

Replace the learnable kernel with a fixed array.

| Parameter | Type | Description |
|---|---|---|
| `W` | array-like | Shape `[C_in, C_out, P]` (broadcast over all G gauges) or `[G, C_in, C_out, P]` (per-gauge). |
| `bias` | array-like or `None` | Shape `[G * C_out]`. `None` resets to zero. |
| `requires_grad` | `bool` | `False` freezes the kernel; `True` allows fine-tuning. |

**Returns** `self` (chainable).

### Stencil point index reference

For `kernel_sz=3`, the 9 stencil points `p=0..8` correspond to the
following positions in the **gauge-rotated** local frame.  With
`gauge_type="phi"` and North-Pole default orientation:

```
p:   0   1   2
     3   4   5      ← p=4 is always the CENTER pixel
     6   7   8
```

> The geometric meaning of each `p` rotates continuously across the
> sphere according to the chosen gauge.  Position labels (N/S/E/W) are
> approximate guides valid near the North-Pole alignment direction.

---

## Case 1 — Learned kernels: U-Net on the sphere

### Architecture

```
Input [B, C_in, N]
  │
  ├─ HealPixConv  nside,    C_in→32,  use_norm=True  ──→ skip s1 [B, 32, N]
  │
  ├─ HealPixDown  nside→nside/2
  │
  ├─ HealPixConv  nside/2,  32→64,    use_norm=True  ──→ skip s2 [B, 64, N/4]
  │
  ├─ HealPixDown  nside/2→nside/4
  │
  ├─ HealPixConv  nside/4,  64→128   (bottleneck)
  │
  ├─ HealPixUp    nside/4→nside/2
  │  cat(s2) → [B, 192, N/4]
  ├─ HealPixConv  nside/2,  192→64
  │
  ├─ HealPixUp    nside/2→nside
  │  cat(s1) → [B, 96, N]
  ├─ HealPixConv  nside,    96→32
  │
  └─ HealPixConv  nside,    32→C_out, kernel_sz=1   (output head)
     │
     Output [B, C_out, N]
```

### Complete example

```python
import torch
import torch.nn as nn
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown
from healpix_analyse.up     import HealPixUp


class SphereUNet(nn.Module):
    """2-level spherical U-Net on the full HEALPix sphere."""

    def __init__(self, nside, in_channels, out_channels):
        super().__init__()

        # Encoder
        self.enc1  = HealPixConv(nside,    in_channels, 32,  kernel_sz=3, use_norm=True)
        self.down1 = HealPixDown(nside,    mode="smooth")
        self.enc2  = HealPixConv(nside//2, 32,          64,  kernel_sz=3, use_norm=True)
        self.down2 = HealPixDown(nside//2, mode="smooth")

        # Bottleneck
        self.bottle = HealPixConv(nside//4, 64, 128, kernel_sz=3, use_norm=True)

        # Decoder
        self.up2  = HealPixUp(nside//4)
        self.dec2 = HealPixConv(nside//2, 128 + 64, 64, kernel_sz=3, use_norm=True)
        self.up1  = HealPixUp(nside//2)
        self.dec1 = HealPixConv(nside,    64  + 32, 32, kernel_sz=3, use_norm=True)

        # Output head: 1×1 conv = pure channel projection
        self.head = HealPixConv(nside, 32, out_channels, kernel_sz=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)                                       # [B, 32, N]
        x2, _ = self.down1(s1)                                  # [B, 32, N/4]
        s2 = self.enc2(x2)                                      # [B, 64, N/4]
        x3, _ = self.down2(s2)                                  # [B, 64, N/16]

        # Bottleneck
        xb = self.bottle(x3)                                    # [B, 128, N/16]

        # Decoder with skip connections
        xu2, _ = self.up2(xb)                                   # [B, 128, N/4]
        xd2 = self.dec2(torch.cat([xu2, s2], dim=1))            # [B, 64, N/4]
        xu1, _ = self.up1(xd2)                                  # [B, 64, N]
        xd1 = self.dec1(torch.cat([xu1, s1], dim=1))            # [B, 32, N]

        return self.head(xd1)                                   # [B, C_out, N]


# ── run ──────────────────────────────────────────────────────────────────
nside = 64
model = SphereUNet(nside=nside, in_channels=2, out_channels=1)
x = torch.randn(4, 2, 12 * nside**2)
y = model(x)
print(f"Input  {tuple(x.shape)}")    # (4, 2, 49152)
print(f"Output {tuple(y.shape)}")    # (4, 1, 49152)
print(f"Params {sum(p.numel() for p in model.parameters()):,}")
```

### Multi-gauge variant

```python
# 4 gauges: kernel orientation rotated by 0°, 45°, 90°, 135° per pixel
conv = HealPixConv(nside=64, in_channels=2, out_channels=8,
                   kernel_sz=3, n_gauges=4, use_norm=True)
x = torch.randn(4, 2, 12 * 64**2)
y = conv(x)
print(y.shape)    # (4, 32, 49152)   — G * C_out = 4 * 8
```

### Accessing learned weights

```python
# After training, weight has shape [G, C_in, C_out, P]
W = model.enc1.weight.detach().cpu().numpy()   # [1, 1, 32, 9]
# Kernel for gauge 0, input ch 0, output ch 5, all 9 stencil points:
print(W[0, 0, 5, :])
```

### Training loop

```python
import torch.optim as optim

opt = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(200):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x_batch), y_true)
    loss.backward()
    opt.step()
```

---

## Case 2 — Fixed kernels: hand-crafted filters

Call `conv.set_kernel(W)` to replace the learnable kernel with a fixed
array and freeze gradients.

### Example A — Isotropic Gaussian smoothing

```python
import numpy as np
from healpix_analyse.convol import HealPixConv

nside = 64
conv  = HealPixConv(nside=nside, in_channels=1, out_channels=1)

sigma    = 1.0           # width in pixel-size units
w_center = 1.0           # exp(-0 / 2σ²)
w_ring   = np.exp(-1.0 / (2 * sigma**2))

W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]              = w_center    # center (p=4)
W[0, 0, [0,1,2,3,5,6,7,8]] = w_ring  # 8 neighbours
W /= W.sum()                          # normalise

conv.set_kernel(W)

sky        = np.random.randn(12 * nside**2).astype(np.float32)
sky_smooth = conv(sky)    # np.ndarray [N]
```

### Example B — Discrete Laplacian

```python
W_lap = np.zeros((1, 1, 9), dtype=np.float32)
W_lap[0, 0, 4]                 =  8.0   # center
W_lap[0, 0, [0,1,2,3,5,6,7,8]] = -1.0  # 8 neighbours

conv_lap = HealPixConv(nside=nside, in_channels=1, out_channels=1)
conv_lap.set_kernel(W_lap)
edges = conv_lap(sky)
```

### Example C — Approximate directional gradients

```python
# With gauge_type="phi", the North-Pole grid approximates:
#   p = 0 1 2       NW  N  NE
#       3 4 5   =    W ctr  E
#       6 7 8       SW  S  SE

W_grad = np.zeros((1, 2, 9), dtype=np.float32)
W_grad[0, 0, 1] = +1.0;  W_grad[0, 0, 7] = -1.0   # ch0: N−S
W_grad[0, 1, 5] = +1.0;  W_grad[0, 1, 3] = -1.0   # ch1: E−W

conv_grad = HealPixConv(nside=nside, in_channels=1, out_channels=2,
                        gauge_type="phi")
conv_grad.set_kernel(W_grad)

grad = conv_grad(sky)    # [2, N]
```

### Example D — Multi-channel co-convolution

```python
C = 4   # physical channels: T, U, V, Q

W_multi = np.zeros((1, C, 2*C, 9), dtype=np.float32)
for c in range(C):
    # smooth channel c → output c
    W_multi[0, c, c,   4]              = w_center
    W_multi[0, c, c,   [0,1,2,3,5,6,7,8]] = w_ring
    W_multi[0, c, c,   :] /= W_multi[0, c, c, :].sum()
    # Laplacian of channel c → output c+C
    W_multi[0, c, c+C, 4]              =  8.0
    W_multi[0, c, c+C, [0,1,2,3,5,6,7,8]] = -1.0

conv_multi = HealPixConv(nside=nside, in_channels=C, out_channels=2*C)
conv_multi.set_kernel(W_multi)

maps   = np.random.randn(8, C, 12*nside**2).astype(np.float32)
result = conv_multi(maps)   # [8, 8, N]
```

### Example E — Partial-sky patch

```python
import healpy as hp

nside = 128;  level = 7
patch = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0.), np.radians(15.), nest=True)

conv_patch = HealPixConv(nside=nside, in_channels=1, out_channels=1,
                         kernel_sz=3, cell_ids=patch, level=level)
conv_patch.set_kernel(W)   # same W as Example A

x_patch  = np.random.randn(len(patch)).astype(np.float32)
y_smooth = conv_patch(x_patch)   # [len(patch)]
```

### Example F — Physics-informed block

Fixed gradient pre-processing + learned feature extraction:

```python
class PhysicsBlock(nn.Module):
    def __init__(self, nside, out_channels):
        super().__init__()
        self.grad = HealPixConv(nside=nside, in_channels=1, out_channels=2,
                                kernel_sz=3, gauge_type="phi")
        self.grad.set_kernel(W_grad)   # from Example C, frozen

        self.feat = HealPixConv(nside=nside, in_channels=2,
                                out_channels=out_channels,
                                kernel_sz=3, use_norm=True)

    def forward(self, x):
        g = self.grad(x)      # [B, 2, N] — no gradient computed
        return self.feat(g)   # [B, out_channels, N] — learned

block = PhysicsBlock(nside=64, out_channels=16)
y = block(torch.randn(4, 1, 12 * 64**2))   # (4, 16, 49152)
```

---

## Choosing the gauge

| Application | Recommended gauge | `ref_direction` |
|---|---|---|
| Global NWP / ERA5 | `"phi"` | — |
| CMB / cosmology | `"cosmo"` | — |
| Regional domain (e.g. Europe) | `"projected_ref"` | Point to the Pacific |
| Any domain avoiding poles | `"projected_ref"` | `[0, 0, -1]` ≡ `"phi"` |
| Minimize artefacts over ocean | `"projected_ref"` | Point toward an ocean |

```python
# Singularities pushed to mid-Pacific and mid-Indian Ocean
conv = HealPixConv(nside=64, in_channels=1, out_channels=16,
                   gauge_type="projected_ref",
                   ref_direction=[0, 1, 0])   # singularities at φ=90° and φ=270°
```

---

## Kernel shape reference

| `n_gauges` | `set_kernel(W)` accepted shapes |
|---|---|
| 1 | `[C_in, C_out, P]` |
| G | `[C_in, C_out, P]` — same kernel broadcast to all G gauges |
| G | `[G, C_in, C_out, P]` — different kernel per gauge |

Internally `weight` always has shape `[G, C_in, C_out, P]`.
