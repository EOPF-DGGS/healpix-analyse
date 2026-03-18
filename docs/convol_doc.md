# `HealPixConv` — Spherical convolution on HEALPix maps

**Module** `healpix_analyse.convol`  
**Class** `HealPixConv(nside, in_channels, out_channels, ...)`  
**Inherits from** `torch.nn.Module`

---

## What it does

`HealPixConv` applies a spatial convolution to a map defined on a HEALPix sphere.
For each pixel, it collects a local neighbourhood of **K = kernel_sz²** points
(the pixel itself plus its up-to-8 nearest neighbours via `healpy.get_all_neighbours`),
then mixes them with a weight tensor **W[C_in, C_out, K]**.

The operation for a single output channel `o` at pixel `n` is:

$$
y_{o,n} = \text{bias}_o + \sum_{c=0}^{C_{in}-1} \sum_{k=0}^{K-1} W_{c,o,k} \cdot x_{c,\;\text{stencil}[n,k]}
$$

where `stencil[n, k]` is the index of the `k`-th neighbour of pixel `n`.

The stencil is **precomputed once** at construction and stored as a
`torch.long` buffer — forward passes do no HEALPix queries at all.

---

## Stencil layout

With `kernel_sz=3` (K = 9), the stencil index `k` maps to the following
geometrical positions:

```
k=0  →  center pixel (always the target pixel itself)
k=1  →  SW neighbour
k=2  →  W  neighbour
k=3  →  NW neighbour
k=4  →  N  neighbour
k=5  →  NE neighbour
k=6  →  E  neighbour
k=7  →  SE neighbour
k=8  →  S  neighbour
```

For pixels at the **boundary of a partial-sky patch** where a true
neighbour does not exist in `cell_ids`, the missing position is silently
replaced by the **center pixel value** (zero-padding equivalent in a
spherical sense).

With `kernel_sz=1` (K = 1), only `k=0` exists — this is a 1×1 (channel-mixing)
convolution with no spatial component.

---

## Input / output shapes

| Input shape | Output shape | Notes |
|---|---|---|
| `[N]` | `[N]` | Only when `C_in = C_out = 1` |
| `[N]` | `[C_out, N]` | When `C_out > 1` |
| `[B, N]` | `[B, C_out, N]` | Single-channel batch |
| `[B, C_in, N]` | `[B, C_out, N]` | Multi-channel batch |

Both **numpy arrays** and **torch tensors** are accepted and the return type
matches the input type.

---

## Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside` | `int` | — | HEALPix resolution (must be a power of 2). |
| `in_channels` | `int` | — | Number of input feature channels `C_in`. |
| `out_channels` | `int` | — | Number of output feature channels `C_out`. |
| `kernel_sz` | `{1, 3}` | `3` | `1` → 1×1 conv (no spatial mixing). `3` → 9-point neighbourhood. |
| `use_norm` | `bool` | `False` | If `True`, apply GroupNorm + ReLU after the linear mix. |
| `cell_ids` | array-like or `None` | `None` | Pixel indices (NESTED) for partial-sky operation. `None` = full sphere. |
| `level` | `int` or `None` | `None` | HEALPix level such that `nside = 2**level`. Required with `cell_ids`. |
| `nest` | `bool` | `True` | NESTED ordering if `True`, RING if `False`. |
| `device` | device or str or `None` | `None` | Torch device. Defaults to CUDA if available, else CPU. |
| `dtype` | `torch.dtype` | `float32` | Dtype for learnable parameters. |

---

## Learnable parameters

| Attribute | Shape | Description |
|---|---|---|
| `weight` | `[C_in, C_out, K]` | Spatial + channel mixing kernel. Initialised with Kaiming uniform. |
| `bias` | `[C_out]` | Per-output-channel bias. Initialised to zero. |

---

# Case 1 — Learned kernels: U-Net on the sphere

This is the primary design intent of `HealPixConv`: use it as a building
block in a learnable architecture such as a U-Net.

## Architecture overview

```
Input map  [B, C_in, N]
     │
     ▼  HealPixConv  (encoder block 1)   nside=64
     │  GroupNorm + ReLU
     ▼
[B, 32, N]
     │
     ▼  HealPixDown  (nside 64 → 32)
     ▼
[B, 32, N/4]
     │
     ▼  HealPixConv  (encoder block 2)   nside=32
     │  GroupNorm + ReLU
     ▼
[B, 64, N/4]
     │
     ▼  HealPixDown  (nside 32 → 16)
     ▼
[B, 64, N/16]     ← bottleneck
     │
     ▼  HealPixUp    (nside 16 → 32)
     ▼
[B, 64, N/4]  +  skip connection
     │
     ▼  HealPixConv  (decoder block)
     │  GroupNorm + ReLU
     ▼
[B, 32, N/4]
     │
     ▼  HealPixUp    (nside 32 → 64)
     ▼
[B, 32, N]
     │
     ▼  HealPixConv  kernel_sz=1  (output projection)
     ▼
Output  [B, C_out, N]
```

## Minimal U-Net example

```python
import torch
import torch.nn as nn
import numpy as np
from healpix_analyse.convol import HealPixConv
from healpix_analyse.down   import HealPixDown
from healpix_analyse.up     import HealPixUp


class SphereUNet(nn.Module):
    """
    Minimal 2-level spherical U-Net on a full HEALPix sphere.

    Input : [B, C_in,  12*nside**2]
    Output: [B, C_out, 12*nside**2]
    """

    def __init__(self, nside: int, in_channels: int, out_channels: int):
        super().__init__()

        self.nside = nside

        # ---- Encoder ----
        self.enc1 = HealPixConv(
            nside=nside, in_channels=in_channels, out_channels=32,
            kernel_sz=3, use_norm=True,
        )
        self.down1 = HealPixDown(nside_in=nside, mode="smooth")

        self.enc2 = HealPixConv(
            nside=nside // 2, in_channels=32, out_channels=64,
            kernel_sz=3, use_norm=True,
        )
        self.down2 = HealPixDown(nside_in=nside // 2, mode="smooth")

        # ---- Bottleneck ----
        self.bottleneck = HealPixConv(
            nside=nside // 4, in_channels=64, out_channels=128,
            kernel_sz=3, use_norm=True,
        )

        # ---- Decoder ----
        self.up2   = HealPixUp(nside_in=nside // 4)
        # After upsampling, concatenate with skip → 128 + 64 = 192 channels
        self.dec2  = HealPixConv(
            nside=nside // 2, in_channels=192, out_channels=64,
            kernel_sz=3, use_norm=True,
        )

        self.up1   = HealPixUp(nside_in=nside // 2)
        # After upsampling, concatenate with skip → 64 + 32 = 96 channels
        self.dec1  = HealPixConv(
            nside=nside, in_channels=96, out_channels=32,
            kernel_sz=3, use_norm=True,
        )

        # ---- Output head (1×1 conv = channel projection) ----
        self.head = HealPixConv(
            nside=nside, in_channels=32, out_channels=out_channels,
            kernel_sz=1, use_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C_in, N]
        returns : [B, C_out, N]
        """
        # ---- Encoder ----
        s1 = self.enc1(x)                           # [B, 32, N]
        x2, _ = self.down1(s1)                      # [B, 32, N/4]

        s2 = self.enc2(x2)                          # [B, 64, N/4]
        x3, _ = self.down2(s2)                      # [B, 64, N/16]

        # ---- Bottleneck ----
        xb = self.bottleneck(x3)                    # [B, 128, N/16]

        # ---- Decoder ----
        xu2, _ = self.up2(xb)                       # [B, 128, N/4]
        xu2 = torch.cat([xu2, s2], dim=1)           # [B, 192, N/4]
        xd2 = self.dec2(xu2)                        # [B, 64, N/4]

        xu1, _ = self.up1(xd2)                      # [B, 64, N]
        xu1 = torch.cat([xu1, s1], dim=1)           # [B, 96, N]
        xd1 = self.dec1(xu1)                        # [B, 32, N]

        return self.head(xd1)                       # [B, C_out, N]


# ---------- instantiate and run ----------
nside = 64
B, C_in, C_out = 4, 2, 1

model = SphereUNet(nside=nside, in_channels=C_in, out_channels=C_out)

N = 12 * nside**2
x = torch.randn(B, C_in, N)
y = model(x)

print(f"Input  shape: {tuple(x.shape)}")   # (4, 2, 49152)
print(f"Output shape: {tuple(y.shape)}")   # (4, 1, 49152)
print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
```

## Training loop sketch

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()

    # x_batch: [B, C_in, N],   y_true: [B, C_out, N]
    y_pred = model(x_batch)
    loss   = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}")
```

## Freezing or inspecting learned kernels

After training, the learned spatial weights are accessible as standard
`nn.Parameter` tensors:

```python
# Shape: [C_in, C_out, K=9]
W = model.enc1.weight.detach().cpu().numpy()

# Weight for input channel 0, output channel 3, all 9 stencil positions:
print(W[0, 3, :])
# [center, SW, W, NW, N, NE, E, SE, S]
```

---

# Case 2 — Manual kernels: hand-crafted spherical filters

`HealPixConv` can also be used as a **fixed filter** by assigning weights
directly into `self.weight` and `self.bias` after construction, then setting
`requires_grad=False`.  This is useful for:

- classical image processing (smoothing, edge detection, gradient estimation),
- physics-motivated priors (isotropic Laplacian, directional derivatives),
- analysis: computing local statistics or spatial correlations.

## The kernel layout

The weight tensor has shape **`[C_in, C_out, K]`** with `K = 9` for
`kernel_sz=3`.  The index `k` maps to a fixed geometrical direction:

```
k : 0=center, 1=SW, 2=W, 3=NW, 4=N, 5=NE, 6=E, 7=SE, 8=S
```

Because HEALPix pixels have the same area but are *not* arranged on a
regular Cartesian grid, "N/S/E/W" here means the direction reported by
`healpy.get_all_neighbours`, which is approximate.  For most filtering
applications this is sufficient; for precision directional derivatives
the `SphericalStencil` class (in the Healpix_UNET package) provides
gauge-aware rotated stencils.

## Helper: freeze weights

```python
def set_fixed_kernel(conv: HealPixConv,
                     W: np.ndarray,
                     bias: np.ndarray | None = None) -> HealPixConv:
    """
    Replace conv.weight (and optionally conv.bias) with fixed numpy arrays.

    Parameters
    ----------
    conv  : HealPixConv  (already constructed)
    W     : np.ndarray, shape [C_in, C_out, K]
    bias  : np.ndarray, shape [C_out], optional (default: zeros)

    Returns
    -------
    conv  : the same object with frozen weights
    """
    with torch.no_grad():
        conv.weight.copy_(
            torch.as_tensor(W, dtype=conv.dtype, device=conv.device)
        )
        if bias is not None:
            conv.bias.copy_(
                torch.as_tensor(bias, dtype=conv.dtype, device=conv.device)
            )
        else:
            conv.bias.zero_()

    conv.weight.requires_grad_(False)
    conv.bias.requires_grad_(False)
    return conv
```

## Example A — Isotropic Gaussian smoothing

Average the 9-point stencil with a Gaussian profile.  The center pixel
carries the most weight; the 8 neighbours are weighted equally (they are
all approximately at the same angular distance from the center for a
regular HEALPix ring).

```python
import numpy as np
import torch
from healpix_analyse.convol import HealPixConv

nside = 64
N     = 12 * nside**2

conv_smooth = HealPixConv(
    nside=nside, in_channels=1, out_channels=1,
    kernel_sz=3, use_norm=False,
)

# k=0: center weight, k=1..8: ring weight
sigma   = 1.0   # in units of "pixel radii"
w_center = np.exp(-0.0 / (2 * sigma**2))   # distance 0 → weight 1
w_ring   = np.exp(-1.0 / (2 * sigma**2))   # distance ~1 pixel

W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 0] = w_center
W[0, 0, 1:] = w_ring
W /= W.sum()   # normalise: output is a weighted mean

set_fixed_kernel(conv_smooth, W)

# Apply to a HEALPix map
import healpy as hp
sky = np.random.randn(N).astype(np.float32)
sky_smooth = conv_smooth(sky)    # returns np.ndarray [N] because C_in=C_out=1
print(sky_smooth.shape)          # (49152,)
```

## Example B — Discrete Laplacian (edge detection / sharpening)

The discrete Laplacian on a 2-D grid is:

```
 0  -1   0
-1  +4  -1        (standard 4-neighbour)
 0  -1   0
```

On the 9-point HEALPix stencil we use all 8 neighbours with equal weight:

```
center weight: +8
ring weight  : -1   (for each of the 8 neighbours)
→ result ≈ Laplacian × pixel_area
```

```python
conv_laplacian = HealPixConv(
    nside=nside, in_channels=1, out_channels=1,
    kernel_sz=3, use_norm=False,
)

W_lap = np.zeros((1, 1, 9), dtype=np.float32)
W_lap[0, 0, 0]  =  8.0   # center
W_lap[0, 0, 1:] = -1.0   # 8 neighbours

set_fixed_kernel(conv_laplacian, W_lap)

edges = conv_laplacian(sky)   # highlights structures
```

## Example C — Directional gradient (N–S and E–W)

Two output channels: one for the approximate North–South derivative,
one for East–West.

```
N–S gradient:  W[0, 0, k] = +1 for k=4 (N),  -1 for k=8 (S), 0 elsewhere
E–W gradient:  W[0, 1, k] = +1 for k=6 (E),  -1 for k=2 (W), 0 elsewhere
```

```python
conv_grad = HealPixConv(
    nside=nside, in_channels=1, out_channels=2,
    kernel_sz=3, use_norm=False,
)

W_grad = np.zeros((1, 2, 9), dtype=np.float32)
# Channel 0: N–S  (k=4 North, k=8 South)
W_grad[0, 0, 4] = +1.0
W_grad[0, 0, 8] = -1.0
# Channel 1: E–W  (k=6 East, k=2 West)
W_grad[0, 1, 6] = +1.0
W_grad[0, 1, 2] = -1.0

set_fixed_kernel(conv_grad, W_grad)

# sky: [N]  →  grad: [2, N]
grad = conv_grad(sky)
grad_NS = grad[0]   # dsky/d(lat)
grad_EW = grad[1]   # dsky/d(lon)
```

## Example D — Multi-channel co-convolution

Apply several fixed filters simultaneously to a multi-channel input.
For instance, compute both the mean and the Laplacian of each channel
in a single pass using `in_channels=C`, `out_channels=2*C`.

```python
C = 4   # number of physical channels (e.g. T, U, V, Q)

conv_multi = HealPixConv(
    nside=nside, in_channels=C, out_channels=2 * C,
    kernel_sz=3, use_norm=False,
)

# W shape: [C_in=4, C_out=8, K=9]
W_multi = np.zeros((C, 2 * C, 9), dtype=np.float32)

for c in range(C):
    # First C output channels: Gaussian smoothing of input channel c
    W_multi[c, c, 0]  = w_center
    W_multi[c, c, 1:] = w_ring
    W_multi[c, c, :]  /= W_multi[c, c, :].sum()

    # Next C output channels: Laplacian of input channel c
    W_multi[c, c + C, 0]  =  8.0
    W_multi[c, c + C, 1:] = -1.0

set_fixed_kernel(conv_multi, W_multi)

# sky_batch: [B, C, N]  →  result: [B, 2*C, N]
sky_batch = np.random.randn(8, C, N).astype(np.float32)
result    = conv_multi(sky_batch)
print(result.shape)   # (8, 8, N)

smooth   = result[:, :C,  :]   # smoothed channels
laplace  = result[:, C:,  :]   # Laplacian channels
```

## Example E — Partial-sky co-convolution

All of the above work identically on a sky patch.  The only change is
passing `cell_ids` and `level` at construction.

```python
import healpy as hp

nside  = 128
level  = 7   # nside = 2**7

# Disc of radius 15° around the Galactic centre
vec    = hp.ang2vec(np.pi / 2, 0.0)
patch  = hp.query_disc(nside, vec, np.radians(15.0), nest=True)

conv_patch = HealPixConv(
    nside=nside, in_channels=1, out_channels=1,
    kernel_sz=3, use_norm=False,
    cell_ids=patch, level=level,
)

# Gaussian smoothing on the patch only
set_fixed_kernel(conv_patch, W)   # same W as Example A, shape [1,1,9]

sky_patch = np.random.randn(len(patch)).astype(np.float32)
sky_patch_smooth = conv_patch(sky_patch)
print(sky_patch_smooth.shape)   # (len(patch),)
```

---

## Combining learned and fixed filters

A common pattern is to use a fixed pre-processing filter (e.g. smoothing
or gradient) followed by a learned block.  Because both are `nn.Module`
instances they compose naturally:

```python
class PhysicsInformedBlock(nn.Module):
    """
    Fixed gradient estimator  →  learned feature extractor.
    """

    def __init__(self, nside: int, out_channels: int):
        super().__init__()

        # Fixed: compute 2 gradient channels from 1 input channel
        self.grad = HealPixConv(nside=nside, in_channels=1, out_channels=2,
                                kernel_sz=3)
        set_fixed_kernel(self.grad, W_grad)   # from Example C

        # Learned: extract features from the 2 gradient channels
        self.feat = HealPixConv(nside=nside, in_channels=2, out_channels=out_channels,
                                kernel_sz=3, use_norm=True)

    def forward(self, x):
        # x: [B, 1, N]  or  [B, N]  or  [N]
        g = self.grad(x)     # [B, 2, N]
        return self.feat(g)  # [B, out_channels, N]


block = PhysicsInformedBlock(nside=64, out_channels=16)
y = block(torch.randn(4, 1, 12 * 64**2))
print(y.shape)   # (4, 16, 49152)
```

---

## Stencil index reference

| `k` | Direction | Notes |
|-----|-----------|-------|
| 0 | **center** | The target pixel itself. Always present. |
| 1 | SW | May fall back to center at patch boundaries. |
| 2 | W  | |
| 3 | NW | |
| 4 | **N** | Use for N–S gradient (positive = northward). |
| 5 | NE | |
| 6 | **E** | Use for E–W gradient (positive = eastward). |
| 7 | SE | |
| 8 | **S** | |

> The exact geometric meaning of "N/S/E/W" depends on the local
> HEALPix tile orientation.  For the full sphere with NESTED ordering,
> the correspondence is consistent within each of the 12 base pixels
> but rotates between them.  For filtering and smoothing applications
> this has no practical impact; for precision directional derivatives
> use gauge-corrected stencils from `SphericalStencil`.
