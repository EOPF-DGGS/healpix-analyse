# `convol.py` — Gauge-Equivariant Spherical Convolution on HEALPix

> **`HealPixConv`** is a PyTorch `nn.Module` that applies a learned
> gauge-equivariant convolution to signals defined on a HEALPix sphere.
> It works on full-sky maps and partial-sky patches, accepts both NumPy
> arrays and PyTorch tensors, and is fully differentiable end-to-end.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation & Dependencies](#2-installation--dependencies)
3. [Quick Start](#3-quick-start)
4. [Algorithm](#4-algorithm)
   - 4.1 [Stage A — Kernel definition and rotation](#41-stage-a--kernel-definition-and-rotation)
   - 4.2 [Stage B — Bilinear binding](#42-stage-b--bilinear-binding)
   - 4.3 [Stage C — Forward pass](#43-stage-c--forward-pass)
5. [Gauge Types and Singularities](#5-gauge-types-and-singularities)
   - 5.1 [`"phi"` — meridian-aligned](#51-phi--meridian-aligned)
   - 5.2 [`"cosmo"` — cosmological convention](#52-cosmo--cosmological-convention)
   - 5.3 [`"projected_ref"` — one freely placed antipodal pair](#53-projected_ref--one-freely-placed-antipodal-pair)
   - 5.4 [`"two_ref"` — two freely placed singularity pairs](#54-two_ref--two-freely-placed-singularity-pairs)
   - 5.5 [Choosing singularity locations in practice](#55-choosing-singularity-locations-in-practice)
6. [Public API — `HealPixConv`](#6-public-api--healpixconv)
   - 6.1 [Constructor parameters](#61-constructor-parameters)
   - 6.2 [`forward(x)`](#62-forwardx)
   - 6.3 [`set_kernel(W, bias, requires_grad)`](#63-set_kernelw-bias-requires_grad)
   - 6.4 [`singularity_info()`](#64-singularity_info)
7. [Tensor shapes reference](#7-tensor-shapes-reference)
8. [Internal helpers](#8-internal-helpers)
9. [Performance notes](#9-performance-notes)
10. [Common recipes](#10-common-recipes)

---

## 1. Overview

`HealPixConv` implements the spherical convolution described in the
*gauge-equivariant* framework: a fixed `kernel_sz × kernel_sz` stencil is
defined at the North Pole, then **rotated** to each target pixel by a
composition of spherical rotations. The same learned kernel is applied at
every pixel, in every gauge orientation, making the layer equivariant to
the chosen gauge group.

Key design choices:

- **The kernel never moves.** It is the *data* that is pulled toward the
  fixed kernel via bilinear interpolation of the input map, not the kernel
  that is warped per pixel.
- **All heavy geometry is precomputed once** at construction time and stored
  as registered buffers. The forward pass is purely index-and-multiply.
- **Fully vectorised**: no Python loops over pixels or gauges in either the
  constructor or the forward pass.
- **Differentiable**: every operation in the hot path (`index_select`,
  `einsum`, arithmetic) is supported by PyTorch autograd.

---

## 2. Installation & Dependencies

```bash
pip install torch healpix-geo numpy
```

The module also imports `get_interp_weights` from the companion module
`healpix_analyse.healpix_interp` (part of the same package).

Python ≥ 3.10 is required (uses `X | Y` union type hints and `tuple[...]`
built-in generics).

---

## 3. Quick Start

```python
import numpy as np
import torch
from healpix_analyse.convol import HealPixConv

nside      = 64                        # HEALPix resolution
npix       = 12 * nside ** 2           # 49 152 pixels
in_ch, out_ch = 3, 16

# --- build the layer (geometry precomputed here) ---
conv = HealPixConv(
    nside       = nside,
    in_channels = in_ch,
    out_channels= out_ch,
    kernel_sz   = 3,          # 3×3 = 9 stencil points
    n_gauges    = 4,          # G=4 gauge orientations
    gauge_type  = "projected_ref",
    singularity_lonlat = (84.0, 28.0),   # Himalayas + antipodal Pacific
)

# --- forward pass (differentiable) ---
x = torch.randn(8, in_ch, npix)         # batch of 8 maps
y = conv(x)                             # [8, G*out_ch, npix] = [8, 64, 49152]

# --- inspect singularity placement ---
print(conv.singularity_info())
```

---

## 4. Algorithm

The layer operates in three stages that are all precomputed at construction
time. The forward pass reduces to a single gather + einsum.

### 4.1 Stage A — Kernel definition and rotation

A `kernel_sz × kernel_sz` grid of `P = kernel_sz²` unit vectors is placed
at the **North Pole** (`z = 1`) with angular spacing equal to one HEALPix
pixel width:

```
alpha_pix = sqrt(4π / (12 · nside²))   ← angular size of one pixel

stencil point (i, j):
    dtheta = sqrt(i² + j²) · alpha_pix
    dphi   = atan2(j, i)
    vec    = [sin(dtheta)·cos(dphi),
              sin(dtheta)·sin(dphi),
              cos(dtheta)]
```

This stencil template is **fixed** and never changes. For every target
pixel `k` (colatitude `θ_k`, longitude `φ_k`) and every gauge orientation
`g`, a total rotation matrix is assembled as:

```
R_total[k, g] = R_gauge(α_g)  @  Rz(φ_k)  @  Ry(θ_k)
                └─ gauge roll ─┘  └─── carry North Pole → pixel k ───┘
```

where `R_gauge(α_g)` is the Rodrigues rotation around the surface normal
`n_k` by the gauge angle `α_g` (see [Section 5](#5-gauge-types-and-singularities)).

Each stencil point is then rotated to its position on the sphere around
pixel `k`:

```
rotated[k, g, p] = R_total[k, g] @ vec_pole[p]    ∈ ℝ³
```

producing a tensor of shape `[K, G, P, 3]`.

### 4.2 Stage B — Bilinear binding

For each of the `K × G × P` rotated direction vectors, the function
`get_interp_weights` returns the 4 nearest HEALPix pixel indices and their
bilinear interpolation weights:

```
idx[4, K·G·P]    — absolute NESTED pixel ids of the 4 neighbours
w  [4, K·G·P]    — bilinear weights (sum to 1 per stencil point)
```

These are reshaped to `[G, 4, K·P]` and stored as persistent buffers
`_pos_safe` and `_w_norm`. For partial-sky inputs, neighbours outside the
patch are masked and weights are renormalised; stencil points with no
available neighbour fall back to the centre pixel.

### 4.3 Stage C — Forward pass

**Important:** it is the **data** that is brought to the fixed kernel, not
the kernel that is deformed per pixel.

At inference, for each stencil point `p` of pixel `k` under gauge `g`, the
signal value is obtained by bilinear interpolation of the input map:

```
x_interp[b, c, g, k, p] = Σ_{j=0}^{3}  w[g, j, k, p] · x[b, c, nbr[g, j, k, p]]
```

The interpolated values are then contracted with the learned kernel via a
single einsum over all gauges simultaneously:

```
y[b, g·C_out + o, k] = Σ_{c, p}  W[g, c, o, p] · x_interp[b, c, g, k, p]
```

In index notation: `"bcgkp, gcop -> bgok"`.

The full forward in code reduces to:

```python
pos_flat  = pos.reshape(-1)                        # [G·4·K·P]
vals_flat = t_sorted.index_select(2, pos_flat)     # [B, C_in, G·4·K·P]
vals      = vals_flat.view(B, C_in, G, 4, K, P)
gathered  = (vals * w_shaped).sum(dim=3)           # [B, C_in, G, K, P]
y         = einsum("bcgkp, gcop -> bgok", gathered, W)  # [B, G, C_out, K]
```

---

## 5. Gauge Types and Singularities

The **hairy-ball theorem** states that every smooth tangent vector field on
S² must have at least one zero. Equivalently, the total index of all
singularities of a gauge field on S² must equal exactly 2.

Each built-in gauge places those unavoidable bad points differently.

### 5.1 `"phi"` — meridian-aligned

```
α_base(k) = 0
```

The kernel is always aligned with the local meridian. Singularities are
fixed at the **geographic North and South Poles** (index +1 each, total = 2).

Best for data that is well-behaved away from the poles and where
interpretability matters. Computationally cheapest.

### 5.2 `"cosmo"` — cosmological convention

```
α_base(k) = -φ_k   (Northern hemisphere, θ ≤ π/2)
α_base(k) = +φ_k   (Southern hemisphere, θ > π/2)
```

Same singularity locations as `"phi"` but the gauge angle flips sign
across the equator to match the CMB/cosmological convention used by
`healpy`. Useful when comparing results with legacy healpy-based pipelines.

### 5.3 `"projected_ref"` — one freely placed antipodal pair

A reference vector **r** ∈ ℝ³ is projected onto the tangent plane at each
pixel `k`:

```
r_proj(k) = r − (r · n_k) · n_k
α_base(k) = atan2( r_proj · e_φ,  r_proj · e_θ )
```

The gauge is undefined exactly where `r_proj = 0`, i.e. where `r ∥ n_k`.
This occurs at the two **antipodal** points:

```
singularity₁ = (lon_s,        lat_s)       ← direction of r
singularity₂ = (lon_s + 180°, −lat_s)      ← antipode, forced
```

To place the first singularity at a desired location, pass:

```python
singularity_lonlat = (lon_s, lat_s)
```

The reference vector is computed automatically:

```
r = [cos(lat_s)·cos(lon_s),  cos(lat_s)·sin(lon_s),  sin(lat_s)]
```

**Constraint:** the second singularity is always the antipode of the first.
You cannot move them independently with this gauge type — use `"two_ref"`
for that.

### 5.4 `"two_ref"` — two freely placed singularity pairs

Two independent reference vectors **r₁** and **r₂** define the gauge angle
via the **complex product** of their tangent-plane projections:

```
z_j(k) = (r_j_proj · e_θ)  +  i · (r_j_proj · e_φ)      j = 1, 2

α_base(k) = arg( z₁(k) · z₂(k) )
           = atan2( Re(z₁)·Im(z₂) + Im(z₁)·Re(z₂),
                    Re(z₁)·Re(z₂) − Im(z₁)·Im(z₂) )
```

Using the complex product avoids two independent `atan2` calls (which would
each wrap and accumulate phase jumps) and computes `arg(z₁) + arg(z₂)` in
a single numerically stable operation.

**Singularity structure (Poincaré–Hopf budget):**

| Location | Count | Index | Origin |
|---|---|---|---|
| `+r₁`, `−r₁` | 2 | +1 each | zeros of z₁ |
| `+r₂`, `−r₂` | 2 | +1 each | zeros of z₂ |
| N-Pole, S-Pole | 2 | −1 each | base-frame side-effect |
| **Total** | | **4−2 = 2** | ✓ |

The four user-controlled points are **index +1** (vortex-like), which is the
well-behaved type to place over unimportant regions. The geographic poles
become **index −1** (hyperbolic saddle), a sharper singularity — keep them
outside the domain of interest or over regions where accuracy is not
required.

To place singularities, pass a list of two `(lon, lat)` pairs:

```python
singularity_lonlat = [(lon₁, lat₁), (lon₂, lat₂)]
```

Each pair controls one **user-specified** singularity; its antipodal point
appears automatically. The full set of four bad points is thus:

```
{(lon₁, lat₁),  (lon₁+180°, −lat₁),  (lon₂, lat₂),  (lon₂+180°, −lat₂)}
```

### 5.5 Choosing singularity locations in practice

| Domain | Strategy |
|---|---|
| **Ocean model** | Place all 4 bad points over land masses. Example: Amazon + Borneo (antipodal pair) and Africa + central Pacific (second pair). |
| **Atmosphere** | Place all 4 bad points over open ocean. Example: central Pacific + Indian Ocean (pair 1) and South Atlantic + Maritime Continent (pair 2). |
| **Full sphere / neutral** | Use `"phi"` (poles). For `"projected_ref"`, passing `singularity_lonlat=(0, 90)` reproduces the pole placement with a smoother field off the poles. |
| **Poles outside domain** | Any `"two_ref"` config naturally moves the geographic-pole index-−1 singularities to a fixed location — confirm they are harmless for your domain. |

---

## 6. Public API — `HealPixConv`

### 6.1 Constructor parameters

```python
HealPixConv(
    nside,
    in_channels,
    out_channels,
    kernel_sz          = 3,
    n_gauges           = 1,
    gauge_type         = "phi",
    singularity_lonlat = None,
    ref_direction      = None,
    cell_ids           = None,
    level              = None,
    nest               = True,
    use_norm           = False,
    device             = None,
    ellipsoid          = "WGS84",
    dtype              = torch.float32,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nside` | `int` | — | HEALPix resolution. Must be a power of 2 (e.g. 32, 64, 128). |
| `in_channels` | `int` | — | Number of input feature channels C_in. |
| `out_channels` | `int` | — | Output channels per gauge C_out. Total output channels = G × C_out. |
| `kernel_sz` | `int` | `3` | Stencil side length. Must be a positive odd integer. P = kernel_sz². |
| `n_gauges` | `int` | `1` | Number of gauge orientations G. Gauge g rotates the stencil by g·π/G. |
| `gauge_type` | `str` | `"phi"` | One of `"phi"`, `"cosmo"`, `"projected_ref"`, `"two_ref"`. See Section 5. |
| `singularity_lonlat` | `(lon, lat)` or `[(lon₁,lat₁),(lon₂,lat₂)]` or `None` | `None` | Geographic coordinates of the desired singularity point(s) in degrees. For `"projected_ref"`: one pair. For `"two_ref"`: a list of two pairs. Overrides `ref_direction`. |
| `ref_direction` | `array (3,)` or `(2, 3)` or `None` | `None` | Low-level alternative: raw unit reference vector(s). Shape `(3,)` for `"projected_ref"`, `(2, 3)` for `"two_ref"`. Ignored when `singularity_lonlat` is provided. |
| `cell_ids` | `array-like` or `None` | `None` | NESTED pixel indices for a partial-sky patch. `None` = full sphere. Requires `level`. |
| `level` | `int` or `None` | `None` | HEALPix level such that `nside = 2**level`. Required when `cell_ids` is given. |
| `nest` | `bool` | `True` | Pixel ordering of the input map. `True` = NESTED, `False` = RING. |
| `use_norm` | `bool` | `False` | Apply GroupNorm + ReLU after the convolution. |
| `device` | `str` or `torch.device` or `None` | `None` | Target device. Defaults to CUDA if available, else CPU. |
| `ellipsoid` | `str` | `"WGS84"` | Reference ellipsoid for healpix_geo coordinate conversions. |
| `dtype` | `torch.dtype` | `torch.float32` | Floating-point precision for all buffers and parameters. |

**Raises:**

- `ValueError` — if `nside` is not a positive power of 2.
- `ValueError` — if `kernel_sz` is not a positive odd integer.
- `ValueError` — if `gauge_type` is not one of the four valid strings.
- `ValueError` — if `cell_ids` is provided without `level`, or with an
  inconsistent `level`.
- `ValueError` — if `singularity_lonlat` is used with a gauge type that
  does not support it, or if the wrong number of pairs is provided for
  `"two_ref"`.

---

### 6.2 `forward(x)`

Apply the gauge-equivariant spherical convolution.

```python
y = conv(x)
```

**Parameters:**

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `x` | `np.ndarray` or `torch.Tensor` | `[N]`, `[B, N]`, or `[B, C_in, N]` | Input map(s) at the HEALPix resolution of this layer. N must equal `len(cell_ids)` for partial-sky, or `12 · nside²` for full sphere. |

**Returns:**

| Name | Type | Shape | Description |
|---|---|---|---|
| `y` | same type as `x` | `[G·C_out, N]` or `[B, G·C_out, N]` | Convolved output. Output type and number of dimensions match the input. |

**Raises:**

- `ValueError` — if `x` has the wrong number of channels or pixels.

---

### 6.3 `set_kernel(W, bias=None, requires_grad=False)`

Replace the learnable kernel with a fixed (or re-initialised) array.
Returns `self` for chaining.

```python
conv.set_kernel(W, bias=None, requires_grad=False)
```

| Parameter | Type | Shape | Description |
|---|---|---|---|
| `W` | `array-like` | `[C_in, C_out, P]` or `[G, C_in, C_out, P]` | Kernel weights. The first form broadcasts the same kernel over all G gauges. |
| `bias` | `array-like` or `None` | `[G · C_out]` | Bias vector. `None` resets the bias to zero. |
| `requires_grad` | `bool` | — | If `True`, the kernel and bias remain learnable after this call. |

**Example — isotropic Gaussian smoothing (kernel_sz=3):**

```python
W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]              = 0.5          # centre point (index 4 in a 3×3 grid)
W[0, 0, [0,1,2,3,5,6,7,8]] = 0.5 / 8  # 8 neighbours equally weighted
conv.set_kernel(W)
```

---

### 6.4 `singularity_info()`

Return a human-readable string describing where the gauge singularities are
placed.

```python
print(conv.singularity_info())
```

Example output for `"two_ref"`:

```
gauge_type='two_ref':
  singularity 1  : lon=-55.00°  lat=-10.00°  (index +1, user-defined)
  singularity 1b : lon=+125.00°  lat=+10.00°  (index +1, antipode of 1)
  singularity 2  : lon=+20.00°  lat=+10.00°  (index +1, user-defined)
  singularity 2b : lon=+200.00°  lat=-10.00°  (index +1, antipode of 2)
  N/S poles      : index -1 each (side-effect, keep outside domain of interest)
  ref_directions : r1=[...]
                   r2=[...]
```

---

## 7. Tensor shapes reference

| Symbol | Meaning |
|---|---|
| B | Batch size |
| C_in | Input channels |
| C_out | Output channels per gauge |
| G | Number of gauges (`n_gauges`) |
| K | Number of pixels (`len(cell_ids)` or `12·nside²`) |
| P | Stencil points (`kernel_sz²`) |

| Buffer / tensor | Shape | Description |
|---|---|---|
| `_pos_safe` | `[G, 4, K·P]` | Column indices into the sorted input for each bilinear neighbour |
| `_w_norm` | `[G, 4, K·P]` | Renormalised bilinear weights |
| `_sort_order` | `[K]` | Index that sorts pixel ids ascending |
| `_inv_order` | `[K]` | Inverse permutation to restore original order |
| `weight` | `[G, C_in, C_out, P]` | Learned kernel (nn.Parameter) |
| `bias` | `[G · C_out]` | Learned bias (nn.Parameter) |

---

## 8. Internal helpers

These functions are not part of the public API but are documented here for
developers who want to extend or debug the module.

### `_local_kernel_grid(kernel_sz, nside) → np.ndarray [P, 3]`

Builds the `kernel_sz × kernel_sz` stencil at the North Pole. Angular
spacing equals one pixel width (`sqrt(4π / (12·nside²))`). The centre
point (index `kernel_sz//2 · (kernel_sz + 1)`) maps to the exact North
Pole direction `[0, 0, 1]`.

### `_build_rotation_matrices(th, ph, G, gauge_type, ...) → torch.Tensor [K, G, 3, 3]`

Assembles the full rotation matrix `R_total = R_gauge(α_g) @ Rz(φ) @ Ry(θ)`
for every pixel and gauge. `R_gauge` is computed via the Rodrigues formula
(axis-angle rotation around the surface normal `n`):

```
R_gauge = I·cos(α) + K_skew·sin(α) + (n⊗n)·(1 − cos(α))
```

where `K_skew` is the skew-symmetric matrix of `n`.

### `_get_interp_weights(nside, vecs, nest, device, dtype) → (idx [4, M], w [4, M])`

Converts `M` direction vectors to bilinear interpolation weights and
neighbour indices on the HEALPix grid. Internally delegates to
`healpix_analyse.healpix_interp.get_interp_weights` (one vectorised call,
no Python loop).

### `_bind_support_batched(idx_t, w_t, ids_sorted, ...) → (pos_safe, w_norm)`

Maps absolute NESTED pixel ids to column indices within the current pixel
patch via `torch.searchsorted`. Handles three edge cases: neighbours
outside the patch (zeroed), stencil points with no valid neighbour (fall
back to the centre pixel), and zero-sum weight columns (assign weight 1 to
the first present neighbour). All G gauges are processed in a single
`searchsorted` call over the full `[G · 4 · K · P]` index array.

---

## 9. Performance notes

### Constructor (one-time cost)

The geometry precomputation scales as O(K · G · P) and is dominated by the
`healpix_geo` coordinate lookup and the `get_interp_weights` call. For
typical configurations (nside=64, G=4, kernel_sz=3):

| Operation | K·G·P entries | Approx. time |
|---|---|---|
| `healpix_to_lonlat` | 49 152 | < 0.1 s |
| `get_interp_weights` | 49 152 × 4 × 9 = 1.77 M | ~0.5 s |
| `_bind_support_batched` | 1.77 M searchsorted | ~0.3 s |

### Forward pass (per batch)

The forward is a single `index_select` over `[G · 4 · K · P]` entries,
followed by a weighted sum and one `einsum`. On GPU (A100) with batch size
8, nside=64, G=4, C_in=16, C_out=32: typically under 5 ms.

### Memory

The dominant buffers are `_pos_safe` and `_w_norm`, each of shape
`[G, 4, K·P]`. For nside=128, G=4, kernel_sz=5 (P=25):

```
K = 12 × 128² = 196 608
Buffer size = 4 × 4 × 196 608 × 25 × 4 bytes (float32) ≈ 750 MB
```

For very large nside, consider reducing G or kernel_sz, or using
`torch.float16` for the buffers.

---

## 10. Common recipes

### Full-sphere convolution, `"phi"` gauge

```python
conv = HealPixConv(nside=64, in_channels=1, out_channels=32,
                   kernel_sz=3, n_gauges=1, gauge_type="phi")
y = conv(x)   # x: [B, 1, 49152]  →  y: [B, 32, 49152]
```

### Multi-gauge equivariant layer

```python
conv = HealPixConv(nside=64, in_channels=16, out_channels=16,
                   kernel_sz=3, n_gauges=4, gauge_type="phi")
# output has G * C_out = 64 channels
```

### Ocean model — singularities over two land masses

```python
# Singularity pair 1: Amazon basin + Borneo (its antipode)
# Singularity pair 2: Africa + central Pacific (its antipode)
conv = HealPixConv(
    nside=64, in_channels=3, out_channels=16,
    gauge_type="two_ref",
    singularity_lonlat=[(-55.0, -10.0), (20.0, 5.0)],
)
print(conv.singularity_info())
```

### Atmospheric model — singularities over open ocean

```python
conv = HealPixConv(
    nside=64, in_channels=5, out_channels=32,
    gauge_type="projected_ref",
    singularity_lonlat=(-160.0, 0.0),   # central Pacific + Indian Ocean antipode
)
```

### Partial-sky patch

```python
import healpix_geo

nside = 64
depth = int(np.log2(nside))
cell_ids, _, _ = healpix_geo.nested.cone_coverage(
    (0.0, 45.0), 20.0, depth, ellipsoid="WGS84"
)
conv = HealPixConv(
    nside=nside, in_channels=1, out_channels=8,
    cell_ids=cell_ids, level=depth,
)
x_patch = torch.randn(4, 1, len(cell_ids))
y_patch = conv(x_patch)
```

### Fixed (non-learnable) isotropic kernel

```python
conv = HealPixConv(nside=32, in_channels=1, out_channels=1, kernel_sz=3)
W = np.zeros((1, 1, 9), dtype=np.float32)
W[0, 0, 4]              = 0.5       # centre
W[0, 0, [0,1,2,3,5,6,7,8]] = 0.5/8 # ring
conv.set_kernel(W, requires_grad=False)
```

### With GroupNorm + ReLU

```python
conv = HealPixConv(
    nside=64, in_channels=16, out_channels=16,
    n_gauges=4, use_norm=True,
)
# The output is already passed through GroupNorm then ReLU inside forward()
```

### NumPy in / NumPy out

```python
x_np = np.random.randn(49152).astype(np.float32)
y_np = conv(x_np)   # returns np.ndarray, same shape policy as input
```
