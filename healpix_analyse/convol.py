"""
convol.py  (optimised)
======================
Gauge-equivariant spherical convolution on HEALPix maps.

Optimisation summary vs original
─────────────────────────────────────────────────────────────────────────────
 Location                   Original                 Optimised
─────────────────────────────────────────────────────────────────────────────
 _get_interp_weights        chunk loop + numpy↔torch  single vectorised call,
                            round-trips                pure torch index ops
 _bind_support (init)       for g in G: separate call  _bind_support_batched:
                                                        one searchsorted over
                                                        G*4*K*P at once
 forward gather (G loop)    for g in G: index_select  single index_select on
                            + sum(4 neighbours)        pos flat [G*4*K*P],
                            + einsum  → G serial       reshape, 1 einsum  (no
                                                        Python loop at all)
─────────────────────────────────────────────────────────────────────────────
 Differentiability: 100 % preserved.
 All hot paths are pure torch ops → autograd works end-to-end.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import healpix_geo
from healpix_analyse.healpix_interp import get_interp_weights

ArrayLike = Union[np.ndarray, torch.Tensor]
    
# ===========================================================================
# I/O helpers  (unchanged)
# ===========================================================================

def _prepare_input_conv(x, device, dtype):
    """Normalise input to [B, C, N]. Returns (tensor, is_numpy, was_1d, was_2d)."""
    is_numpy = isinstance(x, np.ndarray)
    t = torch.as_tensor(x, dtype=dtype, device=device) if is_numpy \
        else x.to(device=device, dtype=dtype)

    was_1d = (t.ndim == 1)
    was_2d = (t.ndim <= 2)

    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 2:
        t = t.unsqueeze(1)
    elif t.ndim != 3:
        raise ValueError(
            f"Input must have shape [N], [B, N] or [B, C, N]; got {tuple(t.shape)}"
        )
    return t, is_numpy, was_1d, was_2d


def _restore_output_conv(t, is_numpy, was_1d):
    """Convert [B, C_out, N] back to the original shape / type."""
    if was_1d:
        t = t.squeeze(0)
        if t.shape[0] == 1:
            t = t.squeeze(0)
    if is_numpy:
        return t.detach().cpu().numpy()
    return t


# ===========================================================================
# Geometry helpers  (unchanged)
# ===========================================================================

def _build_rotation_matrices(th, ph, G, gauge_type, device, dtype, ref_direction=None):
    """
    Build rotation matrices [K, G, 3, 3] carrying the North-Pole kernel
    grid to each of the K target pixels with G gauge angles.

    R_total = R_gauge(alpha_g) @ Rz(phi) @ Ry(theta)

    Gauge types
    -----------
    "phi"
        alpha_base = 0 everywhere.  Singularities at the geographic poles.

    "cosmo"
        Same singularities as "phi" but the gauge angle flips sign across
        the equator to match the cosmological convention.

    "projected_ref"
        ref_direction is a single unit vector r (shape (3,)).
        alpha_base = atan2(r_proj·e_φ,  r_proj·e_θ)
        where r_proj = r - (r·n)·n  is the projection of r onto the
        tangent plane at n.
        Singularities at {+r, −r}  (antipodal pair), total index = 2.

    "two_ref"
        ref_direction is TWO unit vectors (shape (2, 3)): r1 and r2.
        The gauge angle is defined as the argument of the complex PRODUCT
        of the two projected tangent vectors:

            z_j(n) = (r_j_proj·e_θ) + i·(r_j_proj·e_φ)   j = 1, 2
            alpha_base(n) = arg(z1(n) · z2(n))
                          = atan2(Re(z1)·Im(z2) + Im(z1)·Re(z2),
                                  Re(z1)·Re(z2) − Im(z1)·Im(z2))

        Using the complex product is equivalent to arg(z1) + arg(z2), but
        avoids two separate atan2 calls and never wraps independently.

        Singularity structure (Poincaré-Hopf, total index must = 2):
          • +r1, −r1 : index +1 each   (zeros of z1)
          • +r2, −r2 : index +1 each   (zeros of z2)
          • N-Pole, S-Pole : index −1 each  (base-frame singularities
                             absorbed by z1·z2, but each of z1 and z2 winds
                             −1 there, so the poles get net index 1 − 2 = −1)
          Total: 4×(+1) + 2×(−1) = 2  ✓

        In practice the four user-chosen bad points {+r1, −r1, +r2, −r2}
        are index +1 (well-behaved to avoid); the poles become index −1
        (hyperbolic singularity — keep them away from the domain of interest
        or over regions the network need not be accurate in).
    """
    th = np.asarray(th, dtype=np.float64).reshape(-1)
    ph = np.asarray(ph, dtype=np.float64).reshape(-1)
    K  = th.shape[0]

    th_t = torch.as_tensor(th, device=device, dtype=dtype)
    ph_t = torch.as_tensor(ph, device=device, dtype=dtype)
    ct, st = torch.cos(th_t), torch.sin(th_t)
    cp, sp = torch.cos(ph_t), torch.sin(ph_t)

    R_base = torch.zeros(K, 3, 3, device=device, dtype=dtype)
    R_base[:, 0, 0] =  cp * ct;  R_base[:, 0, 1] = -sp;  R_base[:, 0, 2] =  cp * st
    R_base[:, 1, 0] =  sp * ct;  R_base[:, 1, 1] =  cp;  R_base[:, 1, 2] =  sp * st
    R_base[:, 2, 0] = -st;       R_base[:, 2, 1] = 0.;   R_base[:, 2, 2] =  ct

    n = R_base[:, :, 2]
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

    if gauge_type == "cosmo":
        is_south   = th_t > math.pi / 2
        alpha_base = torch.where(is_south,  ph_t, -ph_t)
        sign_g     = torch.where(is_south,
                                 -torch.ones_like(th_t),
                                  torch.ones_like(th_t))

    elif gauge_type in ("projected_ref", "two_ref"):
        # Local orthonormal tangent frame at each pixel
        e_th  = torch.stack([ ct * cp,  ct * sp, -st],               dim=1)  # [K, 3]
        e_ph  = torch.stack([-sp,        cp,      torch.zeros_like(st)], dim=1)  # [K, 3]
        n_pix = torch.stack([ st * cp,  st * sp,  ct],               dim=1)  # [K, 3]

        def _tangent_complex(r_vec):
            """Project unit vector r onto the tangent plane → (r_eth, r_eph)."""
            r  = torch.as_tensor(r_vec, device=device, dtype=dtype)
            r  = r / r.norm().clamp_min(1e-12)
            r_dot_n  = (r[None, :] * n_pix).sum(dim=1, keepdim=True)   # [K, 1]
            r_proj   = r[None, :] - r_dot_n * n_pix                     # [K, 3]
            r_eth    = (r_proj * e_th).sum(dim=1)                        # [K]
            r_eph    = (r_proj * e_ph).sum(dim=1)                        # [K]
            return r_eth, r_eph                                           # Re(z), Im(z)

        if gauge_type == "projected_ref":
            # Single reference vector → singularities at {+r, −r}
            if ref_direction is None:
                ref_direction = [1.0, 0.0, 0.0]
            eth, eph   = _tangent_complex(ref_direction)
            alpha_base = torch.atan2(eph, eth)

        else:  # "two_ref"
            # Two reference vectors → singularities at {+r1,−r1,+r2,−r2}
            # alpha = arg(z1 · z2)  via complex product, never wraps
            eth1, eph1 = _tangent_complex(ref_direction[0])   # Re(z1), Im(z1)
            eth2, eph2 = _tangent_complex(ref_direction[1])   # Re(z2), Im(z2)
            re_prod    = eth1 * eth2 - eph1 * eph2            # Re(z1·z2)
            im_prod    = eth1 * eph2 + eph1 * eth2            # Im(z1·z2)
            alpha_base = torch.atan2(im_prod, re_prod)

        sign_g = torch.ones_like(th_t)

    else:  # "phi"
        alpha_base = torch.zeros_like(th_t)
        sign_g     = torch.ones_like(th_t)

    g_shifts = torch.arange(G, device=device, dtype=dtype) * (math.pi / G)
    alpha_g  = alpha_base[:, None] + sign_g[:, None] * g_shifts[None, :]
    ca = torch.cos(alpha_g);  sa = torch.sin(alpha_g)

    n_g  = n[:, None, :].expand(K, G, 3)
    nxg, nyg, nzg = n_g[..., 0], n_g[..., 1], n_g[..., 2]

    K_skew = torch.zeros(K, G, 3, 3, device=device, dtype=dtype)
    K_skew[..., 0, 1] = -nzg;  K_skew[..., 0, 2] =  nyg
    K_skew[..., 1, 0] =  nzg;  K_skew[..., 1, 2] = -nxg
    K_skew[..., 2, 0] = -nyg;  K_skew[..., 2, 1] =  nxg

    outer  = n_g.unsqueeze(-1) * n_g.unsqueeze(-2)
    I      = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    R_gauge = (
        I      * ca.view(K, G, 1, 1)
        + K_skew * sa.view(K, G, 1, 1)
        + outer  * (1.0 - ca).view(K, G, 1, 1)
    )

    R_tot = torch.matmul(R_gauge, R_base[:, None, :, :].expand(K, G, 3, 3))
    return R_tot   # [K, G, 3, 3]


def _local_kernel_grid(kernel_sz, nside):
    """
    Build a kernel_sz × kernel_sz grid of unit vectors at the North Pole.
    Returns np.ndarray [P=kernel_sz^2, 3].
    """
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    alpha_pix = np.sqrt(4 * np.pi / (12 * nside ** 2))

    dtheta = np.sqrt(xx ** 2 + yy ** 2).ravel() * alpha_pix
    dphi   = np.arctan2(yy, xx).ravel()

    x = np.sin(dtheta) * np.cos(dphi)
    y = np.sin(dtheta) * np.sin(dphi)
    z = np.cos(dtheta)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


# ===========================================================================
# Optimised helper 1 — _get_interp_weights
# ===========================================================================

def _get_interp_weights(nside, vecs, nest, device, dtype):
    """
    Compute bilinear-interpolation neighbours for M direction vectors.

    Optimisation vs original
    ────────────────────────
    • Removed unnecessary chunk loop — ``get_interp_weights`` is already
      fully vectorised; chunking only added Python overhead.
    • Kept the numpy round-trip minimal: one call in, two arrays out,
      immediately transferred to the target device as torch tensors.

    Parameters
    ----------
    vecs : torch.Tensor [M, 3]

    Returns
    -------
    idx_t : LongTensor [4, M]
    w_t   : Tensor     [4, M]
    """
    # Normalise direction vectors (torch, no grad needed here)
    vn    = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(vn[:, 2].clamp(-1., 1.))
    phi   = torch.atan2(vn[:, 1], vn[:, 0]) % (2.0 * math.pi)

    # Convert to lon/lat in degrees (numpy, CPU — healpix_geo expects numpy)
    lon_np = np.rad2deg(phi.detach().cpu().numpy())     # (M,)
    lat_np = 90.0 - np.rad2deg(theta.detach().cpu().numpy())  # (M,)

    # Single vectorised call — already handles arbitrary M
    depth = int(math.log2(nside))
    i_np, w_np = get_interp_weights(lon_np, lat_np, depth)
    # i_np, w_np: (M, 4) → transpose to (4, M)

    return (
        torch.as_tensor(i_np.T.copy(), device=device, dtype=torch.long),
        torch.as_tensor(w_np.T.copy(), device=device, dtype=dtype),
    )


# ===========================================================================
# Optimised helper 2 — _bind_support_batched
# ===========================================================================

def _bind_support_batched(
    idx_t:       torch.Tensor,   # [G, 4, K*P]
    w_t:         torch.Tensor,   # [G, 4, K*P]
    ids_sorted:  torch.Tensor,   # [N_in] sorted pixel ids
    kernel_sz:   int,
    K:           int,
    P:           int,
    device:      torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised binding of stencil neighbours for ALL G gauges at once.

    Replaces the original ``for g in G: _bind_support(...)`` loop.

    Optimisation vs original
    ────────────────────────
    • One ``torch.searchsorted`` over [G*4*K*P] elements instead of G
      separate calls over [4*K*P] each.
    • One ``present`` boolean tensor computed in one shot.
    • The fallback for empty stencil points is rare and still loops over
      the (typically small) number of affected gauges, but the main path
      is fully batched.

    Parameters
    ----------
    idx_t      : [G, 4, K*P]  absolute NESTED pixel ids of the 4 neighbours
    w_t        : [G, 4, K*P]  bilinear weights (contiguous copy expected)
    ids_sorted : [N_in]        sorted pixel ids of the input patch
    kernel_sz, K, P, device

    Returns
    -------
    pos_safe : [G, 4, K*P]   column indices in ids_sorted (0 for absent)
    w_norm   : [G, 4, K*P]   renormalised weights
    """
    G = idx_t.shape[0]
    M = K * P
    N_sorted = ids_sorted.numel()

    # ------------------------------------------------------------------
    # 1. Searchsorted over ALL (G * 4 * M) entries at once
    # ------------------------------------------------------------------
    pos = torch.searchsorted(ids_sorted, idx_t.reshape(-1)).view(G, 4, M)

    # ------------------------------------------------------------------
    # 2. Presence mask — one shot for all G gauges
    # ------------------------------------------------------------------
    in_range = pos < N_sorted
    cmp_vals = torch.full_like(idx_t, -1)
    cmp_vals[in_range] = ids_sorted[pos[in_range]]
    present  = cmp_vals == idx_t          # [G, 4, M]

    # ------------------------------------------------------------------
    # 3. Fallback: stencil points with zero present neighbours
    #    → replace with center stencil point of the same pixel
    #    (rare; still loops over G but avoids redundant searchsorted)
    # ------------------------------------------------------------------
    p_ref  = (kernel_sz // 2) * (kernel_sz + 1)
    empty  = ~present.any(dim=1)          # [G, M]

    if empty.any():
        k_id     = torch.div(
            torch.arange(M, device=device), P, rounding_mode="floor"
        )                                 # [M]
        ref_cols = (k_id * P + p_ref)    # [M]  centre col per stencil pt

        for g in range(G):
            empty_g = empty[g]           # [M] bool
            if not empty_g.any():
                continue

            ref_g = ref_cols[empty_g]
            idx_t[g, :, empty_g] = idx_t[g, :, ref_g]
            w_t  [g, :, empty_g] = w_t  [g, :, ref_g]

            idx_e  = idx_t[g, :, empty_g].reshape(-1)
            pos_e  = torch.searchsorted(ids_sorted, idx_e)
            valid_e = pos_e < N_sorted
            pos_e_c = pos_e.clamp(0, max(N_sorted - 1, 0))
            pres_e  = valid_e & (ids_sorted[pos_e_c] == idx_e)
            present[g, :, empty_g] = pres_e.view(4, -1)
            pos    [g, :, empty_g] = pos_e_c.view(4, -1)

    # ------------------------------------------------------------------
    # 4. Weight renormalisation — batched over G
    # ------------------------------------------------------------------
    w      = w_t * present                              # [G, 4, M]
    colsum = w.sum(dim=1, keepdim=True)                 # [G, 1, M]
    zero_c = colsum == 0                                # [G, 1, M]

    if zero_c.any():
        # For zero-sum columns use the first present neighbour with weight 1
        zero_mask = zero_c.squeeze(1)                   # [G, M]
        w[:, 0, :] = torch.where(
            zero_mask, present[:, 0, :].to(w.dtype), w[:, 0, :]
        )
        colsum = w.sum(dim=1, keepdim=True)

    w_norm   = w / colsum.clamp_min(1e-12)
    pos_safe = torch.where(present, pos, torch.zeros_like(pos))
    return pos_safe, w_norm                             # [G, 4, M], [G, 4, M]


# ===========================================================================
# HealPixConv
# ===========================================================================

class HealPixConv(nn.Module):
    """
    Gauge-equivariant spherical convolution on HEALPix maps.

    Algorithm
    ---------
    The convolution is precomputed at construction time in three stages.

    **Stage A — Kernel + rotation**

    A ``kernel_sz × kernel_sz`` grid of P unit vectors is defined at the
    **North Pole** (z = 1) with angular spacing equal to one HEALPix pixel
    width.  This is the stencil template; it never changes.

    For every target pixel k (colatitude θ_k, longitude φ_k) and every
    gauge g, a rotation matrix is built::

        R_total[k,g] = R_gauge(α_g)  @  Rz(φ_k)  @  Ry(θ_k)
                       └─ gauge roll ┘  └────── carry N.Pole → pixel k ──┘

    Each of the P kernel vectors is then rotated into its position on the
    sphere around pixel k::

        rotated[k, g, p] = R_total[k, g] @ vec_pole[p]   ∈ ℝ³

    **Stage B — Bilinear binding**

    For each of the K × G × P rotated directions, ``get_interp_weights``
    returns the 4 nearest HEALPix neighbours and their bilinear weights.
    These indices and weights are stored as buffers (precomputed once).

    **Stage C — Forward pass (data → kernel)**

    *It is the data that is brought to the (fixed) kernel, not the kernel
    that moves.*  At inference::

        x_interp[b, c, g, k, p] = Σ_{j=0}^{3}  w[g,j,k,p] · x[b, c, nbr[g,j,k,p]]

    i.e. the signal value at rotated stencil point p of pixel k (gauge g)
    is obtained by bilinear interpolation of the input map.  Then::

        y[b, g·C_out + o, k] = Σ_{c,p}  W[g, c, o, p] · x_interp[b, c, g, k, p]

    The learned kernel ``W[G, C_in, C_out, P]`` is a fixed set of weights;
    its P positions implicitly correspond to the stencil template at the
    North Pole.  Gauge equivariance is achieved because the *same* kernel
    is applied after the stencil has been rotated by the gauge.

    Gauge types and singularities
    ------------------------------
    Every smooth gauge on S² has exactly two singular points (hairy-ball
    theorem).  The three built-in gauges place them differently:

    ``"phi"``
        Singularities at the **geographic poles** (θ = 0, π).
        ``α_base = 0`` everywhere → kernel always meridian-aligned.
        Simple and fast; bad only at the poles.

    ``"cosmo"``
        Singularities also at the poles, but the gauge flips sign across
        the equator to match the cosmological convention.

    ``"projected_ref"``
        A reference vector **r** is projected onto the tangent plane; the
        gauge angle is ``atan2(r·e_φ, r·e_θ)``.  The two singularities are
        the antipodal points where **r** is parallel to the surface normal,
        i.e. where the map pixel lies exactly on the direction of **r**::

            singularity₁ = (lon_s, lat_s)
            singularity₂ = (lon_s + 180°, -lat_s)   ← antipode

        You control where those two points land by choosing **r**.
        Use ``singularity_lonlat=(lon_s, lat_s)`` to specify the first
        singularity in geographic coordinates; the reference vector is
        computed automatically::

            r = [cos(lat_s)·cos(lon_s),
                 cos(lat_s)·sin(lon_s),
                 sin(lat_s)]

        Practical choices:
          - Ocean model  → place singularities over land (e.g. Amazon basin
            and its antipode in the Indian Ocean).
          - Atmosphere   → place singularities over open ocean (e.g. central
            Pacific and its antipode in the Atlantic).
          - Full sphere  → place singularities at the geographic poles
            (``singularity_lonlat=(0, 90)``), which reproduces the ``"phi"``
            gauge but with the smooth-everywhere property off the poles.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    in_channels : int
        Number of input channels C_in.
    out_channels : int
        Number of output channels per gauge C_out.
        Total output channels = n_gauges × out_channels.
    kernel_sz : int, default 3
        Odd integer ≥ 1.  P = kernel_sz² stencil points.
    n_gauges : int, default 1
        Number of gauge orientations G.  Each gauge rotates the stencil
        by an additional angle of π/G relative to the previous one.
    gauge_type : {"phi", "cosmo", "projected_ref"}, default "phi"
        Gauge convention.  See *Gauge types and singularities* above.
    singularity_lonlat : (float, float) or None
        Only used when ``gauge_type="projected_ref"``.
        Geographic coordinates ``(longitude_deg, latitude_deg)`` of the
        **first** singularity point.  The second singularity is placed
        automatically at the antipodal point.
        Overrides ``ref_direction`` when both are provided.
        Example — singularities over the Himalayas and the central Pacific::

            singularity_lonlat=(84.0, 28.0)   # Kathmandu region

    ref_direction : array-like (3,) or None
        Low-level alternative to ``singularity_lonlat``: pass the reference
        3-D unit vector **r** directly.  Ignored when ``singularity_lonlat``
        is provided.  Default ``[1, 0, 0]`` → singularities at
        (lon=0°, lat=0°) and (lon=180°, lat=0°).
    cell_ids : array-like or None
        Pixel indices (NESTED) for partial-sky.  None = full sphere.
    level : int or None
        nside = 2**level.  Required when cell_ids is provided.
    nest : bool, default True
        NESTED pixel ordering.
    use_norm : bool, default False
        Apply GroupNorm + ReLU after convolution.
    ellipsoid : str, default "WGS84"
        Reference ellipsoid for healpix_geo coordinate conversions.
    device, dtype : torch device / dtype.

    Examples
    --------
    Ocean model — singularities over Africa and its antipode (Pacific):

    >>> conv = HealPixConv(
    ...     nside=64, in_channels=1, out_channels=16,
    ...     gauge_type="projected_ref",
    ...     singularity_lonlat=(20.0, 5.0),   # Gulf of Guinea coast
    ... )

    Atmosphere — singularities over the Pacific and Indian oceans:

    >>> conv = HealPixConv(
    ...     nside=64, in_channels=1, out_channels=16,
    ...     gauge_type="projected_ref",
    ...     singularity_lonlat=(-160.0, 0.0),  # central Pacific
    ... )
    """

    def __init__(
        self,
        nside: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[tuple[float, float]] = None,
        ref_direction=None,
        cell_ids=None,
        level=None,
        nest: bool = True,
        use_norm: bool = False,
        device=None,
        ellipsoid: str = "WGS84",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype  = dtype
        self.ellipsoid = ellipsoid

        self.nside        = int(nside)
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_sz    = int(kernel_sz)
        self.G            = int(max(1, n_gauges))
        self.P            = self.kernel_sz * self.kernel_sz
        self.nest         = bool(nest)

        if (self.nside & (self.nside - 1)) != 0 or self.nside < 1:
            raise ValueError("nside must be a positive power of 2.")
        if self.kernel_sz < 1 or self.kernel_sz % 2 == 0:
            raise ValueError("kernel_sz must be a positive odd integer.")
        if gauge_type not in ("phi", "cosmo", "projected_ref", "two_ref"):
            raise ValueError(
                "gauge_type must be 'phi', 'cosmo', 'projected_ref', or 'two_ref'."
            )
        self.gauge_type = gauge_type

        # ------------------------------------------------------------------
        # Resolve reference direction(s).
        #
        # "projected_ref"  :  one singularity point  →  ref_direction (3,)
        #                     second singularity = antipode (forced)
        #
        # "two_ref"        :  two freely-placed singularity points
        #                     →  ref_direction  (2, 3)
        #
        #   Construction:  alpha = arg( z1(n) · z2(n) )
        #   where  z_j(n) = r_j_proj · (e_θ + i·e_φ).
        #
        #   Topological budget (Poincaré-Hopf, total index = 2):
        #     +r1, -r1, +r2, -r2 : index +1 each    (4 user-controlled pts)
        #     N-Pole, S-Pole     : index -1 each     (unavoidable side-effect)
        #     4 - 2 = 2  ✓
        #
        #   Strategy: place ALL FOUR of {p1, antipode(p1), p2, antipode(p2)}
        #   over land (for ocean) or over ocean (for atmosphere).
        #   Also keep the geographic poles away from the domain of interest.
        # ------------------------------------------------------------------
        if singularity_lonlat is not None:
            if gauge_type == "projected_ref":
                # Single (lon, lat) → one reference vector, antipodal pair
                lon_s_deg, lat_s_deg = float(singularity_lonlat[0]), float(singularity_lonlat[1])
                lon_s = np.radians(lon_s_deg)
                lat_s = np.radians(lat_s_deg)
                rd = np.array([
                    np.cos(lat_s) * np.cos(lon_s),
                    np.cos(lat_s) * np.sin(lon_s),
                    np.sin(lat_s),
                ], dtype=np.float64)
                self.ref_direction = rd / np.linalg.norm(rd)
                self.singularity_1 = (lon_s_deg % 360.0, lat_s_deg)
                self.singularity_2 = ((lon_s_deg + 180.0) % 360.0, -lat_s_deg)

            elif gauge_type == "two_ref":
                # Sequence of two (lon, lat) pairs → two reference vectors
                if len(singularity_lonlat) != 2:
                    raise ValueError(
                        "For gauge_type='two_ref', singularity_lonlat must be a "
                        "sequence of exactly two (lon, lat) pairs, e.g. "
                        "[(lon1, lat1), (lon2, lat2)]."
                    )

                def _lonlat_to_vec(lon_deg, lat_deg):
                    lon_r, lat_r = np.radians(lon_deg), np.radians(lat_deg)
                    v = np.array([
                        np.cos(lat_r) * np.cos(lon_r),
                        np.cos(lat_r) * np.sin(lon_r),
                        np.sin(lat_r),
                    ], dtype=np.float64)
                    return v / np.linalg.norm(v)

                def _vec_to_lonlat(v):
                    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
                    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
                    return lon, lat

                r1 = _lonlat_to_vec(*singularity_lonlat[0])
                r2 = _lonlat_to_vec(*singularity_lonlat[1])
                self.ref_direction = np.stack([r1, r2], axis=0)   # (2, 3)

                # Store all four singularity locations for introspection
                self.singularity_1  = _vec_to_lonlat(r1)
                self.singularity_1b = _vec_to_lonlat(-r1)   # antipode of s1
                self.singularity_2  = _vec_to_lonlat(r2)
                self.singularity_2b = _vec_to_lonlat(-r2)   # antipode of s2

            else:
                raise ValueError(
                    "singularity_lonlat is only valid for "
                    "gauge_type='projected_ref' or 'two_ref'."
                )

        elif ref_direction is not None:
            if gauge_type == "two_ref":
                rd = np.asarray(ref_direction, dtype=np.float64)
                if rd.shape != (2, 3):
                    raise ValueError(
                        "For gauge_type='two_ref', ref_direction must have "
                        f"shape (2, 3); got {rd.shape}."
                    )
                rd[0] /= np.linalg.norm(rd[0])
                rd[1] /= np.linalg.norm(rd[1])
                self.ref_direction = rd
                def _v2ll(v):
                    lat = np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))
                    lon = np.degrees(np.arctan2(v[1], v[0])) % 360.0
                    return lon, lat
                self.singularity_1  = _v2ll(rd[0])
                self.singularity_1b = _v2ll(-rd[0])
                self.singularity_2  = _v2ll(rd[1])
                self.singularity_2b = _v2ll(-rd[1])
            else:
                rd = np.asarray(ref_direction, dtype=np.float64).ravel()
                self.ref_direction = rd / np.linalg.norm(rd)
                r = self.ref_direction
                lat_s = np.degrees(np.arcsin(np.clip(r[2], -1.0, 1.0)))
                lon_s = np.degrees(np.arctan2(r[1], r[0])) % 360.0
                self.singularity_1 = (lon_s, lat_s)
                self.singularity_2 = ((lon_s + 180.0) % 360.0, -lat_s)
        else:
            if gauge_type == "two_ref":
                # Default: r1 = [1,0,0], r2 = [0,1,0]
                self.ref_direction = np.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
                )
                self.singularity_1  = (0.0,  0.0)
                self.singularity_1b = (180.0, 0.0)
                self.singularity_2  = (90.0,  0.0)
                self.singularity_2b = (270.0, 0.0)
            else:
                # Default projected_ref: r = [1, 0, 0]
                self.ref_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                self.singularity_1 = (0.0,   0.0)
                self.singularity_2 = (180.0, 0.0)

        # ---- pixel domain ----
        self.partial = cell_ids is not None
        if self.partial:
            if level is None:
                raise ValueError("level required with cell_ids (nside = 2**level).")
            if 2 ** int(level) != self.nside:
                raise ValueError(f"2**level={2**level} != nside={self.nside}.")
            ids_np = np.asarray(cell_ids, dtype=np.int64).ravel()
        else:
            ids_np = np.arange(12 * self.nside ** 2, dtype=np.int64)
        self.K = len(ids_np)

        # ---- Stage A: geometry — healpix_to_lonlat already vectorised ----
        lon, lat = healpix_geo.nested.healpix_to_lonlat(
            ids_np.tolist(), int(np.log2(self.nside)),
            ellipsoid=self.ellipsoid,
        )
        th = np.deg2rad(90.0 - np.asarray(lat, dtype=np.float64))
        ph = np.deg2rad(np.asarray(lon, dtype=np.float64))

        R_tot = _build_rotation_matrices(
            th, ph, self.G, self.gauge_type, self.device, self.dtype,
            ref_direction=self.ref_direction,
        )  # [K, G, 3, 3]

        vec_t = torch.as_tensor(
            _local_kernel_grid(self.kernel_sz, self.nside),
            device=self.device, dtype=self.dtype,
        )  # [P, 3]

        # Rotate stencil: [K, G, P, 3]
        rotated = torch.einsum("kgij,pj->kgpi", R_tot, vec_t)
        flat    = rotated.reshape(-1, 3)   # [K*G*P, 3]

        # Single vectorised call — no chunking overhead
        idx_flat, w_flat = _get_interp_weights(
            self.nside, flat, self.nest, self.device, self.dtype
        )  # [4, K*G*P]

        # Reshape to [G, 4, K*P]  — already the layout _bind_support_batched expects
        idx_all = (
            idx_flat.view(4, self.K, self.G, self.P)
                    .permute(2, 0, 1, 3)
                    .reshape(self.G, 4, self.K * self.P)
        )
        w_all = (
            w_flat.view(4, self.K, self.G, self.P)
                  .permute(2, 0, 1, 3)
                  .reshape(self.G, 4, self.K * self.P)
        )

        # ---- Stage B: binding — one batched call instead of G calls ----
        ids_sorted   = np.sort(ids_np)
        ids_sorted_t = torch.as_tensor(
            ids_sorted, device=self.device, dtype=torch.long
        )

        sort_order = np.argsort(ids_np)
        inv_order  = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(len(sort_order))
        self.register_buffer(
            "_sort_order",
            torch.as_tensor(sort_order, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_inv_order",
            torch.as_tensor(inv_order, dtype=torch.long, device=self.device),
        )

        # Single batched call over all G gauges
        pos_all, w_all_norm = _bind_support_batched(
            idx_all.clone(), w_all.clone(),
            ids_sorted_t, self.kernel_sz, self.K, self.P, self.device,
        )  # [G, 4, K*P] each

        self.register_buffer("_pos_safe", pos_all)    # [G, 4, K*P]
        self.register_buffer("_w_norm",   w_all_norm) # [G, 4, K*P]

        # ---- learnable kernel and bias ----
        self.weight = nn.Parameter(
            torch.empty(
                self.G, self.in_channels, self.out_channels, self.P,
                device=self.device, dtype=self.dtype,
            )
        )
        nn.init.kaiming_uniform_(
            self.weight.view(
                self.G * self.in_channels, self.out_channels * self.P
            ),
            a=0., mode="fan_in", nonlinearity="relu",
        )
        self.bias = nn.Parameter(
            torch.zeros(
                self.G * self.out_channels,
                device=self.device, dtype=self.dtype,
            )
        )

        # ---- optional GroupNorm + ReLU ----
        self.use_norm = bool(use_norm)
        if self.use_norm:
            C_tot = self.G * self.out_channels
            g = min(8, C_tot)
            while C_tot % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(g, C_tot)
        else:
            self.norm = None

        self.to(self.device)

    # ------------------------------------------------------------------
    # Kernel management  (unchanged)
    # ------------------------------------------------------------------

    def set_kernel(self, W, bias=None, requires_grad=False):
        """
        Replace the learnable kernel with a fixed (or re-initialised) array.

        Parameters
        ----------
        W : array-like
            Shape [C_in, C_out, P]       — same kernel broadcast over all G.
            Shape [G, C_in, C_out, P]    — per-gauge kernels.
        bias : array-like or None
            Shape [G * C_out].  None resets bias to zero.
        requires_grad : bool, default False
        """
        W_np = np.asarray(W, dtype=np.float32)
        if W_np.ndim == 3:
            W_np = np.broadcast_to(W_np[None], (self.G,) + W_np.shape).copy()
        expected = (self.G, self.in_channels, self.out_channels, self.P)
        if W_np.shape != expected:
            raise ValueError(
                f"W must have shape {expected} or "
                f"({self.in_channels}, {self.out_channels}, {self.P}); "
                f"got {W_np.shape}."
            )
        with torch.no_grad():
            self.weight.copy_(
                torch.as_tensor(W_np, dtype=self.dtype, device=self.device)
            )
            if bias is not None:
                self.bias.copy_(
                    torch.as_tensor(
                        np.asarray(bias, np.float32).ravel(),
                        dtype=self.dtype, device=self.device,
                    )
                )
            else:
                self.bias.zero_()
        self.weight.requires_grad_(requires_grad)
        self.bias.requires_grad_(requires_grad)
        return self

    # ------------------------------------------------------------------
    # Forward — fully vectorised over G  (no Python loop)
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Apply gauge-equivariant spherical convolution.

        Parameters
        ----------
        x : array-like, shape [N], [B, N] or [B, C_in, N]

        Returns
        -------
        y : same type, shape [G*C_out, N] or [B, G*C_out, N]

        Optimisation vs original
        ────────────────────────
        Original::

            for g in range(G):
                gathered = sum(index_select(...) * w for j in 4)
                yg = einsum("bckp,cop->bok", gathered, W[g])
                outs.append(yg)
            y = torch.cat(outs, dim=1)

        Optimised: all G gauges handled in one batched gather + one einsum.

            pos_flat = pos.reshape(-1)            # [G*4*K*P]
            vals     = t_sorted[:, :, pos_flat]   # [B, C_in, G*4*K*P]
            gathered = (vals * w).sum(dim=3)       # [B, C_in, G, K, P]
            y        = einsum("bcgkp,gcop->bgok", gathered, W)
                                                  # [B, G, C_out, K]

        This replaces G serial torch ops with a single gather + einsum,
        which fuses much better on both CPU and GPU.
        All operations are differentiable.
        """
        t, is_numpy, was_1d, _ = _prepare_input_conv(x, self.device, self.dtype)
        B, C_in, N = t.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, got {C_in}."
            )
        if N != self.K:
            raise ValueError(f"Expected {self.K} pixels, got {N}.")

        so  = self._sort_order.to(device=t.device)        # [K]
        io  = self._inv_order.to(device=t.device)         # [K]
        pos = self._pos_safe.to(device=t.device)          # [G, 4, K*P]
        wn  = self._w_norm.to(device=t.device, dtype=t.dtype)   # [G, 4, K*P]
        W   = self.weight.to(device=t.device, dtype=t.dtype)    # [G, C_in, C_out, P]

        G, P, K = self.G, self.P, self.K

        # Sort pixels for searchsorted-aligned indexing
        t_sorted = t[:, :, so]                            # [B, C_in, K]

        # ------------------------------------------------------------------
        # Batched gather over all G gauges and all 4 neighbours
        # ------------------------------------------------------------------
        # pos:  [G, 4, K*P]  →  flatten to  [G*4*K*P]
        # Then index t_sorted once and reshape.
        pos_flat  = pos.reshape(-1)                       # [G*4*K*P]
        vals_flat = t_sorted.index_select(2, pos_flat)   # [B, C_in, G*4*K*P]
        vals      = vals_flat.view(B, C_in, G, 4, K, P)  # [B, C_in, G, 4, K, P]

        # Weighted sum over the 4 bilinear neighbours
        # wn: [G, 4, K*P] → [G, 4, K, P] for broadcasting
        w_shaped  = wn.view(G, 4, K, P)                  # [G, 4, K, P]
        gathered  = (vals * w_shaped[None, None]).sum(dim=3)
        # gathered: [B, C_in, G, K, P]  (dim=3 is the "4 neighbours" dim)

        # ------------------------------------------------------------------
        # Single einsum over all gauges simultaneously
        # original: G separate "bckp,cop->bok"
        # new:      one        "bcgkp,gcop->bgok"
        # ------------------------------------------------------------------
        y = torch.einsum("bcgkp,gcop->bgok", gathered, W)   # [B, G, C_out, K]
        y = y.reshape(B, G * self.out_channels, K)           # [B, G*C_out, K]

        # Bias + unsort
        y = y + self.bias.to(device=t.device, dtype=t.dtype).view(1, -1, 1)
        y = y[:, :, io]                                      # [B, G*C_out, K]

        if self.use_norm and self.norm is not None:
            nm = self.norm.to(device=t.device, dtype=t.dtype)
            y  = F.relu(nm(y), inplace=True)

        return _restore_output_conv(y, is_numpy, was_1d)

    def singularity_info(self) -> str:
        """
        Return a human-readable description of the gauge singularities.

        For "phi" and "cosmo": singularities are always at the geographic poles.
        For "projected_ref":   two antipodal singularities at ±r.
        For "two_ref":         four singularities at {±r1, ±r2} (index +1 each)
                               plus index-(-1) singularities at the geographic poles.
        """
        if self.gauge_type in ("phi", "cosmo"):
            return (
                f"gauge_type='{self.gauge_type}': singularities fixed at "
                "the geographic poles (lat=+90° and lat=-90°)."
            )
        if self.gauge_type == "projected_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            return (
                f"gauge_type='projected_ref':\n"
                f"  singularity 1 : lon={s1[0]:+.2f}°  lat={s1[1]:+.2f}°  (index +1)\n"
                f"  singularity 2 : lon={s2[0]:+.2f}°  lat={s2[1]:+.2f}°  (index +1, antipode)\n"
                f"  ref_direction : {self.ref_direction.tolist()}"
            )
        # "two_ref"
        s1, s1b = self.singularity_1, self.singularity_1b
        s2, s2b = self.singularity_2, self.singularity_2b
        return (
            f"gauge_type='two_ref':\n"
            f"  singularity 1  : lon={s1[0]:+.2f}°  lat={s1[1]:+.2f}°  (index +1, user-defined)\n"
            f"  singularity 1b : lon={s1b[0]:+.2f}°  lat={s1b[1]:+.2f}°  (index +1, antipode of 1)\n"
            f"  singularity 2  : lon={s2[0]:+.2f}°  lat={s2[1]:+.2f}°  (index +1, user-defined)\n"
            f"  singularity 2b : lon={s2b[0]:+.2f}°  lat={s2b[1]:+.2f}°  (index +1, antipode of 2)\n"
            f"  N/S poles      : index -1 each (side-effect, keep outside domain of interest)\n"
            f"  ref_directions : r1={self.ref_direction[0].tolist()}\n"
            f"                   r2={self.ref_direction[1].tolist()}"
        )

    def extra_repr(self):
        base = (
            f"nside={self.nside}, in={self.in_channels}, out={self.out_channels}, "
            f"kernel_sz={self.kernel_sz}, P={self.P}, G={self.G}, "
            f"gauge={self.gauge_type!r}, partial={self.partial}"
        )
        if self.gauge_type == "projected_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            base += (
                f", sing1=({s1[0]:.1f}°,{s1[1]:.1f}°)"
                f", sing2=({s2[0]:.1f}°,{s2[1]:.1f}°)"
            )
        elif self.gauge_type == "two_ref":
            s1, s2 = self.singularity_1, self.singularity_2
            base += (
                f", sing1=({s1[0]:.1f}°,{s1[1]:.1f}°)"
                f", sing2=({s2[0]:.1f}°,{s2[1]:.1f}°)"
            )
        return base
