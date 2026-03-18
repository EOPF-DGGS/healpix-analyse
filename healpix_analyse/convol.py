"""
convol.py
=========
Gauge-equivariant spherical convolution on HEALPix maps.

The convolution is built in three stages precomputed once at construction:

  A) **Geometry** -- a kernel_sz x kernel_sz grid is defined at the North
     Pole in angular coordinates.  Each target pixel gets its own copy of
     this grid, rotated to the pixel (theta, phi) via a gauge rotation
     matrix R_total = R_gauge(alpha_g) @ Rz(phi) @ Ry(theta).

     The gauge angle alpha_g is:
       "phi"   :  alpha_base = 0           (meridian-aligned, good for NWP)
       "cosmo" :  alpha_base = 2*sign*phi  (cosmological convention)
     For n_gauges=G, angle g uses  alpha_base + g * pi/G.

  B) **Binding** -- healpy.get_interp_weights returns 4 bilinear-
     interpolation neighbors per rotated stencil point.  Out-of-patch
     neighbors are zeroed and weights renormalized to 1.  Empty stencil
     points fall back to the center pixel.

  C) **Convolution** -- gathered, interpolated values are contracted with
     the kernel [G, C_in, C_out, P]:

         y[b, g*C_out+o, k] = sum_{c,p}  W[g,c,o,p] * x_interp[b,c,k,p]

Kernel management
-----------------
* Learned (default) : weight is an nn.Parameter updated by autograd.
* Fixed             : call  conv.set_kernel(W)  with a numpy/torch array.

Input shapes accepted:  [N], [B, N], [B, C_in, N]
Output shape:           [G*C_out, N] or [B, G*C_out, N]

Both numpy arrays and torch tensors are accepted; the return type mirrors
the input type.

Dependencies: numpy, torch, healpy.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import healpy as hp
except ImportError as e:
    raise ImportError(
        "healpy is required by healpix_analyse.convol. "
        "Install it with:  pip install healpy"
    ) from e

ArrayLike = Union[np.ndarray, torch.Tensor]


# ===========================================================================
# I/O helpers
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
# Geometry helpers
# ===========================================================================

def _build_rotation_matrices(th, ph, G, gauge_type, device, dtype):
    """
    Build rotation matrices [K, G, 3, 3] that carry the North-Pole kernel
    grid to each of the K target pixels with G gauge angles.

    R_total = R_gauge(alpha_g) @ Rz(phi) @ Ry(theta)

    gauge_type "phi"  : alpha_base = 0
    gauge_type "cosmo": alpha_base = 2 * sign(theta - pi/2) * phi
    """
    th = np.asarray(th, dtype=np.float64).reshape(-1)
    ph = np.asarray(ph, dtype=np.float64).reshape(-1)
    K  = th.shape[0]

    th_t = torch.as_tensor(th, device=device, dtype=dtype)
    ph_t = torch.as_tensor(ph, device=device, dtype=dtype)
    ct, st = torch.cos(th_t), torch.sin(th_t)
    cp, sp = torch.cos(ph_t), torch.sin(ph_t)

    # Base rotation: R_base = Rz(phi) @ Ry(theta)
    R_base = torch.zeros(K, 3, 3, device=device, dtype=dtype)
    R_base[:, 0, 0] =  cp * ct;  R_base[:, 0, 1] = -sp;  R_base[:, 0, 2] =  cp * st
    R_base[:, 1, 0] =  sp * ct;  R_base[:, 1, 1] =  cp;  R_base[:, 1, 2] =  sp * st
    R_base[:, 2, 0] = -st;       R_base[:, 2, 1] = 0.;   R_base[:, 2, 2] =  ct

    # Local normal = third column of R_base (points toward the pixel)
    n = R_base[:, :, 2]
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Gauge base angle
    if gauge_type == "cosmo":
        # Cosmological convention (Delouis et al. 2022, Fig. A.1):
        # The kernel is meridian-aligned everywhere (no rotation with longitude),
        # but its direction is FLIPPED by 180° between the two hemispheres.
        # This hemisphere inversion is the only angular effect:
        #   North (θ ≤ π/2) :  alpha_base = 0      (no rotation)
        #   South (θ > π/2) :  alpha_base = π      (180° flip)
        #
        # Why NOT alpha = ±φ:
        #   R_base = Rz(φ) @ Ry(θ) already carries the local frame as φ varies.
        #   Adding alpha = ±φ on top compounds with the Rz(φ) in R_base and
        #   produces a rotation rate of ±2φ → the kernel completes two full
        #   turns per longitude circle instead of zero.
        alpha_base = torch.where(
            th_t > (math.pi / 2),
            torch.full_like(th_t, math.pi),   # South: 180° flip
            torch.zeros_like(th_t),            # North: meridian-aligned
        )
    else:
        alpha_base = torch.zeros_like(th_t)

    # G gauge angles: [K, G]
    g_shifts = torch.arange(G, device=device, dtype=dtype) * (math.pi / G)
    alpha_g  = alpha_base[:, None] + g_shifts[None, :]
    ca = torch.cos(alpha_g);  sa = torch.sin(alpha_g)

    # Rodrigues rotation around n by alpha_g
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
    Build a kernel_sz x kernel_sz grid of unit vectors at the North Pole.

    Angular offsets are proportional to hp.nside2resol(nside).

    Returns
    -------
    np.ndarray  [P=kernel_sz^2, 3]
    """
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    alpha_pix = hp.nside2resol(nside, arcmin=False)

    dtheta = np.sqrt(xx**2 + yy**2).ravel() * alpha_pix
    dphi   = np.arctan2(yy, xx).ravel()

    x = np.sin(dtheta) * np.cos(dphi)
    y = np.sin(dtheta) * np.sin(dphi)
    z = np.cos(dtheta)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


def _get_interp_weights(nside, vecs, nest, device, dtype, chunk=1_000_000):
    """
    Torch wrapper for healpy.get_interp_weights.

    Parameters
    ----------
    vecs : torch.Tensor [M, 3]

    Returns
    -------
    idx_t : LongTensor [4, M]
    w_t   : Tensor     [4, M]
    """
    M  = vecs.shape[0]
    vn = vecs / vecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    theta = torch.acos(vn[:, 2].clamp(-1., 1.))
    phi   = torch.atan2(vn[:, 1], vn[:, 0]) % (2.0 * math.pi)
    th_np = theta.detach().cpu().numpy()
    ph_np = phi.detach().cpu().numpy()

    idx_acc, w_acc = [], []
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        i_np, w_np = hp.get_interp_weights(nside, th_np[s:e], ph_np[s:e], nest=nest)
        idx_acc.append(i_np);  w_acc.append(w_np)

    idx_np = np.concatenate(idx_acc, axis=1) if len(idx_acc) > 1 else idx_acc[0]
    w_np   = np.concatenate(w_acc,   axis=1) if len(w_acc)   > 1 else w_acc[0]
    return (
        torch.as_tensor(idx_np, device=device, dtype=torch.long),
        torch.as_tensor(w_np,   device=device, dtype=dtype),
    )


def _bind_support(idx_t, w_t, ids_sorted, kernel_sz, K, P, device):
    """
    Map absolute pixel ids in idx_t [4, K*P] to column positions in
    ids_sorted.  Out-of-domain neighbors are zeroed; weights renormalized.
    Empty stencil points fall back to the center kernel position.

    Returns pos_safe [4, K*P] and w_norm [4, K*P].
    """
    M   = K * P
    pos = torch.searchsorted(ids_sorted, idx_t.reshape(-1)).view(4, M)
    in_range  = pos < ids_sorted.numel()
    cmp_vals  = torch.full_like(idx_t, -1)
    cmp_vals[in_range] = ids_sorted[pos[in_range]]
    present   = (cmp_vals == idx_t)   # [4, K*P]

    # Center stencil point index within P positions
    p_ref = (kernel_sz // 2) * (kernel_sz + 1)

    empty = ~present.any(dim=0)
    if empty.any():
        k_id     = torch.div(torch.arange(M, device=device), P, rounding_mode="floor")
        ref_cols = (k_id * P + p_ref)[empty]
        idx_t[:, empty] = idx_t[:, ref_cols]
        w_t[:,   empty] = w_t[:,   ref_cols]

        idx_e  = idx_t[:, empty].reshape(-1)
        pos_e  = torch.searchsorted(ids_sorted, idx_e)
        valid_e = pos_e < ids_sorted.numel()
        pos_e_c = pos_e.clamp(0, max(ids_sorted.numel() - 1, 0))
        pres_e  = valid_e & (ids_sorted[pos_e_c] == idx_e)
        present[:, empty] = pres_e.view(4, -1)
        pos[:,    empty]  = pos_e_c.view(4, -1)

    w = w_t * present
    colsum = w.sum(dim=0, keepdim=True)
    zero_c = (colsum == 0)
    if zero_c.any():
        w[0, zero_c[0]] = present[0, zero_c[0]].to(w.dtype)
        colsum = w.sum(dim=0, keepdim=True)
    w_norm   = w / colsum.clamp_min(1e-12)
    pos_safe = torch.where(present, pos, torch.zeros_like(pos))
    return pos_safe, w_norm


# ===========================================================================
# HealPixConv
# ===========================================================================

class HealPixConv(nn.Module):
    """
    Gauge-equivariant spherical convolution on HEALPix maps.

    Parameters
    ----------
    nside : int
        HEALPix resolution (power of 2).
    in_channels : int
        Number of input channels C_in.
    out_channels : int
        Number of output channels per gauge C_out.
        Total output channels = n_gauges * out_channels.
    kernel_sz : int, default 3
        Odd integer >= 1.  P = kernel_sz^2 stencil points.
    n_gauges : int, default 1
        Number of gauge orientations G (same kernel, G rotations).
    gauge_type : {"phi", "cosmo"}, default "phi"
        Gauge convention.
    cell_ids : array-like or None
        Pixel indices (NESTED) for partial-sky.  None = full sphere.
    level : int or None
        nside = 2**level.  Required when cell_ids is provided.
    nest : bool, default True
        NESTED pixel ordering.
    use_norm : bool, default False
        Apply GroupNorm + ReLU after convolution.
    device, dtype : Torch device / dtype.
    """

    def __init__(
        self,
        nside: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        cell_ids=None,
        level=None,
        nest: bool = True,
        use_norm: bool = False,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype  = dtype

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
        if gauge_type not in ("phi", "cosmo"):
            raise ValueError("gauge_type must be 'phi' or 'cosmo'.")
        self.gauge_type = gauge_type

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

        # ---- Stage A: rotated stencil + interpolation neighbors ----
        th, ph = hp.pix2ang(self.nside, ids_np.tolist(), nest=self.nest)

        R_tot = _build_rotation_matrices(
            th, ph, self.G, self.gauge_type, self.device, self.dtype
        )  # [K, G, 3, 3]

        vec_t = torch.as_tensor(
            _local_kernel_grid(self.kernel_sz, self.nside),
            device=self.device, dtype=self.dtype,
        )  # [P, 3]

        # Rotate stencil: [K, G, P, 3]
        rotated = torch.einsum("kgij,pj->kgpi", R_tot, vec_t)
        flat    = rotated.reshape(-1, 3)   # [K*G*P, 3]

        idx_flat, w_flat = _get_interp_weights(
            self.nside, flat, self.nest, self.device, self.dtype
        )  # [4, K*G*P]

        # Reshape to [G, 4, K*P]
        idx_all = (idx_flat.view(4, self.K, self.G, self.P)
                            .permute(2, 0, 1, 3)
                            .reshape(self.G, 4, self.K * self.P))
        w_all   = (w_flat.view(4, self.K, self.G, self.P)
                          .permute(2, 0, 1, 3)
                          .reshape(self.G, 4, self.K * self.P))

        # ---- Stage B: binding ----
        ids_sorted   = np.sort(ids_np)
        ids_sorted_t = torch.as_tensor(ids_sorted, device=self.device, dtype=torch.long)

        sort_order = np.argsort(ids_np)
        inv_order  = np.empty_like(sort_order)
        inv_order[sort_order] = np.arange(len(sort_order))
        self.register_buffer("_sort_order",
                             torch.as_tensor(sort_order, dtype=torch.long, device=self.device))
        self.register_buffer("_inv_order",
                             torch.as_tensor(inv_order,  dtype=torch.long, device=self.device))

        pos_list, w_list = [], []
        for g in range(self.G):
            ps, wn = _bind_support(
                idx_all[g].clone(), w_all[g].clone(),
                ids_sorted_t, self.kernel_sz, self.K, self.P, self.device,
            )
            pos_list.append(ps);  w_list.append(wn)

        self.register_buffer("_pos_safe", torch.stack(pos_list, dim=0))  # [G, 4, K*P]
        self.register_buffer("_w_norm",   torch.stack(w_list,   dim=0))  # [G, 4, K*P]

        # ---- learnable kernel and bias ----
        self.weight = nn.Parameter(
            torch.empty(self.G, self.in_channels, self.out_channels, self.P,
                        device=self.device, dtype=self.dtype)
        )
        nn.init.kaiming_uniform_(
            self.weight.view(self.G * self.in_channels, self.out_channels * self.P),
            a=0., mode="fan_in", nonlinearity="relu",
        )
        self.bias = nn.Parameter(
            torch.zeros(self.G * self.out_channels, device=self.device, dtype=self.dtype)
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
    # Kernel management
    # ------------------------------------------------------------------

    def set_kernel(self, W, bias=None, requires_grad=False):
        """
        Replace the learnable kernel with a fixed (or re-initialised) array.

        Parameters
        ----------
        W : array-like
            Shape ``[C_in, C_out, P]``  -- same kernel broadcast over all G gauges.
            Shape ``[G, C_in, C_out, P]``  -- per-gauge kernels.
        bias : array-like or None
            Shape ``[G * C_out]``.  If None, bias is reset to zero.
        requires_grad : bool, default False
            Set to True to fine-tune from this initialisation.

        Returns
        -------
        self  (for chaining)

        Examples
        --------
        Isotropic Gaussian smoothing (kernel_sz=3, K=9)::

            W = np.zeros((1, 1, 9), dtype=np.float32)
            # stencil point 4 is the center; 0..3 and 5..8 are the neighbours
            W[0, 0, 4] = 0.5        # center weight
            W[0, 0, [0,1,2,3,5,6,7,8]] = 0.5 / 8.0   # ring weight
            conv.set_kernel(W)
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
            self.weight.copy_(torch.as_tensor(W_np, dtype=self.dtype, device=self.device))
            if bias is not None:
                self.bias.copy_(
                    torch.as_tensor(np.asarray(bias, np.float32).ravel(),
                                    dtype=self.dtype, device=self.device)
                )
            else:
                self.bias.zero_()
        self.weight.requires_grad_(requires_grad)
        self.bias.requires_grad_(requires_grad)
        return self

    # ------------------------------------------------------------------
    # Forward
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
        """
        t, is_numpy, was_1d, _ = _prepare_input_conv(x, self.device, self.dtype)
        B, C_in, N = t.shape

        if C_in != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C_in}.")
        if N != self.K:
            raise ValueError(f"Expected {self.K} pixels, got {N}.")

        so  = self._sort_order.to(device=t.device)
        io  = self._inv_order.to(device=t.device)
        pos = self._pos_safe.to(device=t.device)
        wn  = self._w_norm.to(device=t.device, dtype=t.dtype)
        W   = self.weight.to(device=t.device, dtype=t.dtype)

        t_sorted = t[:, :, so]   # [B, C_in, K] aligned with ids_sorted

        outs = []
        for g in range(self.G):
            ps = pos[g]   # [4, K*P]
            wg = wn[g]    # [4, K*P]
            Wg = W[g]     # [C_in, C_out, P]

            # Bilinear interpolation over 4 neighbours -> [B, C_in, K, P]
            gathered = sum(
                t_sorted.index_select(2, ps[j].reshape(-1))
                        .view(B, C_in, self.K, self.P)
                * wg[j].view(1, 1, self.K, self.P)
                for j in range(4)
            )

            # Channel + spatial mixing: [B, C_out, K]
            yg = torch.einsum("bckp,cop->bok", gathered, Wg)
            outs.append(yg)

        y = torch.cat(outs, dim=1)                        # [B, G*C_out, K]
        y = y + self.bias.to(device=t.device, dtype=t.dtype).view(1, -1, 1)
        y = y[:, :, io]                                    # unsort to original order

        if self.use_norm and self.norm is not None:
            nm = self.norm.to(device=t.device, dtype=t.dtype)
            y  = F.relu(nm(y), inplace=True)

        return _restore_output_conv(y, is_numpy, was_1d)

    def extra_repr(self):
        return (
            f"nside={self.nside}, in={self.in_channels}, out={self.out_channels}, "
            f"kernel_sz={self.kernel_sz}, P={self.P}, G={self.G}, "
            f"gauge={self.gauge_type!r}, partial={self.partial}"
        )
