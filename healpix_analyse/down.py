"""
down.py
=======
HEALPix resolution-reduction operator for the ``healpix_analyse`` package.

Reduces the HEALPix resolution by a factor of 2:
    nside_out = nside_in // 2
    N_out     = N_in // 4

Two modes:
    "smooth"  -- weighted Gaussian average over a disc (linear, differentiable).
    "maxpool" -- non-linear max over the 4 direct NESTED children.

Works on the full sphere (nside inferred from input size) or on a partial-sky
subset (``cell_ids`` + ``level`` parameters).

Accepts numpy arrays or torch tensors of shape ``[N]`` or ``[B, N]`` and
returns an output of the same type and number of dimensions.

Dependencies: numpy, torch, healpy.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import healpix_geo

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------

def _prepare_input(
    x: ArrayLike,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, bool, bool]:
    """
    Convert an input array / tensor into a 2-D torch.Tensor [B, N].

    Returns
    -------
    t       : torch.Tensor  [B, N]
    is_numpy: bool   -- True if the input was a numpy array
    was_1d  : bool   -- True if the input had shape [N] (no batch dim)
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        t = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        t = x.to(device=device, dtype=dtype)

    was_1d = (t.ndim == 1)
    if was_1d:
        t = t.unsqueeze(0)  # [1, N]

    if t.ndim != 2:
        raise ValueError(
            f"Input must have shape [N] or [B, N], got {tuple(t.shape)}"
        )
    return t, is_numpy, was_1d


def _restore_output(
    t: torch.Tensor,
    is_numpy: bool,
    was_1d: bool,
) -> ArrayLike:
    """
    Convert a 2-D torch.Tensor [B, N] back to the original format.
    """
    if was_1d:
        t = t.squeeze(0)   # [N]
    if is_numpy:
        return t.detach().cpu().numpy()
    return t


# ---------------------------------------------------------------------------
# HealPixDown
# ---------------------------------------------------------------------------

class HealPixDown(nn.Module):
    """
    HEALPix downsampling: reduce resolution by a factor of 2.

    Reduces ``nside_in`` → ``nside_out = nside_in // 2`` using either a
    Gaussian-weighted sparse matrix (``mode="smooth"``) or a max-pooling
    over the 4 direct NESTED children (``mode="maxpool"``).

    Can operate on the full sphere or on a partial-sky subset.

    Parameters
    ----------
    nside_in : int
        Input HEALPix resolution.  Must be a power of 2 and ≥ 2.
        When ``cell_ids`` is ``None``, the input is expected to have
        ``N = 12 * nside_in**2`` pixels.
        When ``cell_ids`` is provided, ``nside_in`` must equal ``2**level``.
    mode : {"smooth", "maxpool"}, default "smooth"
        Downsampling strategy.
        *smooth*   -- differentiable Gaussian-weighted average (linear).
        *maxpool*  -- non-linear max over 4 NESTED children (fast).
    radius_deg : float or None
        Angular radius of the Gaussian kernel (degrees).
        Defaults to ~3 times the fine pixel size.
        Only used for ``mode="smooth"``.
    sigma_deg : float or None
        Gaussian sigma (degrees).  Defaults to ``radius_deg / 2``.
        Only used for ``mode="smooth"``.
    weight_norm : {"l1", "l2", "none"}, default "l1"
        Normalisation applied per output pixel in ``mode="smooth"``:
        *l1*   -- sum of weights = 1  (preserves constant fields).
        *l2*   -- sum of squares = 1  (preserves local energy).
        *none* -- raw Gaussian weights (not recommended).
    cell_ids : array-like of int or None
        Pixel indices (NESTED ordering) of the input sub-map.
        If ``None``, the operator covers the full sphere.
        If provided, ``level`` is also required and ``nside_in = 2**level``.
    level : int or None
        HEALPix level such that ``nside_in = 2**level``.
        Required when ``cell_ids`` is not ``None``.
    device : torch.device or str or None
        Device for the sparse matrix and computations.
        Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for the sparse matrix values.

    Examples
    --------
    Full sphere:

    >>> import numpy as np
    >>> from healpix_analyse.down import HealPixDown
    >>> nside = 64
    >>> x = np.random.randn(12 * nside**2)          # [N]
    >>> down = HealPixDown(nside_in=nside)
    >>> y, cell_ids_out = down(x)
    >>> y.shape
    (49152,)   # 12 * 32**2

    Partial sky:

    >>> import healpy as hp
    >>> cell_ids = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0), 0.3, nest=True)
    >>> down = HealPixDown(nside_in=nside, cell_ids=cell_ids, level=6)
    >>> y, coarse_ids = down(x[cell_ids])
    """

    def __init__(
        self,
        nside_in: int,
        mode: str = "smooth",
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        cell_ids: Optional[ArrayLike] = None,
        level: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # ---- resolve device ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.ellipsoid = ellipsoid

        # ---- validate nside ----
        self.nside_in = int(nside_in)
        if self.nside_in < 2 or (self.nside_in & (self.nside_in - 1)) != 0:
            raise ValueError("nside_in must be a power of 2 and >= 2.")
        self.nside_out = self.nside_in // 2
        self.N_in_full  = 12 * self.nside_in  * self.nside_in
        self.N_out_full = 12 * self.nside_out * self.nside_out

        # ---- validate mode / weight_norm ----
        self.mode = mode.lower().strip()
        if self.mode not in ("smooth", "maxpool"):
            raise ValueError("mode must be 'smooth' or 'maxpool'.")
        self.weight_norm = weight_norm.lower().strip()
        if self.weight_norm not in ("l1", "l2", "none"):
            raise ValueError("weight_norm must be 'l1', 'l2', or 'none'.")

        # ---- partial sky ----
        self.partial = cell_ids is not None
        if self.partial:
            if level is None:
                raise ValueError(
                    "level must be provided together with cell_ids "
                    "(nside_in = 2**level)."
                )
            expected_nside = 2 ** int(level)
            if expected_nside != self.nside_in:
                raise ValueError(
                    f"Inconsistent level={level} (→ nside={expected_nside}) "
                    f"and nside_in={self.nside_in}."
                )
            cell_ids_in = np.asarray(cell_ids, dtype=np.int64).ravel()
            # Coarse output pixels: parent of each input pixel in NESTED ordering
            cell_ids_out = np.unique(cell_ids_in // 4).astype(np.int64)
            self._cell_ids_in  = cell_ids_in
            self._cell_ids_out = cell_ids_out
            self.N_in  = len(cell_ids_in)
            self.N_out = len(cell_ids_out)
        else:
            self._cell_ids_in  = None
            self._cell_ids_out = np.arange(self.N_out_full, dtype=np.int64)
            self.N_in  = self.N_in_full
            self.N_out = self.N_out_full

        # ---- Gaussian scale (smooth mode) ----
        if self.mode == "smooth":
            pix_area = 4.0 * np.pi / self.N_in_full
            pix_deg  = np.degrees(np.sqrt(pix_area))
            self.radius_deg = float(radius_deg) if radius_deg is not None else 3.0 * pix_deg
            self.sigma_deg  = float(sigma_deg)  if sigma_deg  is not None else self.radius_deg / 2.0
            self.radius_rad = np.radians(self.radius_deg)
            self.sigma_rad  = np.radians(self.sigma_deg)

            M = self._build_smooth_matrix()
            self.register_buffer("_M_indices", M.indices().clone())
            self.register_buffer("_M_values",  M.values().clone())
            self._M_size = M.size()
        else:
            # maxpool: precompute children indices
            self._build_maxpool_children()

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
        """Haversine angular distance (radians) between (lat1,lon1) and (lat2,lon2)."""
        dlat = 0.5 * (lat2 - lat1)
        dlon = 0.5 * (lon2 - lon1)
        a = np.clip(
            np.sin(dlat)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon)**2,
            0.0, 1.0
        )
        return 2.0 * np.arcsin(np.sqrt(a))

    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        if self.weight_norm == "l1":
            s = w.sum()
            return w / s if s > 0 else np.ones_like(w) / max(len(w), 1)
        if self.weight_norm == "l2":
            s2 = (w * w).sum()
            return w / np.sqrt(s2) if s2 > 0 else np.ones_like(w) / max(np.sqrt(len(w)), 1.0)
        return w  # "none"

    def _build_smooth_matrix(self) -> torch.Tensor:
        """
        Build sparse matrix M of shape (N_out, N_in):
            y[i_out] = sum_j M[i_out, j_in] * x[j_in]
        with Gaussian weights based on haversine distance.
        """
        cell_ids_out = self._cell_ids_out   # coarse pixel ids [N_out]
        cell_ids_in  = self._cell_ids_in    # fine pixel ids [N_in] (None = full sphere)

        # For fast column lookup when partial sky: sorted ids + searchsorted
        if self.partial:
            ids_in_sorted = np.sort(cell_ids_in)
            # Map fine pixel id → column index in the sparse matrix
            def _col(pix_arr):
                idx = np.searchsorted(ids_in_sorted, pix_arr)
                idx = np.clip(idx, 0, len(ids_in_sorted) - 1)
                mask = ids_in_sorted[idx] == pix_arr
                return idx, mask
        else:
            def _col(pix_arr):
                return pix_arr.astype(np.int64), np.ones(len(pix_arr), dtype=bool)

        rows, cols, vals = [], [], []

        for i_out, p_out in enumerate(cell_ids_out):
            #theta0, phi0 = hp.pix2ang(self.nside_out, int(p_out), nest=True)
            theta0, phi0 = healpix_geo.nested.healpix_to_lonlat(int(p_out), int(np.log2(self.nside_out)),ellipsoid=self.ellipsoid)
            lat0 = 0.5 * np.pi - theta0

            # Query fine pixels within the disc
            vec0 = hp.ang2vec(theta0, phi0)
            cand = np.asarray(
                hp.query_disc(self.nside_in, vec0, self.radius_rad,
                              inclusive=True, nest=True),
                dtype=np.int64
            )

            # Fallback: use 4 NESTED children if disc is empty
            if cand.size == 0:
                cand = (4 * p_out + np.arange(4)).astype(np.int64)

            # Filter to available input pixels when partial sky
            col_idx, in_mask = _col(cand)
            cand    = cand[in_mask]
            col_idx = col_idx[in_mask]

            if len(cand) == 0:
                # No available fine pixels: skip this coarse pixel
                warnings.warn(
                    f"Coarse pixel {p_out} has no available input pixels in "
                    "cell_ids; it will produce zero output.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

            # Gaussian weights
            theta_c, phi_c = hp.pix2ang(self.nside_in, cand.tolist(), nest=True)
            lat_c = 0.5 * np.pi - theta_c
            gamma = self._haversine(lat0, phi0, lat_c, phi_c)
            w = np.exp(-0.5 * (gamma / self.sigma_rad) ** 2)
            w[gamma > self.radius_rad] = 0.0
            if w.sum() <= 0.0:
                w[:] = 1.0

            w = self._normalize_weights(w)

            for j, (col, wv) in enumerate(zip(col_idx, w)):
                if wv != 0.0:
                    rows.append(i_out)
                    cols.append(int(col))
                    vals.append(float(wv))

        rows_t = torch.tensor(rows, dtype=torch.long,   device=self.device)
        cols_t = torch.tensor(cols, dtype=torch.long,   device=self.device)
        vals_t = torch.tensor(vals, dtype=self.dtype,   device=self.device)

        M = torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t]),
            vals_t,
            size=(self.N_out, self.N_in),
            device=self.device,
            dtype=self.dtype,
        ).coalesce()
        return M

    def _build_maxpool_children(self) -> None:
        """
        Precompute the child index mapping for the maxpool mode.
        Registers buffer ``_children_idx`` of shape [N_out, 4].
        """
        cell_ids_out = self._cell_ids_out  # coarse pixel ids

        if self.partial:
            # For each coarse pixel, collect available children from cell_ids_in
            ids_in_set = set(self._cell_ids_in.tolist())
            children_list = []
            fallback = int(self._cell_ids_in[0])  # safe fallback index (column 0)

            for p_out in cell_ids_out:
                ch = [int(4 * p_out + k) for k in range(4)]
                # Replace missing children with fallback column 0
                ch_avail = [c if c in ids_in_set else fallback for c in ch]
                children_list.append(ch_avail)

            # Map absolute pixel ids → column indices in cell_ids_in
            ids_in_sorted = np.sort(self._cell_ids_in)
            children_np   = np.array(children_list, dtype=np.int64)  # [N_out, 4]
            flat           = children_np.ravel()
            col_idx        = np.searchsorted(ids_in_sorted, flat).reshape(children_np.shape)
            children_idx   = col_idx
        else:
            children_idx = np.stack([
                4 * cell_ids_out,
                4 * cell_ids_out + 1,
                4 * cell_ids_out + 2,
                4 * cell_ids_out + 3,
            ], axis=1).astype(np.int64)

        self.register_buffer(
            "_children_idx",
            torch.as_tensor(children_idx, dtype=torch.long, device=self.device),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Output cell ids (coarse-resolution NESTED indices)."""
        return self._cell_ids_out

    def forward(
        self, x: ArrayLike
    ) -> Tuple[ArrayLike, np.ndarray]:
        """
        Apply the downsampling operator.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor, shape [N] or [B, N]
            Input map(s) at fine resolution ``nside_in``.
            When ``cell_ids`` was given at construction, ``N`` must equal
            ``len(cell_ids)``.  Otherwise ``N = 12 * nside_in**2``.

        Returns
        -------
        y : same type as x, shape [N_out] or [B, N_out]
            Downsampled map(s) at coarse resolution ``nside_out``.
        cell_ids_out : np.ndarray
            NESTED pixel indices of the output pixels at ``nside_out``.
        """
        t, is_numpy, was_1d = _prepare_input(x, self.device, self.dtype)
        B, N = t.shape

        if N != self.N_in:
            raise ValueError(
                f"Expected input with {self.N_in} pixels, got {N}."
            )

        if self.mode == "smooth":
            M = torch.sparse_coo_tensor(
                self._M_indices.to(device=t.device),
                self._M_values.to(device=t.device, dtype=t.dtype),
                size=self._M_size,
                device=t.device,
                dtype=t.dtype,
            )
            # M: [N_out, N_in];  t: [B, N_in]
            # y_T = M @ t.T  →  [N_out, B]
            y = torch.sparse.mm(M, t.T).T   # [B, N_out]

        else:  # maxpool
            idx = self._children_idx.to(device=t.device)   # [N_out, 4]
            gathered = t[:, idx]                            # [B, N_out, 4]
            y = gathered.max(dim=-1).values                 # [B, N_out]

        return _restore_output(y, is_numpy, was_1d), self._cell_ids_out
