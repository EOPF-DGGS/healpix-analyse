"""
up.py
=====
HEALPix resolution-increase operator for the ``healpix_analyse`` package.

Increases the HEALPix resolution by a factor of 2:
    nside_out = nside_in * 2
    N_out     = N_in * 4

The operation is defined as the *adjoint* (transpose) of the smooth
downsampling matrix from :mod:`healpix_analyse.down`, with an optional
diagonal normalisation that improves round-trip consistency.

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

try:
    import healpy as hp
except ImportError as e:
    raise ImportError(
        "healpy is required by healpix_analyse.up. "
        "Install it with:  pip install healpy"
    ) from e

from healpix_analyse.down import HealPixDown, _prepare_input, _restore_output

ArrayLike = Union[np.ndarray, torch.Tensor]


class HealPixUp(nn.Module):
    """
    HEALPix upsampling: increase resolution by a factor of 2.

    Implements the *adjoint* of the smooth downsampling operator from
    :class:`healpix_analyse.down.HealPixDown`:

        x_up = M^T @ x

    where ``M`` is the (N_coarse × N_fine) downsampling matrix.

    Optional diagonal normalisation modes improve amplitude preservation
    for constant fields or local energy:

    - ``"adjoint"``  -- raw transpose (M^T x).
    - ``"col_l1"``   -- divide by column sums of M: preserves constant fields
                        when ``HealPixDown`` used ``weight_norm="l1"``.
    - ``"diag_l2"``  -- divide by diagonal of M^T M: least-squares diagonal
                        preconditioner (local amplitude correction).

    Parameters
    ----------
    nside_in : int
        Coarse input HEALPix resolution (must be a power of 2, ≥ 1).
        Output resolution will be ``nside_in * 2``.
    radius_deg : float or None
        Angular radius passed to :class:`~healpix_analyse.down.HealPixDown`
        for building the transpose matrix.  ``None`` → default.
    sigma_deg : float or None
        Gaussian sigma (degrees).  ``None`` → default.
    weight_norm : {"l1", "l2", "none"}, default "l1"
        Weight normalisation used when building the downsampling matrix.
        Must match the norm used by the paired :class:`HealPixDown`.
    up_norm : {"adjoint", "col_l1", "diag_l2"}, default "col_l1"
        Diagonal normalisation applied after the transpose:
        *adjoint* -- no normalisation.
        *col_l1*  -- preserve constant fields.
        *diag_l2* -- preserve local energy.
    eps : float, default 1e-12
        Small value added to the denominator to avoid division by zero.
    cell_ids : array-like of int or None
        Coarse pixel indices (NESTED) for partial-sky operation.
        If ``None``, the operator covers the full sphere.
        If provided, ``level`` is also required and
        ``nside_in = 2**level``.
    level : int or None
        HEALPix level such that ``nside_in = 2**level``.
        Required when ``cell_ids`` is not ``None``.
    device : torch.device or str or None
        Device for computation.  Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for the sparse matrix values.

    Examples
    --------
    Full sphere:

    >>> import numpy as np
    >>> from healpix_analyse.up import HealPixUp
    >>> nside = 32
    >>> x = np.random.randn(12 * nside**2)      # [N_coarse]
    >>> up = HealPixUp(nside_in=nside)
    >>> y, cell_ids_out = up(x)
    >>> y.shape
    (49152,)   # 12 * 64**2

    Partial sky:

    >>> import healpy as hp
    >>> cell_ids_coarse = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0), 0.3, nest=True)
    >>> up = HealPixUp(nside_in=nside, cell_ids=cell_ids_coarse, level=5)
    >>> y, fine_ids = up(x[cell_ids_coarse])
    """

    def __init__(
        self,
        nside_in: int,
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        eps: float = 1e-12,
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

        # ---- validate nside ----
        self.nside_in  = int(nside_in)
        if self.nside_in < 1 or (self.nside_in > 1 and (self.nside_in & (self.nside_in - 1)) != 0):
            raise ValueError("nside_in must be a power of 2 and >= 1.")
        self.nside_out = self.nside_in * 2   # fine resolution
        self.N_in_full  = 12 * self.nside_in  * self.nside_in
        self.N_out_full = 12 * self.nside_out * self.nside_out

        # ---- validate up_norm ----
        self.up_norm = up_norm.lower().strip()
        if self.up_norm not in ("adjoint", "col_l1", "diag_l2"):
            raise ValueError("up_norm must be 'adjoint', 'col_l1', or 'diag_l2'.")
        self.eps = float(eps)

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
            # Fine output pixels: 4 NESTED children of each coarse input pixel
            cell_ids_out = np.unique(
                (cell_ids_in[:, None] * 4 + np.arange(4)[None, :]).ravel()
            ).astype(np.int64)
            self._cell_ids_in  = cell_ids_in
            self._cell_ids_out = cell_ids_out
            self.N_in  = len(cell_ids_in)
            self.N_out = len(cell_ids_out)
        else:
            self._cell_ids_in  = None
            self._cell_ids_out = np.arange(self.N_out_full, dtype=np.int64)
            self.N_in  = self.N_in_full
            self.N_out = self.N_out_full

        # ---- build M^T via HealPixDown at fine resolution ----
        # The paired Down operator goes from nside_out (fine) → nside_in (coarse).
        # We build it and immediately transpose it.
        down = HealPixDown(
            nside_in    = self.nside_out,          # fine nside
            mode        = "smooth",
            radius_deg  = radius_deg,
            sigma_deg   = sigma_deg,
            weight_norm = weight_norm,
            cell_ids    = cell_ids_out if self.partial else None,
            level       = (level + 1) if (self.partial and level is not None) else None,
            device      = self.device,
            dtype       = self.dtype,
        )

        # Reconstruct Down matrix M_down: [N_in_coarse, N_out_fine]
        M_down = torch.sparse_coo_tensor(
            down._M_indices,
            down._M_values,
            size=down._M_size,
            device=self.device,
            dtype=self.dtype,
        ).coalesce()

        # Transpose → M_up: [N_out_fine, N_in_coarse]
        M_up = self._transpose_sparse(M_down)
        self.register_buffer("_M_indices", M_up.indices().clone())
        self.register_buffer("_M_values",  M_up.values().clone())
        self._M_size = M_up.size()

        # Diagonal normalisation vectors (computed from M_down once)
        if self.up_norm in ("col_l1", "diag_l2"):
            idx  = M_down.indices()
            vals = M_down.values()
            cols = idx[1]   # fine-grid column indices

            col_sum = torch.zeros(self.N_out, device=self.device, dtype=self.dtype)
            col_l2  = torch.zeros(self.N_out, device=self.device, dtype=self.dtype)
            col_sum.scatter_add_(0, cols, vals)
            col_l2.scatter_add_(0, cols, vals * vals)

            self.register_buffer("_col_sum", col_sum)
            self.register_buffer("_col_l2",  col_l2)
        else:
            self._col_sum = None
            self._col_l2  = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transpose_sparse(M: torch.Tensor) -> torch.Tensor:
        """Return the transpose of a coalesced sparse COO matrix."""
        M = M.coalesce()
        idx = M.indices()
        R, C = M.size()
        idx_T = torch.stack([idx[1], idx[0]], dim=0)
        return torch.sparse_coo_tensor(
            idx_T, M.values(), size=(C, R),
            device=M.device, dtype=M.dtype
        ).coalesce()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Output cell ids (fine-resolution NESTED indices)."""
        return self._cell_ids_out

    def forward(
        self, x: ArrayLike
    ) -> Tuple[ArrayLike, np.ndarray]:
        """
        Apply the upsampling operator.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor, shape [N] or [B, N]
            Input map(s) at coarse resolution ``nside_in``.
            When ``cell_ids`` was given at construction, ``N`` must equal
            ``len(cell_ids)``.  Otherwise ``N = 12 * nside_in**2``.

        Returns
        -------
        y : same type as x, shape [N_out] or [B, N_out]
            Upsampled map(s) at fine resolution ``nside_out = nside_in * 2``.
        cell_ids_out : np.ndarray
            NESTED pixel indices of the output pixels at ``nside_out``.
        """
        t, is_numpy, was_1d = _prepare_input(x, self.device, self.dtype)
        B, N = t.shape

        if N != self.N_in:
            raise ValueError(
                f"Expected input with {self.N_in} pixels, got {N}."
            )

        M_T = torch.sparse_coo_tensor(
            self._M_indices.to(device=t.device),
            self._M_values.to(device=t.device, dtype=t.dtype),
            size=self._M_size,
            device=t.device,
            dtype=t.dtype,
        )

        # M_T: [N_out, N_in];  t: [B, N_in]
        y = torch.sparse.mm(M_T, t.T).T   # [B, N_out]

        # Optional diagonal normalisation
        if self.up_norm == "col_l1" and self._col_sum is not None:
            denom = self._col_sum.to(device=t.device, dtype=t.dtype).clamp_min(self.eps)
            y = y / denom.unsqueeze(0)

        elif self.up_norm == "diag_l2" and self._col_l2 is not None:
            denom = self._col_l2.to(device=t.device, dtype=t.dtype).clamp_min(self.eps)
            y = y / denom.unsqueeze(0)

        # up_norm == "adjoint" → no correction

        return _restore_output(y, is_numpy, was_1d), self._cell_ids_out
