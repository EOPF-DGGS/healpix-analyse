"""
alm_latlon.py
=============
Spherical harmonic transform (map → alm → Cl) for maps defined on arbitrary
iso-latitude ring grids (ERA5/ECMWF Gaussian grids, regular lat/lon grids,
HEALPix pixel coordinates, or any custom ring-structured grid).

This module is entirely self-contained: it depends only on ``numpy`` and
``torch``.  It does **not** depend on the ``foscat`` package.

Public API
----------
build_rings_from_latlon(lat, lon, atol, convention)
    Sort a flat list of pixels into iso-latitude rings.

compute_weights(ring_theta, ring_phi_list, ring_counts, quadrature)
    Compute per-pixel quadrature weights (steradians).

map2alm_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights, quadrature)
    Map → spherical harmonic coefficients a_lm.

alm2map_latlon(alm, ring_theta, ring_phi_list, ring_counts, lmax)
    Spherical harmonic coefficients → reconstructed map (synthesis).

anafast_latlon(im, ring_theta, ring_phi_list, ring_counts, lmax, weights, quadrature)
    Map → angular power spectrum Cl.

grid_summary(ring_theta, ring_phi_list, ring_counts, lmax)
    Print a diagnostic summary of the grid geometry.

Mathematical convention
-----------------------
The transform follows the standard orthonormal convention (same as healpy):

    a_lm = ∫ f(θ,φ) Y_lm*(θ,φ) dΩ

    Y_lm(θ,φ) = sqrt((2l+1)/(4π)) * P̃_lm(cosθ) * exp(imφ)

    C_l = 1/(2l+1) * [|a_l0|² + 2 Σ_{m=1}^{l} |a_lm|²]

where P̃_lm are the normalised associated Legendre polynomials.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, torch.Tensor, List[float]]


# ===========================================================================
# Internal: normalised associated Legendre polynomials
# ===========================================================================

def _double_factorial_log(n: int) -> float:
    """
    Return log(n!!) where n!! is the double factorial.
    Returns 0 for n <= 0 by convention.
    """
    if n <= 0:
        return 0.0
    result = 0.0
    i = n
    while i > 0:
        result += math.log(i)
        i -= 2
    return result


def _compute_legendre_m(
    x: np.ndarray,
    m: int,
    lmax: int,
    limit_range: float = 1e10,
) -> np.ndarray:
    """
    Compute the normalised associated Legendre polynomials P̃_lm(x) for
    l = m, m+1, ..., lmax, evaluated at each point in x.

    The output is scaled so that:

        result[l-m, r] = sqrt(4π) * P̃_lm(x[r])

    i.e. the Y_lm normalisation factor sqrt((2l+1)/4π) is NOT yet applied
    here.  It is applied in map2alm_latlon.

    Parameters
    ----------
    x : np.ndarray, shape [R]
        Values of cos(θ) at which to evaluate the polynomials.
    m : int
        Azimuthal order.
    lmax : int
        Maximum degree.
    limit_range : float
        Guard value to prevent numerical overflow during the recurrence.

    Returns
    -------
    result : np.ndarray, shape [lmax-m+1, R], complex128
        Normalised Legendre polynomials scaled by sqrt(4π).
    """
    n_l = lmax - m + 1
    R   = x.shape[0]
    result = np.zeros((n_l, R), dtype=np.float64)
    ratio  = np.zeros((n_l, 1), dtype=np.float64)
    log_limit = math.log(limit_range)
    inv_limit = 1.0 / limit_range

    # --- log-normalisation for P_mm ---
    ratio[0, 0] = _double_factorial_log(2 * m - 1) - 0.5 * np.sum(
        np.log(1.0 + np.arange(2 * m))
    ) if m > 0 else _double_factorial_log(2 * m - 1)

    # --- seed: P_mm(x) ---
    if m == 0:
        Pmm = np.ones(R, dtype=np.float64)
    else:
        # P_mm = (-1)^m * (1 - x^2)^(m/2)  (with sign absorbed into ratio)
        Pmm = (0.5 - m % 2) * 2.0 * (1.0 - x**2) ** (m / 2.0)
    result[0] = Pmm

    if m == lmax:
        return (result * np.exp(ratio) * math.sqrt(4.0 * math.pi) + 0j)

    # --- P_{m+1, m}(x) ---
    result[1]  = x * (2 * m + 1) * result[0]
    ratio[1, 0] = ratio[0, 0] - 0.5 * math.log(2 * m + 1)

    # --- Recurrence for l = m+2 .. lmax ---
    for ell in range(m + 2, lmax + 1):
        result[ell - m] = (
            (2 * ell - 1) * x * result[ell - m - 1]
            - (ell + m - 1) * result[ell - m - 2]
        ) / (ell - m)
        ratio[ell - m, 0] = (
            0.5 * math.log(ell - m)
            - 0.5 * math.log(ell + m)
            + ratio[ell - m - 1, 0]
        )
        # Overflow guard: rescale if values grow too large
        if np.max(np.abs(result[ell - m])) > limit_range:
            result[ell - m - 1] *= inv_limit
            result[ell - m]     *= inv_limit
            ratio[ell - m - 1, 0] += log_limit
            ratio[ell - m, 0]     += log_limit

    return (result * np.exp(ratio) * math.sqrt(4.0 * math.pi) + 0j)


# ===========================================================================
# Internal: FFT helpers
# ===========================================================================

def _rfft2fft(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the full complex DFT of a real 1-D signal from its rfft.

    torch.fft.rfft returns only the non-redundant half of the spectrum
    (frequencies 0 .. N//2).  This function reconstructs the full N-point
    complex spectrum by appending the conjugate-reversed negative frequencies,
    matching the convention used in the original FOSCAT comp_tf.

    Parameters
    ----------
    v : torch.Tensor, shape [..., N]
        Real-valued input signal along the last axis.

    Returns
    -------
    torch.Tensor, shape [..., N], complex
        Full complex spectrum.
    """
    r     = torch.fft.rfft(v)            # [..., N//2 + 1]
    r_neg = r[..., 1:-1].conj().flip(-1) # conjugate of positive freqs (excl. DC & Nyquist)
    return torch.cat([r, r_neg], dim=-1) # [..., N]


def _check_uniform_phi(phi: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Decide whether the longitudes in a ring are uniformly spaced (Δφ = 2π/N).

    Parameters
    ----------
    phi : np.ndarray, shape [N_r]
        Longitudes in radians (not necessarily sorted).
    tol : float
        Relative tolerance on the spacing variation.

    Returns
    -------
    is_uniform : bool
    phi0 : float
        Minimum longitude (starting phase) if uniform, else 0.0.
    """
    N = len(phi)
    if N <= 1:
        return True, float(phi[0]) if N == 1 else 0.0

    sorted_phi = np.sort(phi)
    dphi       = np.diff(sorted_phi)
    mean_dp    = 2.0 * math.pi / N

    if np.ptp(dphi) < tol * mean_dp:
        return True, float(sorted_phi[0])
    return False, 0.0


# ===========================================================================
# Public helper: parse phi_list
# ===========================================================================

def _parse_phi_list(
    ring_phi_list: Union[List[np.ndarray], np.ndarray],
    ring_counts: np.ndarray,
) -> List[np.ndarray]:
    """
    Accept ring_phi_list either as a list of per-ring arrays or as a single
    flat array of length N_total, and return a list of per-ring float64 arrays.
    """
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    if isinstance(ring_phi_list, np.ndarray) and ring_phi_list.ndim == 1:
        splits = np.cumsum(ring_counts)[:-1]
        return [a.astype(np.float64) for a in np.split(ring_phi_list, splits)]
    return [np.asarray(p, dtype=np.float64) for p in ring_phi_list]


# ===========================================================================
# build_rings_from_latlon
# ===========================================================================

def build_rings_from_latlon(
    lat: ArrayLike,
    lon: ArrayLike,
    atol: float = 1e-10,
    convention: str = "colatitude_rad",
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Group a flat list of N pixels into iso-latitude rings.

    This is the mandatory first step before any transform call.  The function
    sorts pixels by colatitude (then longitude) and identifies contiguous groups
    sharing the same colatitude within tolerance ``atol``.

    Parameters
    ----------
    lat : array-like, shape [N]
        Latitude or colatitude of each pixel.  Unit and sign depend on
        ``convention`` (see below).
    lon : array-like, shape [N]
        Longitude of each pixel.  Unit depends on ``convention``.
    atol : float, default 1e-10
        Angular tolerance in radians to merge two pixels into the same ring.
        Increase to ~1e-6 when coordinates come from single-precision data.
    convention : str, default "colatitude_rad"
        Coordinate convention.  Accepted values:

        ``"colatitude_rad"``
            lat = colatitude θ in **radians**,  0 → π
            lon = longitude  φ in **radians**,  0 → 2π
        ``"colatitude_deg"``
            lat = colatitude θ in **degrees**,  0° → 180°
            lon = longitude  φ in **degrees**,  0° → 360°
        ``"geographic_rad"``
            lat = geographic latitude in **radians**,  -π/2 → +π/2
            lon = longitude in **radians**,  -π → +π  or  0 → 2π
        ``"geographic_deg"``
            lat = geographic latitude in **degrees**,  -90° → +90°
            lon = longitude in **degrees**,  -180° → +180°  or  0° → 360°

    Returns
    -------
    ring_theta : np.ndarray, shape [R]
        Colatitude θ in **radians** of each ring, sorted in ascending order.
    ring_phi_list : list of np.ndarray
        Longitudes φ in **radians** for each ring.
    ring_counts : np.ndarray, shape [R], dtype int64
        Number of pixels N_r in each ring.
    sort_idx : np.ndarray, shape [N], dtype int64
        Permutation index.  Apply as ``im_sorted = im[sort_idx]`` before
        passing a map to any transform function.
    """
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()

    # --- coordinate conversion ---
    conv = convention.lower().strip()
    if conv == "colatitude_rad":
        theta = lat
        phi   = lon
    elif conv == "colatitude_deg":
        theta = np.radians(lat)
        phi   = np.radians(lon)
    elif conv == "geographic_rad":
        theta = np.pi / 2.0 - lat
        phi   = lon % (2.0 * np.pi)
    elif conv == "geographic_deg":
        theta = np.radians(90.0 - lat)
        phi   = np.radians(lon) % (2.0 * np.pi)
    else:
        raise ValueError(
            f"Unknown convention '{convention}'.  "
            "Accepted values: 'colatitude_rad', 'colatitude_deg', "
            "'geographic_rad', 'geographic_deg'."
        )

    N = len(theta)

    # --- sort by colatitude then longitude ---
    order = np.lexsort((phi, theta))  # primary key: theta
    theta_s = theta[order]
    phi_s   = phi[order]

    # --- detect ring boundaries (jump in colatitude > atol) ---
    breaks      = np.where(np.diff(theta_s) > atol)[0] + 1
    ring_starts = np.concatenate([[0], breaks])
    ring_ends   = np.concatenate([breaks, [N]])

    ring_theta    = np.array([theta_s[s] for s in ring_starts], dtype=np.float64)
    ring_phi_list = [phi_s[s:e].copy() for s, e in zip(ring_starts, ring_ends)]
    ring_counts   = np.array(
        [e - s for s, e in zip(ring_starts, ring_ends)], dtype=np.int64
    )

    return ring_theta, ring_phi_list, ring_counts, order.astype(np.int64)


# ===========================================================================
# compute_weights
# ===========================================================================

def compute_weights(
    ring_theta: np.ndarray,
    ring_phi_list: List[np.ndarray],
    ring_counts: np.ndarray,
    quadrature: str = "trapeze",
) -> np.ndarray:
    """
    Compute per-pixel quadrature weights in steradians.

    The weights satisfy  ∫ f dΩ ≈ Σ_i f_i * w_i  when summed over all
    pixels in ring order.

    Parameters
    ----------
    ring_theta : np.ndarray, shape [R]
        Colatitudes in radians (output of ``build_rings_from_latlon``).
    ring_phi_list : list of np.ndarray
        Longitudes in radians per ring.
    ring_counts : np.ndarray, shape [R]
        Number of pixels per ring.
    quadrature : str, default "trapeze"
        Quadrature rule applied in the colatitude direction.

        ``"trapeze"``
            Trapezoidal rule: w_θ = sin(θ_r) * Δθ_r.
            Accurate for grids with regular θ spacing.
        ``"gauss_legendre"``
            Exact Gauss-Legendre weights for nodes x_r = cos(θ_r).
            **Mandatory for Gaussian grids** (ERA5, ECMWF, IFS, ARPEGE).
            The quadrature is exact up to ℓ ≈ 2R - 1.
            A UserWarning is raised if the provided colatitudes do not match
            Gauss-Legendre nodes within a tolerance of 1e-6.
        ``"equal_area"``
            Uniform weight 4π / N_total.
            Appropriate for equal-area pixelisations such as HEALPix.

    Returns
    -------
    weights : np.ndarray, shape [N_total], dtype float64
        Per-pixel quadrature weights in steradians, in ring order.
    """
    ring_theta  = np.asarray(ring_theta,  dtype=np.float64)
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    R       = len(ring_theta)
    N_total = int(ring_counts.sum())
    phi_list = _parse_phi_list(ring_phi_list, ring_counts)

    # --- equal-area shortcut: same weight for every pixel ---
    if quadrature == "equal_area":
        return np.full(N_total, 4.0 * math.pi / N_total, dtype=np.float64)

    # --- colatitude weights ---
    if quadrature == "trapeze":
        w_theta = np.empty(R, dtype=np.float64)
        for r in range(R):
            if R == 1:
                dth = math.pi
            elif r == 0:
                dth = (ring_theta[1] - ring_theta[0]) / 2.0
            elif r == R - 1:
                dth = (ring_theta[-1] - ring_theta[-2]) / 2.0
            else:
                dth = (ring_theta[r + 1] - ring_theta[r - 1]) / 2.0
            w_theta[r] = abs(math.sin(ring_theta[r]) * dth)

    elif quadrature == "gauss_legendre":
        # For a Gaussian grid the R colatitudes are the zeros of P_R(cos θ).
        # numpy.polynomial.legendre.leggauss returns nodes and weights for
        # ∫_{-1}^{1} f(x) dx ≈ Σ w_r f(x_r), where x = cos θ.
        # Because dΩ = dφ dx (with x = cos θ), these weights are directly
        # the colatitude weights w_theta (sin θ is already absorbed in dx).
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(R)

        # Verify the provided colatitudes match the GL nodes
        x_provided   = np.sort(np.cos(ring_theta))  # ascending
        gl_nodes_asc = np.sort(gl_nodes)
        max_err = np.max(np.abs(x_provided - gl_nodes_asc))
        if max_err > 1e-6:
            warnings.warn(
                f"gauss_legendre: provided colatitudes do not match Gauss-Legendre "
                f"nodes for R={R} (max error = {max_err:.2e}).  "
                "Check that the grid is a proper Gaussian grid with that many "
                "latitude rings.",
                UserWarning,
                stacklevel=2,
            )

        # Map GL weights (sorted ascending in x) back to original ring order
        # (rings are sorted by ascending θ, i.e. descending cos θ)
        sort_prov = np.argsort(np.cos(ring_theta))   # indices: cos(θ) ascending
        sort_gl   = np.argsort(gl_nodes)             # indices: x ascending
        w_theta   = np.empty(R, dtype=np.float64)
        w_theta[sort_prov] = gl_weights[sort_gl]

    else:
        raise ValueError(
            f"Unknown quadrature '{quadrature}'.  "
            "Accepted values: 'trapeze', 'gauss_legendre', 'equal_area'."
        )

    # --- longitude weights and final per-pixel weights ---
    all_w: List[np.ndarray] = []
    for r in range(R):
        N_r   = int(ring_counts[r])
        phi_r = phi_list[r]

        if N_r == 1:
            w_phi = np.array([2.0 * math.pi], dtype=np.float64)
        else:
            sorted_phi = np.sort(phi_r)
            dphi       = np.diff(sorted_phi)
            mean_dp    = 2.0 * math.pi / N_r
            if np.ptp(dphi) < 1e-10 * mean_dp:
                # Uniformly spaced ring: equal longitude weight
                w_phi = np.full(N_r, 2.0 * math.pi / N_r, dtype=np.float64)
            else:
                # Irregular ring: wrapped trapezoidal rule
                gap_wrap = (sorted_phi[0] + 2.0 * math.pi) - sorted_phi[-1]
                dp_ext   = np.concatenate([[gap_wrap], dphi, [gap_wrap]])
                w_sorted = (dp_ext[:-1] + dp_ext[1:]) / 2.0
                # Restore original pixel order
                back   = np.argsort(np.argsort(phi_r))
                w_phi  = w_sorted[back]

        all_w.append(w_theta[r] * w_phi)

    return np.concatenate(all_w)


# ===========================================================================
# comp_tf_latlon  — per-ring longitude Fourier transform
# ===========================================================================

def comp_tf_latlon(
    im: torch.Tensor,
    ring_phi_list: List[np.ndarray],
    ring_counts: np.ndarray,
    pixel_weights: np.ndarray,
    mmax: int,
    cdtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """
    Compute the weighted longitude DFT of a map, ring by ring.

    For each ring r with N_r pixels at longitudes φ_j:

        F_r(m) = Σ_j  f(θ_r, φ_j) * w_j * exp(-i m φ_j)

    For uniformly-spaced rings the computation uses torch.fft.rfft followed
    by a phase-shift exp(-i m φ_0), matching the FOSCAT comp_tf convention.
    For irregular rings a direct DFT is used.

    Parameters
    ----------
    im : torch.Tensor, shape [..., N_total]
        Map values in ring order (apply sort_idx before calling).
    ring_phi_list : list of np.ndarray
        Longitudes in radians per ring.
    ring_counts : np.ndarray, shape [R]
        Number of pixels per ring.
    pixel_weights : np.ndarray, shape [N_total]
        Per-pixel quadrature weights (output of ``compute_weights``).
    mmax : int
        Maximum azimuthal order to compute.
    cdtype : torch.dtype, default torch.complex128
        Complex dtype for the output tensor.

    Returns
    -------
    ft : torch.Tensor, shape [..., R, mmax+1], complex
        Ring-by-ring Fourier coefficients.
    """
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    R           = len(ring_counts)
    m_vec       = np.arange(mmax + 1, dtype=np.float64)

    out: List[torch.Tensor] = []
    offset = 0

    for r in range(R):
        N_r   = int(ring_counts[r])
        phi_r = np.asarray(ring_phi_list[r], dtype=np.float64)
        w_r   = pixel_weights[offset : offset + N_r]
        v     = im[..., offset : offset + N_r]       # [..., N_r]
        offset += N_r

        is_unif, phi0 = _check_uniform_phi(phi_r)

        if is_unif:
            # --- FFT path ---
            # Sort pixels by φ into ascending order
            sort_phi = np.argsort(phi_r)
            v_sorted = v[..., sort_phi]              # [..., N_r]

            # All longitude weights are equal for a uniform ring.
            # Absorb the single scalar weight into the signal.
            w_scalar = float(w_r[0])
            #v_weighted = v_sorted.to(cdtype) * w_scalar corrected
            v_weighted = v_sorted.to(torch.float64) * w_scalar

            # Full complex spectrum via rfft (avoids computing negative freqs twice)
            tmp = _rfft2fft(v_weighted)              # [..., N_r]

            # Tile to reach mmax+1 if the ring has fewer pixels than modes
            l_n = tmp.shape[-1]
            if l_n < mmax + 1:
                repeat_n = (mmax // l_n) + 1
                #tmp = tmp.repeat_interleave(repeat_n, dim=-1)
                repeats = [1] * (tmp.ndim - 1) + [repeat_n]
                tmp = tmp.repeat(repeats)
            tmp = tmp[..., : mmax + 1]               # [..., mmax+1]

            # Apply per-ring phase shift exp(-i m φ_0)
            shift = torch.tensor(
                np.exp(-1j * m_vec * phi0), dtype=cdtype,device=tmp.device,
            )
            tmp = tmp * shift

        else:
            # --- Direct DFT path (irregular ring) ---
            # kernel[j, m] = w_j * exp(-i m φ_j),  shape [N_r, mmax+1]
            ang = np.outer(phi_r, m_vec)             # [N_r, M]
            ker = (np.exp(-1j * ang) * w_r[:, None]).astype(np.complex128)
            ker_t = torch.tensor(ker, dtype=cdtype)  # [N_r, M]

            # ft[..., m] = Σ_j v[..., j] * ker[j, m]
            tmp = (v.to(cdtype).unsqueeze(-1) * ker_t).sum(dim=-2)  # [..., M]

        out.append(tmp.unsqueeze(-2))                # [..., 1, mmax+1]

    return torch.cat(out, dim=-2)                    # [..., R, mmax+1]


# ===========================================================================
# map2alm_latlon
# ===========================================================================

def map2alm_latlon(
    im: Union[np.ndarray, torch.Tensor],
    ring_theta: np.ndarray,
    ring_phi_list: Union[List[np.ndarray], np.ndarray],
    ring_counts: np.ndarray,
    lmax: int = 64,
    weights: Optional[Union[np.ndarray, str]] = None,
    quadrature: str = "trapeze",
    limit_range: float = 1e10,
) -> torch.Tensor:
    """
    Compute spherical harmonic coefficients a_lm from a map.

    The transform follows the standard orthonormal convention (same as healpy):

        a_lm = ∫ f(θ,φ) Y_lm*(θ,φ) dΩ

    Parameters
    ----------
    im : array-like or torch.Tensor, shape [..., N_total]
        Map values in ring order (apply ``sort_idx`` from
        ``build_rings_from_latlon`` first).  An optional leading batch
        dimension is supported.
    ring_theta : np.ndarray, shape [R]
        Colatitudes in radians (output of ``build_rings_from_latlon``).
    ring_phi_list : list of np.ndarray or flat np.ndarray [N_total]
        Longitudes in radians per ring.
    ring_counts : np.ndarray, shape [R]
        Number of pixels per ring.
    lmax : int, default 24
        Maximum multipole ℓ_max.
    weights : np.ndarray [N_total], None, or "uniform"
        Per-pixel quadrature weights.

        - ``None`` : computed automatically via ``compute_weights(quadrature=quadrature)``.
        - ``"uniform"`` : w_i = 1/N_total (comparable to healpy.map2alm).
        - array : user-supplied weights (``quadrature`` is then ignored).
    quadrature : str, default "trapeze"
        Quadrature rule passed to ``compute_weights``.
        Accepted values: ``"trapeze"``, ``"gauss_legendre"``, ``"equal_area"``.
        Ignored when ``weights`` is not ``None``.

        .. note::
            Use ``"gauss_legendre"`` for ERA5/ECMWF Gaussian grids to avoid
            oscillations at high ℓ.

    limit_range : float, default 1e10
        Overflow guard in the Legendre recurrence.

    Returns
    -------
    alm_out : torch.Tensor, shape [..., n_alm], complex128
        Spherical harmonic coefficients where
        n_alm = Σ_{m=0}^{lmax} (lmax - m + 1).

        Memory layout (same as healpy)::

            [m=0: ℓ=0..lmax | m=1: ℓ=1..lmax | … | m=lmax: ℓ=lmax]
    """
    ring_theta  = np.asarray(ring_theta,  dtype=np.float64)
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    phi_list    = _parse_phi_list(ring_phi_list, ring_counts)
    N_total     = int(ring_counts.sum())

    # --- convert input to torch ---
    if not torch.is_tensor(im):
        im = torch.tensor(np.asarray(im), dtype=torch.float64)
    else:
        im = im.to(torch.float64)

    # --- handle optional batch dimension ---
    added_batch = False
    if im.ndim == 1:
        im = im.unsqueeze(0)
        added_batch = True

    # --- quadrature weights ---
    if weights is None:
        pixel_weights = compute_weights(ring_theta, phi_list, ring_counts, quadrature)
    elif isinstance(weights, str) and weights == "uniform":
        pixel_weights = np.ones(N_total, dtype=np.float64) / N_total
    else:
        pixel_weights = np.asarray(weights, dtype=np.float64)

    # --- step 1: per-ring longitude DFT ---
    # ft shape: [..., R, lmax+1], complex128
    ft = comp_tf_latlon(
        im, phi_list, ring_counts, pixel_weights,
        mmax=lmax, cdtype=torch.complex128
    )

    # cos(θ) for the Legendre projection
    co_th = np.cos(ring_theta)   # [R]

    # --- step 2: Legendre projection for each m ---
    alm_parts: List[torch.Tensor] = []

    for m in range(lmax + 1):
        # Legendre polynomials at all ring colatitudes
        # plm_raw shape: [lmax-m+1, R], scaled by sqrt(4π)
        plm_raw = _compute_legendre_m(co_th, m, lmax, limit_range=limit_range)

        # Apply the Y_lm normalisation factor sqrt((2l+1) / 4π)
        # so that plm[l-m, r] = Y_lm(θ_r, 0)  (no φ-dependence here)
        l_vals     = np.arange(m, lmax + 1, dtype=np.float64)
        ylm_factor = np.sqrt(2.0 * l_vals + 1.0) / (4.0 * math.pi)  # [L]
        plm = plm_raw * ylm_factor[:, np.newaxis]                     # [L, R]

        plm_t = torch.tensor(plm, dtype=torch.complex128)             # [L, R]

        # ft_m: [..., R]  — Fourier coefficient at order m for every ring
        ft_m = ft[..., :, m]

        # a_lm[l] = Σ_r  ft_m[r] * Y_lm(θ_r, 0)
        # ft_m unsqueezed: [..., 1, R] × plm_t [L, R] → sum over R → [..., L]
        tmp = (ft_m.unsqueeze(-2) * plm_t).sum(dim=-1)   # [..., L]
        alm_parts.append(tmp)

    alm_out = torch.cat(alm_parts, dim=-1)               # [..., n_alm]

    if added_batch:
        alm_out = alm_out[0]

    return alm_out


# ===========================================================================
# alm2map_latlon
# ===========================================================================

def alm2map_latlon(
    alm: torch.Tensor,
    ring_theta: np.ndarray,
    ring_phi_list: Union[List[np.ndarray], np.ndarray],
    ring_counts: np.ndarray,
    lmax: int = 24,
    limit_range: float = 1e10,
) -> torch.Tensor:
    """
    Synthesise a map from spherical harmonic coefficients (inverse transform).

    This is the adjoint of ``map2alm_latlon``.  The output map is evaluated
    at the same grid positions as the input to the analysis, but it can also
    be used to synthesise on a different target grid.

    Parameters
    ----------
    alm : torch.Tensor, shape [..., n_alm], complex
        Spherical harmonic coefficients in the layout produced by
        ``map2alm_latlon``.
    ring_theta : np.ndarray, shape [R]
        Colatitudes in radians of the target grid.
    ring_phi_list : list of np.ndarray or flat np.ndarray [N_total]
        Longitudes in radians of the target grid.
    ring_counts : np.ndarray, shape [R]
        Number of pixels per ring of the target grid.
    lmax : int, default 24
        Maximum multipole used when the alm were computed.
    limit_range : float, default 1e10
        Overflow guard in the Legendre recurrence.

    Returns
    -------
    im_out : torch.Tensor, shape [..., N_total], float64
        Reconstructed map values in ring order.
    """
    ring_theta  = np.asarray(ring_theta,  dtype=np.float64)
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    phi_list    = _parse_phi_list(ring_phi_list, ring_counts)
    R           = len(ring_theta)

    added_batch = False
    if alm.ndim == 1:
        alm = alm.unsqueeze(0)
        added_batch = True

    alm = alm.to(torch.complex128)
    batch_shape = alm.shape[:-1]
    device      = alm.device

    co_th = np.cos(ring_theta)   # [R]

    # --- accumulate Fourier modes on each ring ---
    # ft_synth[..., r, m] = Σ_l  a_lm * Y_lm(θ_r, 0)
    ft_synth = torch.zeros(
        batch_shape + (R, lmax + 1),
        dtype=torch.complex128,
        device=device,
    )

    idx = 0
    for m in range(lmax + 1):
        L      = lmax - m + 1
        alm_m  = alm[..., idx : idx + L]   # [..., L]
        idx   += L

        plm_raw = _compute_legendre_m(co_th, m, lmax, limit_range=limit_range)
        l_vals     = np.arange(m, lmax + 1, dtype=np.float64)
        ylm_factor = np.sqrt(2.0 * l_vals + 1.0) / (4.0 * math.pi)
        plm = plm_raw * ylm_factor[:, np.newaxis]          # [L, R]
        plm_t = torch.tensor(plm, dtype=torch.complex128, device=device)

        # ft[..., r, m] = Σ_l  alm_m[l] * plm[l, r]
        # alm_m [..., L, 1] × plm_t [L, R] → sum over L → [..., R]
        contrib = (alm_m.unsqueeze(-1) * plm_t).sum(dim=-2)  # [..., R]
        ft_synth[..., :, m] = contrib

    # --- per-ring inverse DFT: f(θ_r, φ_j) = Re[ Σ_m ft[r,m] * exp(im φ_j) ] ---
    out_rings: List[torch.Tensor] = []
    m_vec = np.arange(lmax + 1, dtype=np.float64)

    for r in range(R):
        N_r   = int(ring_counts[r])
        phi_r = phi_list[r]

        # kernel[m, j] = exp(i m φ_j),  shape [lmax+1, N_r]
        ang   = np.outer(m_vec, phi_r)               # [M, N_r]
        ker   = np.exp(1j * ang).astype(np.complex128)
        ker_t = torch.tensor(ker, dtype=torch.complex128, device=device)

        ft_r = ft_synth[..., r, :]                   # [..., M]

        # pixel[j] = Σ_m ft_r[m] * exp(im φ_j)
        # ft_r [..., M, 1] × ker_t [M, N_r] → sum over M → [..., N_r]
        pix = (ft_r.unsqueeze(-1) * ker_t).sum(dim=-2)   # [..., N_r]
        out_rings.append(pix.real)

    im_out = torch.cat(out_rings, dim=-1)            # [..., N_total]

    if added_batch:
        im_out = im_out[0]

    return im_out


# ===========================================================================
# anafast_latlon
# ===========================================================================

def anafast_latlon(
    im: Union[np.ndarray, torch.Tensor],
    ring_theta: np.ndarray,
    ring_phi_list: Union[List[np.ndarray], np.ndarray],
    ring_counts: np.ndarray,
    lmax: int = 64,
    weights: Optional[Union[np.ndarray, str]] = None,
    quadrature: str = "trapeze",
    limit_range: float = 1e10,
) -> torch.Tensor:
    """
    Estimate the angular power spectrum C_l from a map.

    Internally calls ``map2alm_latlon`` then accumulates:

        C_l = 1/(2l+1) * [|a_l0|² + 2 Σ_{m=1}^{l} |a_lm|²]

    All parameters are identical to ``map2alm_latlon``.

    Returns
    -------
    cl : torch.Tensor, shape [..., lmax+1], float64
        Angular power spectrum.
    """
    alm = map2alm_latlon(
        im, ring_theta, ring_phi_list, ring_counts,
        lmax=lmax, weights=weights, quadrature=quadrature,
        limit_range=limit_range,
    )

    # --- handle batch dimension consistently ---
    added_batch = False
    if alm.ndim == 1:
        alm = alm.unsqueeze(0)
        added_batch = True

    batch_shape = alm.shape[:-1]
    device      = alm.device

    cl = torch.zeros(batch_shape + (lmax + 1,), dtype=torch.float64, device=device)

    idx = 0
    for m in range(lmax + 1):
        L = lmax - m + 1
        a = alm[..., idx : idx + L]          # [..., L]
        idx += L

        power  = (a * a.conj()).real          # |a_lm|², shape [..., L]
        weight = 1.0 if m == 0 else 2.0      # account for m and -m
        cl[..., m : m + L] += weight * power

    # Normalise by (2l+1)
    two_l_p1 = (2.0 * torch.arange(lmax + 1, dtype=torch.float64, device=device) + 1.0)
    two_l_p1 = two_l_p1.reshape((1,) * len(batch_shape) + (lmax + 1,))
    cl = cl / two_l_p1

    if added_batch:
        cl = cl[0]

    return cl


# ===========================================================================
# grid_summary
# ===========================================================================

def grid_summary(
    ring_theta: np.ndarray,
    ring_phi_list: Union[List[np.ndarray], np.ndarray],
    ring_counts: np.ndarray,
    lmax: int = 24,
) -> dict:
    """
    Print a human-readable summary of the grid and estimated compute cost.

    Parameters
    ----------
    ring_theta : np.ndarray, shape [R]
    ring_phi_list : list of np.ndarray
    ring_counts : np.ndarray, shape [R]
    lmax : int

    Returns
    -------
    info : dict
        Dictionary with the same fields printed to stdout.
    """
    ring_theta  = np.asarray(ring_theta,  dtype=np.float64)
    ring_counts = np.asarray(ring_counts, dtype=np.int64)
    phi_list    = _parse_phi_list(ring_phi_list, ring_counts)
    N_total     = int(ring_counts.sum())
    R           = len(ring_theta)

    n_uniform = sum(1 for r in range(R) if _check_uniform_phi(phi_list[r])[0])
    n_alm     = sum(lmax - m + 1 for m in range(lmax + 1))

    # Rough operation count estimate
    cost_fft = sum(
        int(ring_counts[r]) * math.log2(max(2, int(ring_counts[r])))
        for r in range(R)
        if _check_uniform_phi(phi_list[r])[0]
    )
    cost_dft = sum(
        int(ring_counts[r]) * (lmax + 1)
        for r in range(R)
        if not _check_uniform_phi(phi_list[r])[0]
    )

    print(f"=== Grid summary  (lmax={lmax}) ===")
    print(f"  Total pixels      : {N_total}")
    print(f"  Latitude rings    : {R}")
    print(f"  Uniform rings     : {n_uniform}/{R}  (FFT acceleration)")
    print(f"  θ range           : [{np.degrees(ring_theta.min()):.3f}°, "
          f"{np.degrees(ring_theta.max()):.3f}°]")
    print(f"  Pixels/ring       : min={ring_counts.min()}, "
          f"max={ring_counts.max()}, mean={ring_counts.mean():.1f}")
    print(f"  n_alm             : {n_alm}")
    print(f"  Cost estimate     : {cost_fft + cost_dft:.2e} ops  "
          f"(FFT: {cost_fft:.2e}, DFT: {cost_dft:.2e})")

    return {
        "N_total":   N_total,
        "n_rings":   R,
        "n_uniform": n_uniform,
        "n_alm":     n_alm,
        "theta_min_deg": float(np.degrees(ring_theta.min())),
        "theta_max_deg": float(np.degrees(ring_theta.max())),
    }
