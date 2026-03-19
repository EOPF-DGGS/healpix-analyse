import numpy as np
import torch

from healpix_analyse.alm import AlmTransform

from typing import Generic, Tuple, Optional, Sequence,Union

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]

import numpy as np
import torch

from healpix_analyse.alm import AlmTransform

from typing import Sequence, Union, Optional

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]


def powerspectra_lonlat(
    lon: ArrayLike,
    lat: ArrayLike,
    data: ArrayLike,
    dx: float = 1.0,
    cross: Optional[ArrayLike] = None,
    ellipsoid: str = "sphere",
    plot_2D_fft=False,
    weights: np.ndarray = None
):
    """
    Compute the isotropic 1D power spectrum of a field sampled on an
    iso-latitude lon/lat grid (e.g. GRIB-like grid), without using any
    HEALPix-specific concepts.

    Parameters
    ----------
    lon : array-like, shape [N]
        Longitudes of the samples.
    lat : array-like, shape [N]
        Latitudes of the samples.
        The sampling is assumed to form iso-latitude rings.
    data : array-like, shape [N]
        Field values at (lon, lat).
    dx : float, default=1.0
        Effective sampling step used to label Fourier frequencies.
        Keep the same convention as in your current `powerspectra`.
    cross : array-like, optional
        Second field for cross-spectrum.
    ellipsoid : {"sphere", "WGS84"}, default="sphere"
        Kept for compatibility with AlmTransform.

    Returns
    -------
    f_centers : ndarray
        Radial spatial frequencies.
    Pk : ndarray
        Azimuthally averaged 1D power spectrum.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    data = np.asarray(data)

    if lon.ndim != 1 or lat.ndim != 1 or data.ndim != 1:
        raise ValueError("lon, lat and data must be 1D arrays")
    if not (lon.shape == lat.shape == data.shape):
        raise ValueError("lon, lat and data must have the same shape")

    # Optional safety if the GRIB domain crosses the longitude discontinuity
    lon_rad = np.deg2rad(lon) if np.nanmax(np.abs(lon)) > 2.0 * np.pi else lon
    lon_unwrapped = np.unwrap(lon_rad)
    if np.nanmax(np.abs(lon)) > 2.0 * np.pi:
        lon_for_transform = np.rad2deg(lon_unwrapped)
    else:
        lon_for_transform = lon_unwrapped

    # Dummy cell_ids/level since we explicitly provide lon/lat to AlmTransform.
    # They are not used for geometry in this code path.
    dummy_cell_ids = np.arange(lon.size, dtype=np.int64)

    ltf = AlmTransform(
        cell_ids=dummy_cell_ids,
        level=0,
        ellipsoid=ellipsoid,
        lon=lon_for_transform,
        lat=lat,
        weights=weights,
    )

    F = np.fft.fftshift(ltf.fft(data))

    
    del ltf

            
    if cross is not None:
        cross = np.asarray(cross)
        if cross.shape != data.shape:
            raise ValueError("cross must have the same shape as data")
        F2 = np.fft.fftshift(ltf.fft(cross))
        P2D = (F * np.conjugate(F2)).real
    else:
        P2D = np.abs(F) ** 2

    if plot_2D_fft:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.fft.fftshift(P2D), norm='log')
        plt.colorbar(label='Power (log scale)', orientation='horizontal')
        plt.show()

    ny, nx = F.shape
    ldy=dx
    ldx=dx
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=ldx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=ldy))
    fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
    fr = np.sqrt(fx2d**2 + fy2d**2)

    nbins = min(nx, ny) // 2
    f_bins = np.linspace(0.0, fr.max(), nbins + 1)

    fr_flat = fr.ravel()
    P_flat = P2D.ravel()
    bin_idx = np.digitize(fr_flat, f_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)

    sum_per_bin = np.bincount(bin_idx[valid], weights=P_flat[valid], minlength=nbins)
    cnt_per_bin = np.bincount(bin_idx[valid], minlength=nbins)

    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = sum_per_bin / cnt_per_bin
    Pk[cnt_per_bin == 0] = np.nan

    f_centers = 0.5 * (f_bins[1:] + f_bins[:-1])

    return f_centers, Pk / (nx * ny)**2
    
    