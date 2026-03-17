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


def powerspectra(cell_ids,
                 level,
                 data,
                ellipsoid: str = "sphere",
                method: str = "fft",
                 dx=1.0,
                 cross=None):
        """
        Compute the isotropic 1D power spectrum of a 2D field.
    
        Parameters
        ----------
        data : ndarray (ny, nx)
            Input 2D field.
        dx : float
            Pixel size in the same spatial unit as desired frequency inverse.
            If dx is in meters, returned frequencies are in m^-1 (cycles per meter).
    
        Returns
        -------
        f_centers : ndarray
            Radial spatial frequencies (cycles per unit length), e.g., m^-1 if dx is in meters.
        Pk : ndarray
            Azimuthally averaged power spectrum over radial frequency bins (arbitrary units unless you add a normalization).
        """
        # 2D FFT and power
        
        ltf = AlmTransform(cell_ids,level,ellipsoid=ellipsoid)
        
        F = np.fft.fftshift(ltf.fft(data))
        
        if cross is not None:
            F2 = np.fft.fftshift(ltf.fft(cross))
            P2D = (F*np.conjugate(F2)).real
        else:
            P2D = np.abs(F) ** 2
        del ltf
    
        # Spatial frequency grids (cycles per unit length; NOT radians)
        ny, nx = F.shape
        fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # cycles per unit length (e.g., m^-1)
        fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))  # cycles per unit length (e.g., m^-1)
        fx2d, fy2d = np.meshgrid(fx, fy, indexing="xy")
        fr = np.sqrt(fx2d**2 + fy2d**2)  # radial spatial frequency (cycles per unit length)
    
        # Radial binning
        nbins = min(nx, ny) // 2
        f_bins = np.linspace(0.0, fr.max(), nbins + 1)
    
        # Vectorized bin average of P2D over annuli
        fr_flat = fr.ravel()
        P_flat = P2D.ravel()
        bin_idx = np.digitize(fr_flat, f_bins) - 1  # -> [0, nbins-1]
        valid = (bin_idx >= 0) & (bin_idx < nbins)
    
        # Sum and count per bin, then mean
        sum_per_bin = np.bincount(bin_idx[valid], weights=P_flat[valid], minlength=nbins)
        cnt_per_bin = np.bincount(bin_idx[valid], minlength=nbins)
        with np.errstate(invalid="ignore", divide="ignore"):
            Pk = sum_per_bin / cnt_per_bin
        Pk[cnt_per_bin == 0] = np.nan  # empty bins
    
        # Bin centers
        f_centers = 0.5 * (f_bins[1:] + f_bins[:-1])
    
        return f_centers, Pk/(nx*ny)