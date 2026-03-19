from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import math
import time

import numpy as np
import torch

import healpix_geo


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float], Sequence[int]]


@dataclass
class AlmCoeffs:
    """
    Container for local complex spherical harmonic coefficients.

    Parameters
    ----------
    alm : torch.Tensor
        Complex coefficient tensor with shape [K] or [B, K].
    l : torch.Tensor
        Degree indices with shape [K].
    m : torch.Tensor
        Order indices with shape [K]. Only m >= 0 is stored in V1.
    meta : dict
        Metadata associated with the transform.
    """

    alm: torch.Tensor
    l: torch.Tensor
    m: torch.Tensor
    meta: Dict[str, Any]

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.alm):
            raise TypeError("alm must be a torch.Tensor")
        if not torch.is_tensor(self.l):
            raise TypeError("l must be a torch.Tensor")
        if not torch.is_tensor(self.m):
            raise TypeError("m must be a torch.Tensor")
        if self.l.ndim != 1 or self.m.ndim != 1:
            raise ValueError("l and m must be 1D tensors")
        if self.l.shape != self.m.shape:
            raise ValueError("l and m must have the same shape")
        if self.alm.ndim not in (1, 2):
            raise ValueError("alm must have shape [K] or [B, K]")
        if self.alm.shape[-1] != self.l.numel():
            raise ValueError("The last alm dimension must match len(l) and len(m)")

    @property
    def n_modes(self) -> int:
        return int(self.l.numel())

    @property
    def is_batched(self) -> bool:
        return self.alm.ndim == 2

    def clone(self) -> "AlmCoeffs":
        return AlmCoeffs(
            alm=self.alm.clone(),
            l=self.l.clone(),
            m=self.m.clone(),
            meta=dict(self.meta),
        )

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "AlmCoeffs":
        return AlmCoeffs(
            alm=self.alm.to(device=device, dtype=dtype),
            l=self.l.to(device=device),
            m=self.m.to(device=device),
            meta=dict(self.meta),
        )

    def cpu(self) -> "AlmCoeffs":
        return self.to(device="cpu")

    def cuda(self) -> "AlmCoeffs":
        return self.to(device="cuda")


class AlmTransform:
    """
    Local FFT-oriented spherical harmonic transform on a HEALPix patch.

    This implementation follows a two-stage local solver:

    1. local Fourier analysis on each observed iso-latitude ring,
    2. local Legendre analysis over the observed ring colatitudes.

    The design avoids explicit dense Y_lm evaluation over all pixels. It is
    intended as a practical starting point for large local patches and keeps
    only m >= 0 coefficients.

    Parameters
    ----------
    level : int
        HEALPix level such that nside = 2**level.
    cell_ids : array-like
        HEALPix cell identifiers at the given level.
    indexing_scheme : str, default="ring"
        The indexing scheme for the cell IDs.
    ellipsoid : {"sphere", "WGS84"}, default="sphere"
        Geometry model. Ignored in this version but stored for compatibility.
    method : {"fft", "alm"}, default="fft", Alm approximation, alm option under development.
        Analysis mode used in the Legendre stage.
    dtype : torch.dtype, default=torch.float32
        Real dtype for maps and geometry.
    device : str or torch.device, optional
        Execution device. If None, uses CUDA when available, else CPU.
    debug : bool, default=False
        Whether to store diagnostic timing information.

    Notes
    -----
    Current assumptions:
    - all input cells are at the same HEALPix level,
    - reconstruction is only performed on the same input cells,
    - input maps are real-valued.
    """

    def __init__(
        self,
        cell_ids: ArrayLike,
        level: int,
        indexing_scheme: str = "ring",
        ellipsoid: str = "sphere",
        method: str = "fft",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        debug: bool = False,
        lon: np.ndarray = None,
        lat: np.ndarray = None,
        weights: np.array = None
    ) -> None:

        self.level = int(level)
        self.indexing_scheme = str(indexing_scheme)
        self.ellipsoid = str(ellipsoid)
        self.method = str(method)
        self.dtype = dtype
        self.cdtype = self._infer_complex_dtype(dtype)
        device='cpu'
        self.device = self._resolve_device(device)
        self.debug = bool(debug)

        if self.ellipsoid not in {"sphere", "WGS84"}:
            raise ValueError("ellipsoid must be 'sphere' or 'WGS84'")
        if self.method not in {"fft", "alm"}:
            raise ValueError("method must be 'fft' or 'alm'")
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.cell_ids = self._as_long_tensor(cell_ids, device=self.device)
        if self.cell_ids.ndim != 1:
            raise ValueError("cell_ids must be a 1D array-like")
        if self.cell_ids.numel() == 0:
            raise ValueError("cell_ids must not be empty")

        self.size = int(self.cell_ids.numel())
        

        if lon is None:
            if self.indexing_scheme == "ring":
                lon, lat = healpix_geo.ring.healpix_to_lonlat(cell_ids, level, ellipsoid=ellipsoid)
                self.idx_ordering = lon.argsort() # TODO: optimize knowing that it is already sorted
            else:
                if self.indexing_scheme == "nested":
                    lon, lat = healpix_geo.nested.healpix_to_lonlat(cell_ids, level, ellipsoid=ellipsoid)
                else:
                    raise NotImplementedError("For now, indexing_scheme must be 'ring' or 'nested'")
                self.idx_ordering = lon.argsort()
        else:
            self.idx_ordering = lon.argsort()
        
        # TODO: optimize knowing that it is already sorted
        theta_uniq, idx_ring = np.unique(lat[self.idx_ordering], return_inverse=True)
    
        self.n_rings = len(theta_uniq)
        self.N_k = np.bincount(idx_ring)
        self.N_max = int(np.max(self.N_k))
        self.idx_ring = idx_ring
        self.lon = (lon - np.min(lon))[self.idx_ordering]
        if weights is None:
            self.weights = self.N_max/self.N_k
        

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shift_from_fft_phase(self,fft_tensor, x0, dx=1.0, dim=-1):
        """
        Applique un décalage spatial x0 à un signal via sa FFT déjà calculée.

        Parameters
        ----------
        fft_tensor : torch.Tensor
            FFT déjà calculée (complexe).
        x0 : float or torch.Tensor
            Décalage à appliquer dans l'espace réel.
            Convention: f(x - x0) <-> F(k) * exp(-i 2π k x0)
        dx : float
            Pas d'échantillonnage dans l'espace réel.
        dim : int
            Dimension correspondant à la FFT 1D.

        Returns
        -------
        torch.Tensor
            FFT modifiée par la rampe de phase.
        """
        n = fft_tensor.shape[dim]
        device = fft_tensor.device
        dtype = fft_tensor.real.dtype

        # fréquences en cycles / unité de x
        k = torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)

        # facteur de phase
        phase = torch.exp(-2j * torch.pi * k * x0)

        # reshape pour broadcast sur la bonne dimension
        shape = [1] * fft_tensor.ndim
        shape[dim] = n
        phase = phase.reshape(shape)

        return fft_tensor * phase
        
    def fft(
        self,
        data: ArrayLike,
    ) -> ArrayLike:
        # TODO: implement rfft
         
        ldata = data[self.idx_ordering]
        out_fft = torch.zeros([self.n_rings*2,self.N_max], dtype=self.cdtype)
        
        for k in range(self.n_rings):
            idx = np.where(self.idx_ring==k)[0] # TODO: optimize knowing that it is already sorted?????????????
            if not torch.is_tensor(ldata):
                ldata = torch.as_tensor(ldata)
            ring_data = ldata[idx]
            
            # 1D FFT on latitude ring k
            if torch.is_complex(ring_data): 
                tmp_fft = torch.fft.fft(ring_data)
                cutoff = ring_data.shape[0]//2+1 # si n pair : coupure à n/2 inclus, si n impair : coupure à (n-1)/2 = n//2 inclus
                tmp_pos_freq = tmp_fft[:cutoff] 
                tmp_neg_freq = tmp_fft[cutoff:]
            else:
                tmp_fft = torch.fft.rfft(ring_data)
                tmp_pos_freq = tmp_fft # len n//2+1
                tmp_neg_freq = tmp_fft[1:ring_data.shape[0] - ring_data.shape[0]//2] # do not copy Nyquist if n even, len n - n//2 - 1
                tmp_neg_freq = tmp_neg_freq.flip(dims=[0]) # reverse negative frequencies to match FFT convention

            # apply upsampling with padding in Fourier space
            tmp_fft = torch.zeros(self.N_max, dtype=tmp_fft.dtype, device=tmp_fft.device)
            tmp_fft[:tmp_pos_freq.shape[0]] = tmp_pos_freq
            tmp_fft[-tmp_neg_freq.shape[0]:] = tmp_neg_freq

            # Apply phase shift to align with the original longitude
            tmp_fft = self.shift_from_fft_phase(tmp_fft, self.lon[idx[0]])
            # TODO: raise Warning if phase shift is large

            # Normalize
            out_fft[k]  = tmp_fft * self.N_max / self.N_k[k]
            if k>0:
                out_fft[-k] = - out_fft[k].flip(dims=[0])
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(abs(out_fft)+1E-7, norm='log')
        plt.colorbar(label='Power (log scale)', orientation='horizontal')

        
        # parallel 1D FFTs along latitude axis
        # TODO: NUFFT or direct Legendre analysis for the second stage, currently only FFT stage implemented
        out_fft = np.fft.fft(out_fft) 
        #out_fft = torch.fft.nufft(out_fft, dim=1)
        
        # TODO: Legendre analysis over colatitude to get final Alm coefficients, currently only FFT stage implemented
        # TODO: Should we reorder the output to match the original cell ordering?

        return out_fft


    def ifft(
        self,
        data: ArrayLike,
    ) -> ArrayLike:
        """Reconstruct a map on the same local HEALPix cells."""
        o_fft = np.fft.ifft(data_fft)
    
        idata = np.zeros([self.size])
        for k in range(self.n_rings):
            idx = np.where(self.idx_ring==k)[0]
            inv_fft = o_fft[:,k]/self.weights[k]
            inv_fft = self.shift_from_fft_phase(inv_fft,-self.xa[idx[0]])
            idata[idx]=torch.fft.ifft(inv_fft)[0:self.N_k[k]].real
        return idata 

    
    @staticmethod
    def _infer_complex_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype == torch.float32:
            return torch.complex64
        if dtype == torch.float64:
            return torch.complex128
        raise ValueError("Unsupported real dtype")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _as_long_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long)
        return torch.as_tensor(x, device=device, dtype=torch.long)

    @staticmethod
    def _as_real_tensor(
        x: ArrayLike,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(x, device=device, dtype=dtype)
