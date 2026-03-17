#!/usr/bin/env python3
"""
LocalizedFlatSkyAlm.py

Modified from LocalizedAlm_optimized_v13_fixed.py.
- Removed exact Y_lm (Legendre Polynomial) precomputation to solve RAM explosion at high N_side.
- Implements Phase-Aligned Flat-Sky Approximation using a 2D FFT (1D Longitude + 1D Latitude).
- Isotropically bins the 2D Cartesian power spectrum into 1D multipoles (ell).
"""

import numpy as np
import torch
import foscat.alm as alm_base

# =============================================================================
# Backend utilities
# =============================================================================
class _AlmBackendAdapter:
    def __init__(self, bk_backend):
        self.backend = bk_backend

def _as_alm_backend_object(obj):
    if hasattr(obj, "bk_log") and hasattr(obj, "bk_cast"):
        return _AlmBackendAdapter(obj)
    if hasattr(obj, "backend") and hasattr(obj.backend, "bk_log"):
        return obj
    raise TypeError("Invalid backend.")

def _get_torch_device_from_bk(bk):
    if hasattr(bk, "torch_device"): return bk.torch_device
    if hasattr(bk, "device"): return bk.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# LocalizedFlatSkyAlm class
# =============================================================================
class LocalizedFlatSkyAlm(alm_base.alm):
    """
    Localized spherical harmonic analysis using a Phase-Aligned Flat-Sky FFT.
    Avoids O(lmax^2) memory bottlenecks for ultra-high resolution maps.
    """
    
    def __init__(self, nside, patch_idx_ring, backend, lmax=None, lmax_compute=None):
        patch_idx_ring = np.asarray(patch_idx_ring, dtype=np.int64)
        if patch_idx_ring.size > 1:
            patch_idx_ring = np.sort(patch_idx_ring)
        
        alm_backend = _as_alm_backend_object(backend)
        super().__init__(backend=alm_backend, lmax=lmax, nside=nside)
        
        self.nside = nside
        self.patch_idx_ring = patch_idx_ring
        self.device = _get_torch_device_from_bk(self.backend)
        self.dtype = self.backend.all_bk_type
        self.lmax_full = int(self.lmax)
        self.lmax_compute = lmax_compute if lmax_compute is not None else self.lmax_full
        
        self._precompute_geometry(nside, patch_idx_ring)
        
        self.shift_ph(nside)
        full_shift = self.matrix_shift_ph[nside]
        idx_t = torch.as_tensor(self.active_ring_indices_global, dtype=torch.long, device=full_shift.device)
        self.active_shift_ph = full_shift.index_select(0, idx_t)
        
        self._build_torch_scatter_tables()
        
        print(f"  LocalizedFlatSkyAlm ready: {len(self.active_rings)} active rings, lmax_compute={self.lmax_compute}")

    def _precompute_geometry(self, nside, patch_idx):
        rings_info = []
        pix_cursor = 0
        
        for k in range(nside - 1):
            length = 4 * (k + 1)
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        for k in range(2 * nside + 1):
            length = 4 * nside
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        for k in range(nside - 1):
            length = 4 * (nside - 1 - k)
            rings_info.append({"start": pix_cursor, "length": length, "global_idx": len(rings_info)})
            pix_cursor += length
        
        active_rings = []
        active_ring_indices_global = []
        u_indices = []
        ring_batch_indices = []
        ring_pix_indices = []
        
        for r in rings_info:
            s = r["start"]
            e = s + r["length"]
            start_search = np.searchsorted(patch_idx, s)
            end_search = np.searchsorted(patch_idx, e)
            
            if end_search > start_search:
                n_active_pixels = end_search - start_search
                pixels_global = patch_idx[start_search:end_search]
                pixels_local_ring = pixels_global - s
                
                active_rings.append(r)
                active_ring_indices_global.append(r["global_idx"])
                u_indices.append(np.arange(start_search, end_search, dtype=np.int64))
                local_ring_idx = len(active_rings) - 1
                ring_batch_indices.append(np.full(n_active_pixels, local_ring_idx, dtype=np.int64))
                ring_pix_indices.append(pixels_local_ring.astype(np.int64))
        
        self.active_rings = active_rings
        self.active_ring_indices_global = np.asarray(active_ring_indices_global, dtype=np.int64)
        
        if len(u_indices) > 0:
            self.scatter_src = np.concatenate(u_indices)
            self.scatter_dst_ring = np.concatenate(ring_batch_indices)
            self.scatter_dst_pix = np.concatenate(ring_pix_indices)
        else:
            self.scatter_src = np.array([], dtype=np.int64)
            self.scatter_dst_ring = np.array([], dtype=np.int64)
            self.scatter_dst_pix = np.array([], dtype=np.int64)

    def _build_torch_scatter_tables(self):
        self.group_info = {}
        if self.scatter_src.size == 0:
            return
        
        t_src = torch.as_tensor(self.scatter_src, dtype=torch.long, device=self.device)
        t_dst_ring = torch.as_tensor(self.scatter_dst_ring, dtype=torch.long, device=self.device)
        t_dst_pix = torch.as_tensor(self.scatter_dst_pix, dtype=torch.long, device=self.device)
        
        rings_by_length = {}
        for i, r in enumerate(self.active_rings):
            rings_by_length.setdefault(int(r["length"]), []).append(i)
        
        n_active = len(self.active_rings)
        for L, local_rings in rings_by_length.items():
            local_rings_t = torch.as_tensor(local_rings, dtype=torch.long, device=self.device)
            ring_to_pos = torch.full((n_active,), -1, dtype=torch.long, device=self.device)
            ring_to_pos[local_rings_t] = torch.arange(local_rings_t.numel(), device=self.device, dtype=torch.long)
            sel = ring_to_pos[t_dst_ring] >= 0
            sel_idx = torch.nonzero(sel, as_tuple=False).flatten()
            if sel_idx.numel() == 0:
                continue
            self.group_info[int(L)] = {
                "L": int(L),
                "local_rings": local_rings_t,
                "src": t_src[sel_idx],
                "dst_ring": ring_to_pos[t_dst_ring[sel_idx]],
                "dst_pix": t_dst_pix[sel_idx],
                "n_group": int(local_rings_t.numel())
            }

    def comp_tf_localized(self, u_patch):
        """Compute 1D Longitude FFT on active rings & apply phase alignment."""
        if not isinstance(u_patch, torch.Tensor):
            u_patch = self.backend.bk_cast(u_patch)
        else:
            if u_patch.dtype != self.dtype:
                u_patch = u_patch.to(dtype=self.dtype)
            u_patch = u_patch.to(self.device)
        
        if u_patch.dim() == 1:
            u_patch = u_patch.unsqueeze(0)
        
        n_batch = u_patch.shape[0]
        n_active = len(self.active_rings)
        target_dim = 3 * self.nside
        
        ft_im = torch.zeros(
            (n_batch, n_active, target_dim),
            dtype=self.backend.all_cbk_type,
            device=self.device
        )
        
        for L, info in self.group_info.items():
            dense = torch.zeros((n_batch, info["n_group"], L), dtype=self.dtype, device=self.device)
            vals = u_patch.index_select(1, info["src"])
            base = (torch.arange(n_batch, device=self.device, dtype=torch.long) * (info["n_group"] * L)).unsqueeze(1)
            flat_idx = (base + info["dst_ring"].unsqueeze(0) * L + info["dst_pix"].unsqueeze(0)).reshape(-1)
            dense.view(-1).scatter_(0, flat_idx, vals.reshape(-1))
            
            r = torch.fft.rfft(dense, dim=-1)
            full_fft = torch.cat([r, torch.conj(torch.flip(r[..., 1:-1], dims=[-1]))], dim=-1)
            if full_fft.shape[-1] < target_dim + 1:
                full_fft = full_fft.repeat(1, 1, target_dim // full_fft.shape[-1] + 1)
            ft_im.index_copy_(1, info["local_rings"], full_fft[..., :target_dim])
        
        return ft_im * self.active_shift_ph.unsqueeze(0)

    def anafast_localized(self, u_patch, map2_patch=None, bandpowers_binning=None):
        """
        Compute pseudo-Cl using a 2D Phase-Aligned Flat-Sky FFT.
        """
        lmax = self.lmax_compute
        out_dtype = self.dtype
        n_batch = u_patch.shape[0] if u_patch.dim() > 1 else 1
        
        # 1. LONGITUDE FFT (m-modes)
        ft_im = self.comp_tf_localized(u_patch)
        ft_im2 = self.comp_tf_localized(map2_patch) if map2_patch is not None else ft_im
        
        # 2. LATITUDE FFT (ell_y modes)
        # Apply 1D FFT along the active rings axis (dim=1)
        F2D = torch.fft.fft(ft_im, dim=1)
        F2D2 = torch.fft.fft(ft_im2, dim=1) if map2_patch is not None else F2D
        
        # We only need to evaluate up to lmax for the x-axis (m-modes)
        # The column indices exactly match m = 0, 1, 2...
        F2D = F2D[:, :, :lmax + 1]
        F2D2 = F2D2[:, :, :lmax + 1]
        
        # 3. 2D POWER SPECTRUM
        # Optional: scale FFT to approximate spherical integral area
        dOmega = (4.0 * np.pi) / (12.0 * self.nside * self.nside)
        F2D = F2D * dOmega
        F2D2 = F2D2 * dOmega
        
        P2D = torch.real(F2D * torch.conj(F2D2))
        
        # Account for symmetric negative m-modes in real maps
        if lmax > 0:
            P2D[:, :, 1:] *= 2.0
            
        # 4. RADIAL FREQUENCY MAPPING (ell = sqrt(m^2 + ell_y^2))
        N_y = F2D.shape[1]
        
        m_arr = torch.arange(lmax + 1, device=self.device, dtype=torch.float32)
        # Total rings in full sphere = 4 * nside. 
        # fftfreq returns cycles per row. Multiply by 4*nside to get cycles per full sphere (ell).
        ell_y = torch.fft.fftfreq(N_y, device=self.device, dtype=torch.float32) * (4.0 * self.nside)
        
        mesh_ell_y, mesh_m = torch.meshgrid(ell_y, m_arr, indexing='ij')
        
        ell_2d = torch.sqrt(mesh_m**2 + mesh_ell_y**2)
        ell_ints = torch.round(ell_2d.flatten()).to(torch.long)
        
        # Mask out values beyond lmax
        valid = ell_ints <= lmax
        ell_ints_valid = ell_ints[valid]
        
        # 5. ISOTROPIC BINNING
        cl_out = torch.zeros((n_batch, lmax + 1), dtype=out_dtype, device=self.device)
        
        for b in range(n_batch):
            P_valid = P2D[b].flatten()[valid]
            
            # Sum power in each ell bin
            binned_power = torch.bincount(ell_ints_valid, weights=P_valid, minlength=lmax+1)[:lmax+1]
            
            # Count the number of modes that fell into each ell bin
            counts = torch.bincount(ell_ints_valid, minlength=lmax+1)[:lmax+1]
            
            # Average the power (avoid division by zero)
            cl_out[b] = torch.where(counts > 0, binned_power / counts, torch.zeros_like(binned_power))

        # 6. APPLY BANDPOWER BINNING (if requested)
        if bandpowers_binning is not None:
            if not isinstance(bandpowers_binning, torch.Tensor):
                bandpowers_binning = torch.as_tensor(bandpowers_binning, device=self.device, dtype=out_dtype)
            else:
                bandpowers_binning = bandpowers_binning.to(self.device, dtype=out_dtype)
            # Apply binning matrix: [B, lmax+1] x [lmax+1, n_bins]
            cl_out = torch.matmul(cl_out, bandpowers_binning.T)
            
        return cl_out