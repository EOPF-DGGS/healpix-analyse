import numpy as np
import torch
import torch.nn as nn
import healpy as hp


def ring_nphi(iring: int, nside: int) -> int:
    # iring: 1..(4*nside-1)
    if iring < nside:
        return 4 * iring
    elif iring <= 3 * nside:
        return 4 * nside
    else:
        return 4 * (4 * nside - iring)


def _make_phase(delta: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    delta: (...,) radians
    k: (F,) integer modes 0..F-1
    returns: (..., F) complex phase exp(1j*k*delta)
    """
    ang = delta[..., None] * k[None, ...]
    # build complex with same float precision as ang
    return torch.complex(torch.cos(ang), torch.sin(ang))


class HealpixNestToRingGrid(nn.Module):
    """
    HEALPix NESTED [..., K] -> ring-grid [..., H, W]
    H = 4*nside - 1 (anneaux), W constant (ex: 8*nside)
    Interpolation FFT uniquement en longitude + correction d'offset par anneau.
    Option: flip longitude (astro) par défaut pour matcher healpy flip='astro'.
    """
    def __init__(self, nside: int, W: int = None, phi_ref: float = 0.0, flip_lon: bool = True):
        super().__init__()
        self.nside = int(nside)
        self.H = 4 * self.nside - 1
        self.K = 12 * self.nside * self.nside
        self.W = int(8 * self.nside if W is None else W)
        self.phi_ref = float(phi_ref)
        self.flip_lon = bool(flip_lon)

        # --- mapping NESTED -> RING (exact, permutation) ---
        # m_ring[i] = m_nest[n2r_idx[i]]
        n2r = hp.reorder(np.arange(self.K), n2r=True).astype(np.int64)
        self.register_buffer("n2r_idx", torch.from_numpy(n2r), persistent=True)

        # --- anneaux : starts, nphi, ring_ids par nphi ---
        starts = []
        nphis = []
        p = 0
        for iring in range(1, self.H + 1):
            nphi = ring_nphi(iring, self.nside)
            starts.append(p)
            nphis.append(nphi)
            p += nphi
        assert p == self.K

        self.register_buffer("ring_start", torch.tensor(starts, dtype=torch.long), persistent=True)
        self.register_buffer("ring_nphi", torch.tensor(nphis, dtype=torch.long), persistent=True)

        # --- phi0 pour chaque anneau (longitude du 1er pixel) ---
        # ipix0 = start en ordre RING
        phi0 = np.empty(self.H, dtype=np.float64)
        for r in range(self.H):
            ipix0 = int(starts[r])
            _, ph = hp.pix2ang(self.nside, ipix0, nest=False)
            phi0[r] = ph
        self.register_buffer("phi0", torch.from_numpy(phi0).to(torch.float64), persistent=True)

        # --- groupes par nphi pour batcher les FFT (une FFT par groupe) ---
        uniq_nphi = sorted(set(nphis))
        self._group_ns = uniq_nphi  # python list (petit, stable)

        # Buffers par groupe : ring_ids, gather_idx (Rg,N)
        self._group_names = []
        for gi, N in enumerate(uniq_nphi):
            ring_ids = [r for r in range(self.H) if nphis[r] == N]
            ring_ids_t = torch.tensor(ring_ids, dtype=torch.long)

            # indices des pixels RING pour gather : (Rg, N)
            idx = []
            for r in ring_ids:
                s = starts[r]
                idx.append(np.arange(s, s + N, dtype=np.int64))
            idx_t = torch.from_numpy(np.stack(idx, axis=0))

            self.register_buffer(f"g_ring_ids_{gi}", ring_ids_t, persistent=True)
            self.register_buffer(f"g_gather_idx_{gi}", idx_t, persistent=True)
            self._group_names.append((f"g_ring_ids_{gi}", f"g_gather_idx_{gi}", N))

        # --- phases FFT (dépend de N) : on les calcule à la volée par groupe, mais sans healpy ---
        # (on garde juste phi0 en buffer)

    def forward(self, m_nest: torch.Tensor) -> torch.Tensor:
        """
        m_nest: [..., K] float
        returns: [..., H, W] float
        """
        assert m_nest.shape[-1] == self.K, f"attendu K={self.K}, reçu {m_nest.shape[-1]}"
        device = m_nest.device
        dtype = m_nest.dtype

        # flatten batch dims
        batch_shape = m_nest.shape[:-1]
        B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1
        x = m_nest.reshape(B, self.K)

        # NESTED -> RING (gather)
        x_ring = x.index_select(dim=1, index=self.n2r_idx.to(device))

        out = torch.empty((B, self.H, self.W), device=device, dtype=dtype)

        # constante torch pour phases
        phi_ref = torch.tensor(self.phi_ref, device=device, dtype=torch.float64)

        for gi, (name_rids, name_idx, N) in enumerate(self._group_names):
            ring_ids = getattr(self, name_rids).to(device)        # (Rg,)
            gather_idx = getattr(self, name_idx).to(device)       # (Rg,N)

            # gather: (B, Rg, N)
            xg = x_ring[:, gather_idx]  # advanced indexing

            # rFFT sur N
            X = torch.fft.rfft(xg.to(torch.float64), dim=-1)

            # phase exp(i*k*(phi_ref - phi0_ring))
            phi0 = self.phi0.index_select(0, ring_ids).to(device)   # (Rg,)
            delta = (phi_ref - phi0)                                # (Rg,)
            k = torch.arange(0, N // 2 + 1, device=device, dtype=torch.float64)
            phase = _make_phase(delta, k)                           # (Rg, F_N)
            X = X * phase.unsqueeze(0)                              # (B,Rg,F_N)

            # copie modes vers W//2+1
            FW = self.W // 2 + 1
            FN = N // 2 + 1
            L = min(FN, FW)
            Y = torch.zeros((B, ring_ids.numel(), FW), device=device, dtype=X.dtype)
            Y[..., :L] = X[..., :L]

            yg = torch.fft.irfft(Y, n=self.W, dim=-1)               # (B,Rg,W)
            yg = yg * (self.W / N)                                  # scaling numpy-compatible

            out[:, ring_ids, :] = yg.to(dtype)

        if self.flip_lon:
            out = torch.flip(out, dims=[-1])

        return out.reshape(*batch_shape, self.H, self.W)


class RingGridToHealpixNest(nn.Module):
    """
    ring-grid [..., H, W] -> HEALPix NESTED [..., K]
    Inverse (au sens FFT) + permutation RING->NESTED.
    """
    def __init__(self, nside: int, W: int = None, phi_ref: float = 0.0, flip_lon: bool = True):
        super().__init__()
        self.nside = int(nside)
        self.H = 4 * self.nside - 1
        self.K = 12 * self.nside * self.nside
        self.W = int(8 * self.nside if W is None else W)
        self.phi_ref = float(phi_ref)
        self.flip_lon = bool(flip_lon)

        # mapping RING -> NESTED (exact)
        # m_nest[j] = m_ring[r2n_idx[j]] ? (attention aux conventions)
        # On veut: m_nest = hp.reorder(m_ring, r2n=True)
        # Donc: m_nest[i] = m_ring[r2n_map[i]]
        r2n = hp.reorder(np.arange(self.K), r2n=True).astype(np.int64)
        self.register_buffer("r2n_idx", torch.from_numpy(r2n), persistent=True)

        # anneaux (mêmes infos que l'autre classe)
        starts = []
        nphis = []
        p = 0
        for iring in range(1, self.H + 1):
            nphi = ring_nphi(iring, self.nside)
            starts.append(p)
            nphis.append(nphi)
            p += nphi
        assert p == self.K

        # phi0
        phi0 = np.empty(self.H, dtype=np.float64)
        for r in range(self.H):
            ipix0 = int(starts[r])
            _, ph = hp.pix2ang(self.nside, ipix0, nest=False)
            phi0[r] = ph

        self.register_buffer("phi0", torch.from_numpy(phi0).to(torch.float64), persistent=True)

        # groupes par nphi + indices gather (Rg,N)
        uniq_nphi = sorted(set(nphis))
        self._group_names = []
        for gi, N in enumerate(uniq_nphi):
            ring_ids = [r for r in range(self.H) if nphis[r] == N]
            ring_ids_t = torch.tensor(ring_ids, dtype=torch.long)

            idx = []
            for r in ring_ids:
                s = starts[r]
                idx.append(np.arange(s, s + N, dtype=np.int64))
            idx_t = torch.from_numpy(np.stack(idx, axis=0))

            self.register_buffer(f"g_ring_ids_{gi}", ring_ids_t, persistent=True)
            self.register_buffer(f"g_scatter_idx_{gi}", idx_t, persistent=True)
            self._group_names.append((f"g_ring_ids_{gi}", f"g_scatter_idx_{gi}", N))

        # phase sur source W (dépend uniquement de delta_inv et k sur W)
        # on la calcule par groupe dans forward (cheap), car delta_inv dépend du ring_id.

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        grid: [..., H, W]
        returns: [..., K] NESTED
        """
        assert grid.shape[-2] == self.H and grid.shape[-1] == self.W, \
            f"attendu (H,W)=({self.H},{self.W}), reçu ({grid.shape[-2]},{grid.shape[-1]})"
        device = grid.device
        dtype = grid.dtype

        batch_shape = grid.shape[:-2]
        B = int(torch.tensor(batch_shape).prod().item()) if len(batch_shape) > 0 else 1
        g = grid.reshape(B, self.H, self.W)

        if self.flip_lon:
            g = torch.flip(g, dims=[-1])

        # m_ring reconstruit
        m_ring = torch.empty((B, self.K), device=device, dtype=dtype)

        phi_ref = torch.tensor(self.phi_ref, device=device, dtype=torch.float64)

        # FFT sur W une fois par groupe (batched)
        kW = torch.arange(0, self.W // 2 + 1, device=device, dtype=torch.float64)

        for gi, (name_rids, name_idx, N) in enumerate(self._group_names):
            ring_ids = getattr(self, name_rids).to(device)      # (Rg,)
            scatter_idx = getattr(self, name_idx).to(device)    # (Rg,N)

            rows = g[:, ring_ids, :]                            # (B,Rg,W)
            XW = torch.fft.rfft(rows.to(torch.float64), dim=-1) # (B,Rg,FW)

            phi0 = self.phi0.index_select(0, ring_ids).to(device)  # (Rg,)
            delta_inv = (phi0 - phi_ref)                           # (Rg,)
            phaseW = _make_phase(delta_inv, kW)                    # (Rg,FW)
            XW = XW * phaseW.unsqueeze(0)

            FN = N // 2 + 1
            FW = self.W // 2 + 1
            L = min(FN, FW)

            Y = torch.zeros((B, ring_ids.numel(), FN), device=device, dtype=XW.dtype)
            Y[..., :L] = XW[..., :L]

            x_rec = torch.fft.irfft(Y, n=N, dim=-1)              # (B,Rg,N)
            x_rec = x_rec * (N / self.W)

            # scatter vers les bons indices RING
            idx_flat = scatter_idx.reshape(-1)                   # (Rg*N,)
            val_flat = x_rec.to(dtype).reshape(B, -1)            # (B, Rg*N)
            m_ring.scatter_(dim=1, index=idx_flat.unsqueeze(0).expand(B, -1), src=val_flat)

        # RING -> NESTED (gather)
        m_nest = m_ring.index_select(dim=1, index=self.r2n_idx.to(device))
        return m_nest.reshape(*batch_shape, self.K)

