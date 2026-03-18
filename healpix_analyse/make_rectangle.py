from cdshealpix import to_ring
import healpix_geo
import numpy as np

def make_healpix_rectangle_from_lonlat(bbox, level, ellipsoid):
    """bbox = (lon_min, lat_min, lon_max, lat_max)"""

    cell_ids, _, _ = healpix_geo.nested.zone_coverage(bbox=bbox,
                                                      depth=level, 
                                                      ellipsoid=ellipsoid)
    ring_cell_ids = to_ring(cell_ids, depth=level)
    lon_, lat_ = healpix_geo.ring.healpix_to_lonlat(ring_cell_ids, level, ellipsoid=ellipsoid)


    ring_ordering = ring_cell_ids.argsort() # TODO: optimize knowing that it is already sorted
    
    # reorder ring_cell_ids, lat_, lon_ according to ring ordering
    ring_cell_ids = ring_cell_ids[ring_ordering]
    lat_ = lat_[ring_ordering]
    lon_ = (lon_ - np.min(lon_))[ring_ordering]

    # TODO: optimize knowing that it is already sorted
    theta_uniq, idx_ring = np.unique(lat_, return_inverse=True)

    n_rings = len(theta_uniq)
    N_k = np.bincount(idx_ring)
    N_max = int(np.max(N_k))

    cell_ids_2D_array = np.zeros((n_rings, N_max), dtype=ring_cell_ids.dtype) - 1 # initialize with -1 to indicate empty cells

    for k in range(n_rings):
        idx = np.where(idx_ring==k)[0]
        ring_k_cell_ids = ring_cell_ids[idx]

        append_size = N_max - ring_k_cell_ids.shape[0]

        if append_size > 0:
            eastern_append = ring_k_cell_ids[-1] + 1 + np.arange(append_size - append_size//2)
            western_append = ring_k_cell_ids[0] - 1 - np.arange(append_size//2)

            # TODO: modulo to wrap around the ring!!

            ring_k_cell_ids = np.concatenate([western_append, ring_k_cell_ids, eastern_append])

        # store output for ring k
        cell_ids_2D_array[k] = ring_k_cell_ids

    return cell_ids_2D_array