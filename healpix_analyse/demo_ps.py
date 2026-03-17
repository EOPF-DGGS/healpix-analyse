import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import torch
import time
import sys
import scipy.optimize as opt
import foscat.scat_cov as sc
import foscat.Synthesis as synthe
import foscat.alm as foscat_alm
import gc

# Make sure LocalizedFlatSkyAlm.py is in the same directory
from LocalizedFlatSkyAlm import LocalizedFlatSkyAlm

# ==============================================================================
# 1. SETUP PARAMETERS & EXTRACT DISK PATCH WITH HEALPY
# ==============================================================================
level = 12
nside = 2**level
lon_center, lat_center = 15.0, 15.0  # Center of our patch in degrees
radius_deg = 2.0                     # Radius of the disk in degrees

print(f"Generating disk patch at N_side={nside}...")
# Convert lon/lat to a 3D vector for healpy
vec = hp.ang2vec(lon_center, lat_center, lonlat=True)

# Query the pixels inside the disk. 
# nest=False ensures we get RING indices, which is what we want!
idx_ring = hp.query_disc(nside, vec, radius=np.radians(radius_deg), nest=False)

# ==============================================================================
# 2. GENERATE SYNTHETIC DATA STRICTLY ON THE PATCH
# ==============================================================================
# Get the actual coordinates of just our patch pixels
lon_patch, lat_patch = hp.pix2ang(nside, idx_ring, lonlat=True)

# Generate some synthetic data: a spatial wave pattern + random Gaussian noise
# (This ensures our power spectrum has an interesting shape to look at)
print("Generating random + synthetic data...")
data = np.cos(100 * np.radians(lon_patch)) * np.sin(100 * np.radians(lat_patch))
data += 0.5 * np.random.randn(len(idx_ring))

# ==============================================================================
# 3. GEOMETRY ALIGNMENT FOR THE ESTIMATOR
# ==============================================================================
# The Flat-Sky estimator requires the RING indices to be strictly sorted
sorter = np.argsort(idx_ring)
idx_ring_sorted = idx_ring[sorter]

# Apply the EXACT same sorting to our data array
data_ring_sorted = data[sorter]

# ==============================================================================
# 4. BACKEND & ESTIMATOR INITIALIZATION
# ==============================================================================
lmax_full = 3 * nside
lmax_compute = 1500  # Computing up to ell=1500

print(f"Initializing LocalizedFlatSkyAlm...")

f2 = sc.funct(BACKEND="torch", KERNELSZ=5, all_type="float32")

alm_loc = LocalizedFlatSkyAlm(
    nside=nside,
    patch_idx_ring=idx_ring_sorted,
    backend=f2.backend, 
    lmax=lmax_full,
    lmax_compute=lmax_compute
)

# ==============================================================================
# 5. COMPUTE POWER SPECTRUM
# ==============================================================================
# Push the sorted data to a PyTorch tensor (adding a batch dimension)
t_patch = torch.tensor(data_ring_sorted, dtype=torch.float32, device=alm_loc.device).unsqueeze(0)
print("t_patch.shape: ", t_patch.shape)

print(f"Running 2D FFT pseudo-Cl estimator on {len(idx_ring_sorted)} pixels...")
t0 = time.time()
cl_loc = alm_loc.anafast_localized(t_patch)
print(f"Computation finished in {time.time() - t0:.4f} seconds.")

# Bring result back to CPU numpy array
cl_loc_np = cl_loc.detach().cpu().numpy()[0]

# ==============================================================================
# 6. PLOT RESULTS (POWER SPECTRUM)
# ==============================================================================
ell_arr = np.arange(len(cl_loc_np))

plt.figure(figsize=(10, 5))
# Skip the first few ell bins (monopole/dipole)
plt.plot(ell_arr[2:], cl_loc_np[2:], label="Localized $C_\ell$", color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Multipole Moment ($\ell$)')
plt.ylabel('Power ($C_\ell$)')
plt.title(f'Power Spectrum of healpy Disk Patch (N_side={nside}, Radius={radius_deg}°)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ==============================================================================
# 7. PLOT THE MAP AND PATCH (VISUALIZATION)
# ==============================================================================
print("Preparing full sky map for visualization...")
# Initialize a full sky array with the UNSEEN value (so empty space is gray/blank)
npix_full = hp.nside2npix(nside)
full_map = np.full(npix_full, hp.UNSEEN, dtype=np.float32)

# Insert our generated patch data into the full sky array
full_map[idx_ring_sorted] = data_ring_sorted

# 7A. Mollweide View (Global context - patch will be a small dot)
print("Rendering Mollview...")
hp.mollview(
    full_map, 
    title=f"Global Mollweide View (Patch at lon={lon_center}°, lat={lat_center}°)", 
    cmap='viridis',
    cbar=True
)
hp.graticule()
plt.show()

# 7B. Gnomonic View (Zoomed in on the patch to see the synthetic data)
print("Rendering Gnomview (Zoomed)...")
hp.gnomview(
    full_map, 
    rot=[lon_center, lat_center], # Center the camera on our patch
    reso=0.5,                     # Resolution in arcmin per pixel for the zoom
    xsize=800,                    # Image size 
    title=f"Zoomed Gnomonic View of the 2° Patch",
    cmap='viridis',
    cbar=True
)
hp.graticule()
plt.show()

# Free up the large full_map array from memory
del full_map
gc.collect()