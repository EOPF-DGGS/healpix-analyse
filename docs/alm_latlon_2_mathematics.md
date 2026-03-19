# `alm_latlon` — Part 2: Mathematical foundations

> **Module** `healpix_analyse.alm_latlon`  
> **Parts** [1 · Quickstart](alm_latlon_1_quickstart.md) · [2 · Mathematics](#) · [3 · API reference](alm_latlon_3_api.md)

---

## 1. Spherical harmonic decomposition

A square-integrable function on the sphere $f(\theta, \phi)$ can be decomposed
as:

$$
f(\theta, \phi) = \sum_{\ell=0}^{\infty} \sum_{m=-\ell}^{\ell}
a_{\ell m} \, Y_{\ell m}(\theta, \phi)
$$

where $\theta \in [0, \pi]$ is the **colatitude** (measured from the North pole)
and $\phi \in [0, 2\pi]$ is the **longitude**.

### Normalised spherical harmonics

The module uses the **orthonormal** convention, identical to `healpy`:

$$
Y_{\ell m}(\theta, \phi) =
\sqrt{\frac{2\ell+1}{4\pi}} \, \tilde{P}_{\ell m}(\cos\theta) \, e^{im\phi}
$$

where $\tilde{P}_{\ell m}$ are the **normalised associated Legendre polynomials**,
satisfying:

$$
\int_{-1}^{1} \tilde{P}_{\ell m}(x) \, \tilde{P}_{\ell' m}(x) \, dx
= \frac{2\delta_{\ell\ell'}}{2\ell+1}
$$

The full orthonormality relation is:

$$
\int Y_{\ell m}(\hat{n}) \, Y_{\ell' m'}^*(\hat{n}) \, d\hat{n} = \delta_{\ell\ell'} \delta_{mm'}
$$

### Analysis: map → alm

The harmonic coefficients are obtained by:

$$
a_{\ell m} = \int_0^{2\pi} \int_0^{\pi}
f(\theta, \phi) \, Y_{\ell m}^*(\theta, \phi) \, \sin\theta \, d\theta \, d\phi
$$

Because $f$ is real-valued, the negative-$m$ coefficients satisfy
$a_{\ell,-m} = (-1)^m a_{\ell m}^*$, so only $m \geq 0$ needs to be stored.

### Synthesis: alm → map

The inverse transform reconstructs the map:

$$
f(\theta, \phi) = \sum_{\ell=0}^{L_{\max}} \left[
a_{\ell 0} \, Y_{\ell 0}(\theta) +
2 \, \text{Re} \sum_{m=1}^{\ell} a_{\ell m} \, Y_{\ell m}(\theta, \phi)
\right]
$$

### Angular power spectrum

For a statistically isotropic field, all $m$ modes at the same $\ell$ carry equal
power.  The angular power spectrum is:

$$
C_\ell = \frac{1}{2\ell+1} \sum_{m=-\ell}^{\ell} |a_{\ell m}|^2
       = \frac{1}{2\ell+1}
         \left[ |a_{\ell 0}|^2 + 2 \sum_{m=1}^{\ell} |a_{\ell m}|^2 \right]
$$

---

## 2. The two-step algorithm

The exact integral is approximated over a discrete set of $N$ pixels arranged
in $R$ **iso-latitude rings**.  Within each ring $r$, all pixels share the same
colatitude $\theta_r$ but may differ in longitude.  The algorithm exploits this
structure by splitting the integral into two sequential 1-D operations.

### Step 1 — Longitude: per-ring weighted DFT

For ring $r$ with $N_r$ pixels at positions $(\theta_r, \phi_j)$ and weights
$w_j$, the weighted longitude DFT at azimuthal order $m$ is:

$$
\tilde{F}_r(m) = \sum_{j=0}^{N_r - 1} f(\theta_r, \phi_j) \, w_j \, e^{-im\phi_j}
$$

The full $\phi$ integral is thus approximated as:

$$
\int_0^{2\pi} f(\theta_r, \phi) \, w_r^\theta \, e^{-im\phi} \, d\phi
\approx \tilde{F}_r(m)
$$

where the quadrature weight $w_j = w_r^\theta \cdot w_j^\phi$ combines the
colatitude weight $w_r^\theta$ (see Section 3) and the longitude weight
$w_j^\phi$ (see Section 4).

### Step 2 — Colatitude: Legendre projection

The harmonic coefficient at degree $\ell$, order $m$ is then:

$$
a_{\ell m} = \sum_{r=0}^{R-1} \tilde{F}_r(m) \, Y_{\ell m}(\theta_r, 0)
$$

where $Y_{\ell m}(\theta, 0) = \sqrt{(2\ell+1)/4\pi} \, \tilde{P}_{\ell m}(\cos\theta)$
is the $\phi=0$ slice of the spherical harmonic (the $\phi$-dependence is already
absorbed into $\tilde{F}_r$).

### Why this factorisation is exact for ring grids

The key identity is:

$$
a_{\ell m} = \int_0^{\pi} \left[
\underbrace{\int_0^{2\pi} f(\theta, \phi) \, e^{-im\phi} \, d\phi}_{\text{Step 1}}
\right] Y_{\ell m}(\theta, 0) \, \sin\theta \, d\theta
\xrightarrow{\text{Step 2}} \sum_r \tilde{F}_r(m) \, Y_{\ell m}(\theta_r, 0) \, w_r^\theta
$$

The factorisation is algebraically exact (up to the accuracy of the quadrature
rules chosen for the two integrals).

---

## 3. Quadrature in colatitude

The integral $\int_0^\pi g(\theta) \sin\theta \, d\theta$ is approximated by:

$$
\int_0^\pi g(\theta) \sin\theta \, d\theta \approx \sum_{r=0}^{R-1} g(\theta_r) \, w_r^\theta
$$

Three quadrature rules are implemented, each optimal for a specific family of grids.

### 3.1 Trapezoidal rule (`quadrature="trapeze"`)

$$
w_r^\theta = \sin(\theta_r) \cdot \Delta\theta_r, \qquad
\Delta\theta_r = \frac{|\theta_{r+1} - \theta_{r-1}|}{2}
$$

with half-interval boundary conditions at $r=0$ and $r=R-1$.

This rule converges at second order in $\Delta\theta$ for smooth integrands and
is appropriate for **regularly-spaced** grids.

### 3.2 Gauss-Legendre quadrature (`quadrature="gauss_legendre"`)

After the substitution $x = \cos\theta$, the integral becomes:

$$
\int_0^\pi g(\theta) \sin\theta \, d\theta
= \int_{-1}^{1} g(\arccos x) \, dx
\approx \sum_{r=0}^{R-1} g(\arccos x_r) \, w_r^{\text{GL}}
$$

where $x_r$ are the $R$ zeros of the Legendre polynomial $P_R(x)$ and $w_r^{\text{GL}}$
are the corresponding Gauss-Legendre weights (computed by
`numpy.polynomial.legendre.leggauss(R)`).

**Exactness:** The GL quadrature is exact for polynomials in $x$ of degree up to
$2R - 1$.  Since $Y_{\ell m}(\theta, 0) \propto \tilde{P}_{\ell m}(\cos\theta)$ is
a polynomial in $x = \cos\theta$ of degree $\ell$, the Legendre projection is exact
for all $\ell \leq 2R - 1$, i.e. up to $\ell_{\max} \approx 2R - 1$.

**When to use:** The ERA5, ECMWF IFS, ARPEGE, and similar NWP models output
data on a **Gaussian grid** where latitudes are exactly the GL nodes.  Using the
trapezoidal rule on such a grid introduces a systematic error in the Legendre
projection, visible as regular oscillations in $C_\ell$ at high $\ell$.  Only the
GL weights suppress this artefact.

> **Diagnostic:** A `UserWarning` is emitted if the provided colatitudes deviate
> from the expected GL nodes by more than $10^{-6}$ radians.

### 3.3 Equal-area weights (`quadrature="equal_area"`)

$$
w_i = \frac{4\pi}{N_{\text{total}}} \quad \text{for all pixels } i
$$

This is the correct weight when every pixel covers the same solid angle, as is
the case for the **HEALPix** pixelisation.

---

## 4. Quadrature in longitude

Within ring $r$, the longitude integral $\int_0^{2\pi} h(\phi) d\phi$ is
approximated by summing over the $N_r$ pixels at $\phi_{j}$.

**Uniform ring** ($\phi_j = \phi_0 + j \cdot 2\pi/N_r$): the weights are
$w_j^\phi = 2\pi / N_r$ (rectangle rule, exact for harmonics up to $m = N_r/2 - 1$).

**Irregular ring**: the wrapped-trapezoidal rule is used:
$w_j^\phi = (\Delta\phi_{j-1} + \Delta\phi_j)/2$ where $\Delta\phi_j$ are the
angular gaps between consecutive sorted longitudes (with wrap-around).

---

## 5. FFT acceleration for uniform rings

For a uniform ring with $N_r$ pixels and starting phase $\phi_0$, the DFT

$$
\tilde{F}_r(m) = \frac{2\pi}{N_r} \sum_{j=0}^{N_r-1}
f_j \, e^{-im\phi_j}
= \frac{2\pi}{N_r} \, e^{-im\phi_0}
\underbrace{\sum_{j=0}^{N_r-1} f_j \, e^{-i 2\pi m j / N_r}}_{\text{DFT at frequency } m}
$$

is the standard $N_r$-point DFT evaluated at frequency $m$, multiplied by a
**phase shift** $e^{-im\phi_0}$ that accounts for the non-zero starting longitude.

In code this is implemented as:

1. Sort the $N_r$ pixel values by ascending longitude.
2. Compute `torch.fft.rfft` on the sorted values (exploiting real symmetry).
3. Reconstruct the full complex spectrum with `_rfft2fft`.
4. **Tile** the spectrum if $N_r < \ell_{\max} + 1$ (aliasing by periodicity):
   the DFT at frequency $m$ equals the DFT at $m \bmod N_r$, so repeating the
   spectrum in blocks of $N_r$ gives the correct aliased value at every $m$.
5. Multiply by the phase vector $e^{-im\phi_0}$ for $m = 0, \ldots, \ell_{\max}$.

> **Tiling note:** Step 4 uses `.repeat()`, not `.repeat_interleave()`.
> `.repeat([1, k])` tiles the entire spectrum $k$ times: $[F_0, F_1, \ldots, F_{N-1},
> F_0, F_1, \ldots]$, which corresponds to the correct periodic aliasing.
> `.repeat_interleave(k)` would instead produce $[F_0, F_0, \ldots, F_1, F_1, \ldots]$,
> which is wrong.

---

## 6. Normalised associated Legendre polynomials — recurrence

The module computes $\tilde{P}_{\ell m}(\cos\theta)$ using the standard
three-term recurrence, evaluated in log-space to avoid numerical overflow at
large $\ell$.

### Seed ($\ell = m$)

$$
\tilde{P}_{mm}(x) = (-1)^m \sqrt{\frac{(2m)!}{4\pi \cdot (2m-1)!!^2}} \, (1-x^2)^{m/2}
$$

The prefactor is computed in log-space:

$$
\log \tilde{P}_{mm} = \log(2m-1)!! - \tfrac{1}{2}\sum_{k=1}^{2m} \log k + m \log(1-x^2)
$$

### First step ($\ell = m+1$)

$$
\tilde{P}_{m+1,m}(x) = x \sqrt{2m+1} \, \tilde{P}_{mm}(x)
$$

### Recurrence ($\ell \geq m+2$)

$$
\tilde{P}_{\ell m}(x) =
\frac{(2\ell-1) \, x \, \tilde{P}_{\ell-1,m}(x) - (\ell+m-1) \, \tilde{P}_{\ell-2,m}(x)}
{\ell - m}
$$

with normalisation ratio update:

$$
\log r_\ell = \log r_{\ell-1} + \tfrac{1}{2}\log(\ell-m) - \tfrac{1}{2}\log(\ell+m)
$$

### Overflow guard

When any element of the current recurrence row exceeds `limit_range` ($= 10^{10}$
by default), both the current and previous rows are rescaled by `1/limit_range`
and `log_limit` is added to the corresponding log-ratio.  This keeps the
mantissa in a safe numerical range without losing relative precision.

### Output normalisation

The raw recurrence produces values scaled by $\sqrt{4\pi}$.  In `map2alm_latlon`,
the factor $\sqrt{(2\ell+1)/4\pi}$ is applied to recover the true $Y_{\ell m}(\theta, 0)$:

$$ 
Y_{\ell m}(\theta, 0) = \underbrace{\frac{\sqrt{2\ell+1}}{4\pi}}_{\text{applied in code}}
\times \underbrace{\sqrt{4\pi} \cdot \tilde{P}_{\ell m}(\cos\theta)}_{\text{output of compute legendre}}
$$

---

## 7. Memory layout of `alm`

The output tensor `alm` is a flat complex vector.  The layout groups coefficients
by azimuthal order $m$, then by degree $\ell$ within each group:

$$
\underbrace{a_{00}, a_{10}, a_{20}, \ldots, a_{L0}}_{m=0,\; \ell=0\ldots L},\;
\underbrace{a_{11}, a_{21}, \ldots, a_{L1}}_{m=1,\; \ell=1\ldots L},\;
\ldots,\;
\underbrace{a_{LL}}_{m=L}
$$

The total number of coefficients is:

$$
n_{\text{alm}} = \sum_{m=0}^{L} (L - m + 1) = \frac{(L+1)(L+2)}{2}
$$

To extract the coefficient $a_{\ell m}$ from the flat vector:

```python
def alm_index(ell, m, lmax):
    """Index of a_lm in the flat alm vector."""
    offset = sum(lmax - mp + 1 for mp in range(m))   # start of m-block
    return offset + (ell - m)                          # position within block
```

This layout is identical to the convention used by `healpy.map2alm`.

---

## 8. Power spectrum normalisation

Starting from the flat `alm` vector, `anafast_latlon` accumulates:

$$
C_\ell = \frac{1}{2\ell+1}
\left[
\sum_{\ell'=0}^{L} |a_{\ell' 0}|^2 \cdot \mathbb{1}[\ell'=\ell]
+ 2 \sum_{m=1}^{L} \sum_{\ell'=m}^{L} |a_{\ell' m}|^2 \cdot \mathbb{1}[\ell'=\ell]
\right]
$$

The factor 2 for $m > 0$ accounts for the conjugate mode $m < 0$ (not stored
explicitly because the map is real-valued).

---

## 9. Accuracy and limitations

### Accuracy vs `lmax`

The transform is accurate as long as:
- **Longitude:** $\ell_{\max} \leq N_r^{\min}/2$ (Nyquist for the ring with the
  fewest pixels).  Rings with fewer pixels contribute aliased modes.
- **Colatitude:** depends on the quadrature rule:
  - `trapeze`: error $O(\Delta\theta^2)$, converges for smooth fields.
  - `gauss_legendre`: exact for $\ell \leq 2R - 1$.
  - `equal_area`: pixel noise below $\ell \sim 2 N_{\text{side}}$ for HEALPix.

### Precision

All Legendre computations are done in `float64`.  The FFT path uses
`torch.complex128`.  The output `alm` and `Cl` are `complex128` and `float64`
respectively.

### Current limitations

- Only **scalar (spin-0)** fields.  Spin-weighted harmonics are not yet implemented.
- `alm2map_latlon` uses a direct DFT in longitude ($O(N_r \cdot \ell_{\max})$ per ring);
  an IFFT optimisation for uniform rings is not yet implemented.
- `lmax` must satisfy `lmax <= 3 * nside - 1` for HEALPix grids to match
  `healpy.anafast`.
