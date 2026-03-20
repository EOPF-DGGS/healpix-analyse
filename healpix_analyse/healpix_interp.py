"""
get_interp_val — interpolation bilinéaire sur une grille HEALPix (NESTED)
via healpix-geo pour les conversions de coordonnées.

Équivalent à healpy.get_interp_val, avec support des ellipsoïdes de référence
(ex. WGS84) grâce au package healpix-geo.

Dépendances :
    pip install healpix-geo  # inclut cdshealpix comme dépendance
    pip install numpy

Auteurs : Claude (Anthropic)
"""

import numpy as np
from healpix_geo.nested import healpix_to_lonlat, kth_neighbourhood, lonlat_to_healpix


# ---------------------------------------------------------------------------
# Utilitaires géométriques
# ---------------------------------------------------------------------------

def _gnomonic_project(
    lon_ref_deg: np.ndarray,
    lat_ref_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projection gnomonique (plan tangent) de points (lon, lat) par rapport à un
    point de référence.

    Transforme chaque point (lon_deg[i], lat_deg[i]) en coordonnées (x, y) dans
    le plan tangent centré en (lon_ref_deg, lat_ref_deg). Le point de référence
    se projette à l'origine (0, 0).

    Parameters
    ----------
    lon_ref_deg, lat_ref_deg : array_like, shape broadcastable avec lon_deg/lat_deg
        Point(s) de référence en degrés.
    lon_deg, lat_deg : array_like
        Points à projeter, en degrés.

    Returns
    -------
    x, y : np.ndarray
        Coordonnées en radians dans le plan tangent.
    """
    lon_ref = np.radians(lon_ref_deg)
    lat_ref = np.radians(lat_ref_deg)
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)

    dlon = lon - lon_ref
    cos_c = (
        np.sin(lat_ref) * np.sin(lat)
        + np.cos(lat_ref) * np.cos(lat) * np.cos(dlon)
    )
    # Eviter la division par zéro pour des points antipodaux (cos_c ≈ -1)
    cos_c = np.where(np.abs(cos_c) < 1e-15, 1e-15, cos_c)

    x = np.cos(lat) * np.sin(dlon) / cos_c
    y = (
        np.cos(lat_ref) * np.sin(lat)
        - np.sin(lat_ref) * np.cos(lat) * np.cos(dlon)
    ) / cos_c
    return x, y


def _bilinear_weights_from_tangent_plane(
    sel_px: np.ndarray,
    sel_py: np.ndarray,
) -> np.ndarray:
    """
    Calcule les poids d'interpolation bilinéaire pour un point requête situé à
    l'origine du plan tangent, étant donné 4 centres de cellules projetés.

    Formule bilinéaire d'area :
        Le poids de chaque coin est proportionnel à l'aire du rectangle opposé,
        ce qui donne, pour un quad approximativement rectangulaire :

            w_i = u_contrib_i * v_contrib_i

        où :
            u = ax / (ax + bx)   fraction vers l'est  (ax = |px| côté ouest,
                                                        bx = |px| côté est)
            v = ay / (ay + by)   fraction vers le nord

        Et pour chaque cellule i :
            u_contrib = u    si sel_px[i] >= 0  (côté est)
                       (1-u) si sel_px[i] <  0  (côté ouest)
            v_contrib = v    si sel_py[i] >= 0  (côté nord)
                       (1-v) si sel_py[i] <  0  (côté sud)

    Parameters
    ----------
    sel_px, sel_py : np.ndarray, shape (N, 4)
        Coordonnées des 4 cellules sélectionnées dans le plan tangent.
        La requête est à (0, 0).

    Returns
    -------
    weights : np.ndarray, shape (N, 4)
        Poids bilinéaires, somme = 1 sur axis=1.
    """
    # Bornes de l'emprise : distance max à l'ouest/est/sud/nord
    ax = np.maximum(-sel_px.min(axis=1), 0.0)  # |px| côté ouest (shape N)
    bx = np.maximum(sel_px.max(axis=1), 0.0)   # |px| côté est
    ay = np.maximum(-sel_py.min(axis=1), 0.0)  # |py| côté sud
    by = np.maximum(sel_py.max(axis=1), 0.0)   # |py| côté nord

    total_x = ax + bx
    total_y = ay + by

    # Fractions dans [0, 1]  (0.5 si dégénéré, i.e. tous les points d'un côté)
    u = np.where(total_x > 0, ax / total_x, 0.5)[:, None]  # (N, 1)
    v = np.where(total_y > 0, ay / total_y, 0.5)[:, None]

    # Contribution de chaque cellule selon son quadrant
    u_contrib = np.where(sel_px >= 0, u, 1.0 - u)   # (N, 4)
    v_contrib = np.where(sel_py >= 0, v, 1.0 - v)

    weights = u_contrib * v_contrib  # (N, 4)

    # Normalisation de sécurité (doit déjà sommer à 1)
    w_sum = weights.sum(axis=1, keepdims=True)
    weights = np.where(w_sum > 0, weights / w_sum, 0.25)

    return weights


# ---------------------------------------------------------------------------
# API principale
# ---------------------------------------------------------------------------

def get_interp_weights(
    lon: np.ndarray,
    lat: np.ndarray,
    depth: int,
    ellipsoid: str = "sphere",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Renvoie les 4 cellules HEALPix et leurs poids d'interpolation bilinéaire
    pour chaque position (lon, lat).

    Équivalent à healpy.get_interp_weights, mais utilise healpix-geo pour les
    conversions de coordonnées et supporte les ellipsoïdes (ex. WGS84).

    Algorithm
    ---------
    1. Trouve la cellule HEALPix contenante (via healpix_geo.nested.lonlat_to_healpix).
    2. Récupère les 9 cellules candidates : la cellule centrale + ses 8 voisines
       immédiates (via kth_neighbourhood, ring=1).
    3. Projette les centres de ces 9 cellules sur le plan tangent en chaque
       point requête (projection gnomonique).
    4. Sélectionne les 4 centres les plus proches de la requête (= à l'origine).
    5. Calcule les poids bilinéaires à partir des positions dans le plan tangent.

    Parameters
    ----------
    lon : np.ndarray, shape (N,)
        Longitudes en degrés.
    lat : np.ndarray, shape (N,)
        Latitudes en degrés.
    depth : int
        Profondeur HEALPix (nside = 2**depth).
    ellipsoid : str, optional
        Ellipsoïde de référence : "sphere" (défaut, identique à healpy) ou
        "WGS84", "GRS80", etc.  Voir la doc de healpix-geo pour la liste.

    Returns
    -------
    pixels : np.ndarray of uint64, shape (N, 4)
        Indices des 4 cellules HEALPix (schéma NESTED).
    weights : np.ndarray of float64, shape (N, 4)
        Poids bilinéaires correspondants. Chaque ligne somme à 1.

    Raises
    ------
    ValueError
        Si lon et lat n'ont pas la même forme.

    Notes
    -----
    Pour l'ellipsoïde "sphere", les résultats sont très proches de ceux de
    healpy.get_interp_weights (schéma NESTED). Les très légères différences
    viennent de l'utilisation du plan tangent local plutôt que du schéma RING
    interne de healpy.

    Pour les ellipsoïdes non-sphériques (ex. WGS84), la conversion lon/lat →
    cellule HEALPix intègre la latitude authalique (voir healpix-geo), ce qui
    n'est pas possible avec healpy.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.shape != lat.shape:
        raise ValueError(
            f"lon et lat doivent avoir la même forme "
            f"(got {lon.shape} vs {lat.shape})"
        )
    N = lon.size
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()

    # ------------------------------------------------------------------
    # 1. Cellule contenante pour chaque point requête
    # ------------------------------------------------------------------
    ipix = lonlat_to_healpix(
        lon_flat, lat_flat, depth, ellipsoid=ellipsoid
    )  # shape (N,), dtype uint64

    # ------------------------------------------------------------------
    # 2. Cellules candidates : cellule centrale + 8 voisines (ring=1)
    #    kth_neighbourhood renvoie (N, 9)
    #    L'élément d'indice 4 est la cellule centrale elle-même.
    #    Les voisins manquants (pôles) sont signalés par -1 ou une valeur
    #    invalide ; on les masque.
    # ------------------------------------------------------------------
    all_cells = kth_neighbourhood(ipix, depth, ring=1)  # (N, 9), int64

    # ------------------------------------------------------------------
    # 3. Centres lon/lat des 9 cellules candidates
    # ------------------------------------------------------------------
    flat_cells = all_cells.ravel()                       # (N*9,)
    valid_mask = flat_cells >= 0                          # (N*9,) bool
    safe_cells = np.where(valid_mask, flat_cells, 0).astype(np.uint64)

    c_lon, c_lat = healpix_to_lonlat(
        safe_cells, depth, ellipsoid=ellipsoid
    )  # (N*9,)

    c_lon = c_lon.reshape(N, 9)
    c_lat = c_lat.reshape(N, 9)
    valid_mask = valid_mask.reshape(N, 9)  # (N, 9)

    # ------------------------------------------------------------------
    # 4. Projection gnomonique : plan tangent centré sur la requête
    #    La requête est à l'origine (0, 0) par définition.
    # ------------------------------------------------------------------
    px, py = _gnomonic_project(
        lon_flat[:, None], lat_flat[:, None],  # (N, 1)
        c_lon, c_lat,                           # (N, 9)
    )  # (N, 9)

    # Distance² au point requête (= à l'origine)
    dist2 = px**2 + py**2
    # Invalider les voisins manquants
    dist2 = np.where(valid_mask, dist2, np.inf)

    # ------------------------------------------------------------------
    # 5. Sélection des 4 centres les plus proches
    # ------------------------------------------------------------------
    idx4 = np.argsort(dist2, axis=1)[:, :4]   # (N, 4)
    i_row = np.arange(N)[:, None]

    sel_cells = all_cells[i_row, idx4]         # (N, 4)
    sel_px = px[i_row, idx4]                   # (N, 4)
    sel_py = py[i_row, idx4]                   # (N, 4)

    # ------------------------------------------------------------------
    # 6. Poids bilinéaires
    # ------------------------------------------------------------------
    weights = _bilinear_weights_from_tangent_plane(sel_px, sel_py)

    return sel_cells.astype(np.uint64), weights


def get_interp_val(
    hpx_map: np.ndarray,
    lon,
    lat,
    depth: int,
    ellipsoid: str = "sphere",
) -> np.ndarray:
    """
    Interpolation bilinéaire d'une carte HEALPix aux coordonnées géographiques.

    Équivalent à healpy.get_interp_val(m, theta, phi, nest=True, lonlat=True)
    mais utilise healpix-geo pour les projections, ce qui permet de travailler
    sur des ellipsoïdes de référence (ex. WGS84).

    Parameters
    ----------
    hpx_map : np.ndarray, shape (12 * 4**depth,)
        Carte HEALPix en ordre NESTED.
    lon : float ou np.ndarray
        Longitude(s) en degrés.
    lat : float ou np.ndarray
        Latitude(s) en degrés.
    depth : int
        Profondeur HEALPix (nside = 2**depth).
    ellipsoid : str, optional
        Ellipsoïde de référence : "sphere" (défaut) ou "WGS84", etc.

    Returns
    -------
    np.ndarray ou float
        Valeurs interpolées. Scalaire si lon/lat sont scalaires, sinon array de
        même forme que lon/lat.

    Examples
    --------
    >>> import numpy as np
    >>> from healpix_interp import get_interp_val
    >>>
    >>> depth = 3
    >>> nside  = 2**depth
    >>> npix   = 12 * nside**2
    >>> hpx_map = np.arange(npix, dtype=float)
    >>>
    >>> # Point unique
    >>> val = get_interp_val(hpx_map, lon=45.0, lat=30.0, depth=depth)
    >>> print(val)
    >>>
    >>> # Grille de points
    >>> lons = np.linspace(0, 360, 50)
    >>> lats = np.linspace(-80, 80, 40)
    >>> lon_grid, lat_grid = np.meshgrid(lons, lats)
    >>> vals = get_interp_val(hpx_map, lon_grid, lat_grid, depth=depth)
    >>> print(vals.shape)  # (40, 50)
    >>>
    >>> # Avec ellipsoïde WGS84
    >>> vals_wgs84 = get_interp_val(
    ...     hpx_map, lon=2.3522, lat=48.8566, depth=depth, ellipsoid="WGS84"
    ... )

    Notes
    -----
    La carte hpx_map doit contenir exactement 12 * 4**depth éléments et être
    en ordre NESTED (le schéma utilisé par healpix-geo).

    Si votre carte est en ordre RING (healpy), convertissez-la d'abord :
        import healpy as hp
        hpx_map_nested = hp.reorder(hpx_map_ring, r2n=True)
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    scalar_input = lon.ndim == 0
    original_shape = lon.shape
    lon = np.atleast_1d(lon).ravel()
    lat = np.atleast_1d(lat).ravel()

    pixels, weights = get_interp_weights(lon, lat, depth, ellipsoid=ellipsoid)
    # hpx_map[pixels] : (N, 4) — valeurs aux 4 cellules
    vals = np.sum(weights * hpx_map[pixels], axis=1)

    if scalar_input:
        return float(vals[0])
    return vals.reshape(original_shape)


# ---------------------------------------------------------------------------
# Test rapide (optionnel)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== Test get_interp_val avec healpix-geo ===\n")
    depth = 5
    nside = 2**depth
    npix = 12 * nside**2
    rng = np.random.default_rng(42)
    hpx_map = rng.standard_normal(npix)

    # Grille de test
    lons = np.linspace(0.1, 359.9, 200)
    lats = np.linspace(-89.0, 89.0, 200)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # --- healpix-geo (sphere) ---
    t0 = time.perf_counter()
    vals_sphere = get_interp_val(hpx_map, lon_grid, lat_grid, depth=depth, ellipsoid="sphere")
    t1 = time.perf_counter()
    print(f"healpix-geo (sphere)  : shape={vals_sphere.shape}  "
          f"min={vals_sphere.min():.4f}  max={vals_sphere.max():.4f}  "
          f"[{(t1-t0)*1000:.1f} ms]")

    # --- healpix-geo (WGS84) ---
    t0 = time.perf_counter()
    vals_wgs84 = get_interp_val(hpx_map, lon_grid, lat_grid, depth=depth, ellipsoid="WGS84")
    t1 = time.perf_counter()
    print(f"healpix-geo (WGS84)   : shape={vals_wgs84.shape}  "
          f"min={vals_wgs84.min():.4f}  max={vals_wgs84.max():.4f}  "
          f"[{(t1-t0)*1000:.1f} ms]")

    # Différence sphere vs WGS84
    diff = np.abs(vals_sphere - vals_wgs84)
    print(f"\nDiff sphere vs WGS84  : mean={diff.mean():.6f}  max={diff.max():.6f}")

    # --- Comparaison avec healpy (si disponible) ---
    try:
        import healpy as hp
        # healpy utilise le schéma RING par défaut ; on convertit la carte
        hpx_map_ring = hp.reorder(hpx_map, n2r=True)
        lons_flat = lon_grid.ravel()
        lats_flat = lat_grid.ravel()
        # healpy attend colatitude en radians et longitude en radians
        theta = np.radians(90.0 - lats_flat)
        phi = np.radians(lons_flat)
        vals_healpy = hp.get_interp_val(hpx_map_ring, theta, phi).reshape(lon_grid.shape)
        diff_hp = np.abs(vals_sphere - vals_healpy)
        print(f"\nDiff vs healpy (sphere): mean={diff_hp.mean():.6f}  max={diff_hp.max():.6f}")
    except ImportError:
        print("\n(healpy non disponible pour comparaison)")

    print("\n=== OK ===")
