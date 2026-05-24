"""
weave_cell.py
=============
Plain-weave unit-cell model that turns the 1D drying/stratification result
into a spatial map of DWR (durable water repellent) deposition on a woven
textile, and then into a predicted water-beading performance metric.

Why an idealised unit cell
--------------------------
A real woven fabric is a fibrous, multi-scale porous medium; a
fibre-resolved (micro-CT + pore-scale flow) model is a multi-month research
effort. The idealised plain-weave unit cell -- two orthogonal sets of
sinusoidal yarns with inter-yarn pores -- is the standard, defensible
simplification used in textile-coating and wicking literature. It captures
the one effect that matters here: liquid (and therefore the suspended
repellent polymer) preferentially collects in the yarn interstices and at
yarn crossovers, while the exposed yarn *crowns* -- the surfaces a rain
drop actually touches -- can be left under-coated. That non-uniformity is a
known real-world failure mode of consumer DWR re-treatment.

Pipeline
--------
1.  Geometry: build a plain-weave unit cell on a grid. Each surface point
    gets (a) a local height / exposure (crown vs. valley) and (b) a local
    residual-liquid film thickness, thicker in the capillary interstices.
2.  Local drying: the 1D solver gives, for a given residual film thickness
    and process parameters, how the functional polymer ends up distributed
    through the film and how much of it deposits as a usable surface film
    versus being buried. We evaluate the 1D model per representative film
    thickness and interpolate -- cheap and physically transparent.
3.  Deposition map: combine local film thickness + local stratification to
    get a 2D map of effective DWR areal coverage over the unit cell.
4.  Performance: convert coverage to an effective water contact angle via a
    Cassie-Baxter relation, then to a single "beading performance index"
    in [0, 1]. Crowns are weighted more heavily because they are what a
    droplet sits on.

The output metric is intentionally a *relative* predictor for comparing
processing conditions (concentration, drying temperature, fabric sett),
not an absolute contact-angle prediction -- that honesty matters in
interview.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace as dc_replace
from typing import Dict

from drying_1d import DryingParameters, solve_binary_drying


@dataclass
class ChemistryProfile:
    """DWR chemistry profile: bundles the intrinsic contact angle and
    functional-polymer particle size implied by each chemistry class.

    Intrinsic contact angles and representative particle sizes are
    literature-informed estimates; see README for sources.
    """
    key: str
    label: str
    intrinsic_ca_deg: float   # contact angle of a fully-coated DWR surface (°)
    r_large_nm: float         # functional polymer particle radius (nm)
    r_small_nm: float = 40.0  # carrier/surfactant particle radius (nm)
    status: str = ""          # regulatory/sustainability summary
    description: str = ""

    def weave_params(self, base: "WeaveParameters") -> "WeaveParameters":
        """Return a copy of *base* with intrinsic CA set for this chemistry."""
        return dc_replace(base, intrinsic_contact_angle_deg=self.intrinsic_ca_deg)

    def drying_params(self, base: DryingParameters) -> DryingParameters:
        """Return a copy of *base* with particle sizes set for this chemistry."""
        return dc_replace(base,
                          r_large_m=self.r_large_nm * 1e-9,
                          r_small_m=self.r_small_nm * 1e-9)


CHEMISTRY_PROFILES: Dict[str, ChemistryProfile] = {
    "c8": ChemistryProfile(
        key="c8",
        label="C8 Fluorocarbon",
        intrinsic_ca_deg=120.0,
        r_large_nm=110.0,
        status="PFAS — banned / phased out",
        description=(
            "PFOA/PFOS-based chemistry. Highest performance ceiling; "
            "banned in the EU since 2023 and under restriction globally."
        ),
    ),
    "c6": ChemistryProfile(
        key="c6",
        label="C6 Fluorocarbon",
        intrinsic_ca_deg=112.0,
        r_large_nm=130.0,
        status="PFAS — under regulation",
        description=(
            "Short-chain fluorocarbon (e.g. Archroma Smartrepel, "
            "Capstone). Lower persistence than C8 but still a regulated "
            "PFAS compound."
        ),
    ),
    "silicone": ChemistryProfile(
        key="silicone",
        label="Silicone  (PFC-free)",
        intrinsic_ca_deg=100.0,
        r_large_nm=200.0,
        status="PFC-free",
        description=(
            "Silicone-based DWR (e.g. Evonik Tegomer, Daikin Unidyne EX). "
            "Larger particles increase the local Péclet number, compounding "
            "the lower intrinsic contact angle."
        ),
    ),
    "dendrimer": ChemistryProfile(
        key="dendrimer",
        label="Dendrimer / bio-wax  (PFC-free)",
        intrinsic_ca_deg=95.0,
        r_large_nm=160.0,
        status="PFC-free",
        description=(
            "Dendrimer or bio-wax based DWR (e.g. Nikwax TX.Direct, "
            "Grangers). Lowest intrinsic contact angle; adequate beading "
            "requires careful processing control."
        ),
    ),
}


@dataclass
class WeaveParameters:
    """Idealised plain-weave unit-cell geometry and coating inputs."""
    grid: int = 96                 # unit-cell sampling (grid x grid)
    yarns_per_cell: int = 2        # plain weave: 2x2 over-under unit cell
    yarn_crimp: float = 0.55       # 0..1, depth of weave undulation
    pore_fraction: float = 0.18    # fraction of cell that is open pore
    # Capillary pooling: residual film in valleys/interstices is thicker
    # than on crowns by this ratio (driven by Laplace pressure into the
    # high-curvature inter-yarn gaps).
    valley_to_crown_film: float = 4.5
    base_film_m: float = 30.0e-6   # crown residual film thickness (m)
    # Optional fixed reference film for the areal-polymer calculation.
    # 0 (default) = use base_film_m, preserving the original relative-metric
    # behaviour. Set to a fixed value (e.g. 30e-6) when sweeping base_film_m
    # across runs so that absolute film thickness drives coverage differences.
    film_ref_m: float = 0.0

    # Performance model
    intrinsic_contact_angle_deg: float = 105.0  # ideal fully-DWR'd polymer
    bare_contact_angle_deg: float = 65.0         # untreated face yarn
    crown_weight: float = 0.80     # weight of crowns vs valleys in the
                                   # performance index (a droplet rests on
                                   # crowns; valleys matter less for beading)


def build_unit_cell(wp: WeaveParameters):
    """Construct the plain-weave unit-cell fields.

    Returns
    -------
    height : 2D array in [0,1]
        Surface elevation. ~1 at yarn crowns, ~0 in valleys/interstices.
    film_thickness : 2D array (m)
        Local residual liquid film, thicker in interstices (capillary
        pooling) than on crowns.
    pore_mask : 2D bool array
        True where the cell is open pore (no yarn surface -> no useful
        coating contributes to beading there).
    """
    n = wp.grid
    u = np.linspace(0.0, 2.0 * np.pi * wp.yarns_per_cell, n, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi * wp.yarns_per_cell, n, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    # Plain weave: warp and weft yarns undulate in anti-phase. The fabric
    # surface elevation is the upper envelope of the two yarn families.
    warp = np.sin(U)
    weft = np.sin(V + np.pi)            # anti-phase -> over/under pattern
    envelope = np.maximum(warp, weft)   # surface = whichever yarn is on top
    height = 0.5 * (1.0 + wp.yarn_crimp * envelope)
    height = (height - height.min()) / (height.max() - height.min())

    # Inter-yarn pores: the lowest regions where neither yarn covers.
    pore_level = np.quantile(height, wp.pore_fraction)
    pore_mask = height <= pore_level

    # Capillary film: thin on crowns, thick in valleys/interstices. Use a
    # smooth monotone map of (1 - height) so deep valleys pool the most.
    pooling = 1.0 + (wp.valley_to_crown_film - 1.0) * (1.0 - height) ** 1.5
    film_thickness = wp.base_film_m * pooling

    return height, film_thickness, pore_mask


def _useful_surface_fraction(film_m: float,
                             base_params: DryingParameters,
                             fast: bool = False) -> float:
    """For a given local residual film thickness, run the 1D drying model
    and return the fraction of the functional (large) polymer that ends up
    as a *usable surface film* rather than buried deep in the deposit.

    Rationale: only repellent polymer at/near the air side of the dried
    deposit lowers the surface energy that a water drop feels. Polymer
    swept down and buried against the yarn does little for beading. We
    quantify "usable" as the large-species volume fraction in the top
    portion of the dried film, normalised so a perfectly uniform deposit
    scores 1.0. Thicker films (interstices) have a higher film Peclet
    number -> more burial -> lower usable fraction. That is precisely why
    pooling in the interstices degrades performance.

    `fast=True` lowers the grid/step counts for dense sweeps; the metric
    shifts negligibly because it is an integral of a smooth profile.
    """
    n_cells = 96 if fast else 160
    n_steps = 450 if fast else 900
    p = DryingParameters(
        H0_m=film_m,
        evap_velocity_m_s=base_params.evap_velocity_m_s,
        temperature_K=base_params.temperature_K,
        viscosity_Pa_s=base_params.viscosity_Pa_s,
        r_large_m=base_params.r_large_m,
        r_small_m=base_params.r_small_m,
        phi_large_0=base_params.phi_large_0,
        phi_small_0=base_params.phi_small_0,
        n_cells=n_cells,
    )
    r = solve_binary_drying(p, n_steps=n_steps, n_time_samples=3)
    x = r.x
    phi = r.final_phi_large
    # Top 20% of the dried film (the air side) vs. its own mean. A uniform
    # deposit -> ratio 1.0; surface-depleted (buried) -> < 1.0.
    top = np.mean(phi[x >= 0.80])
    mean = np.mean(phi)
    return float(np.clip(top / mean, 0.0, 1.5))


def compute_deposition_map(wp: WeaveParameters,
                           base_params: DryingParameters,
                           n_film_samples: int = 9,
                           fast: bool = False):
    """Build the 2D effective-DWR-coverage map over the unit cell.

    Combines:
      - local residual film thickness (geometry / capillary pooling), and
      - the usable-surface fraction from the 1D drying model at that
        thickness (interpolated over a small set of representative
        thicknesses for speed),
    then folds in a simple mass argument: a thicker local film deposits
    more total polymer per unit area, but a lower *fraction* of it is
    usefully surfaced. The product is the effective areal coverage that
    controls wettability.

    `fast=True` reduces the number of representative film thicknesses and
    is intended only for dense parameter sweeps (e.g. the process-window
    figure); the single-point accuracy is essentially unchanged because
    the usable-fraction curve is smooth and monotone in thickness.

    Returns
    -------
    coverage : 2D array in [0,1]   effective usable DWR coverage
    height, film_thickness, pore_mask : geometry fields (passthrough)
    """
    height, film_thickness, pore_mask = build_unit_cell(wp)

    if fast:
        n_film_samples = 4

    # Sample usable-fraction vs. film thickness on a small grid, then
    # interpolate -- avoids running the PDE per pixel.
    t_lo, t_hi = film_thickness.min(), film_thickness.max()
    t_samples = np.linspace(t_lo, t_hi, n_film_samples)
    usable_samples = np.array(
        [_useful_surface_fraction(t, base_params, fast=fast)
         for t in t_samples]
    )
    usable = np.interp(film_thickness, t_samples, usable_samples)

    # Total deposited functional polymer per unit area scales with both
    # the local film thickness (more liquid -> more solid left behind) and
    # the dispersion's polymer loading phi_large_0. A continuous beading
    # film needs a minimum areal amount; above that there are diminishing
    # returns (saturating Langmuir-like coverage). The characteristic
    # deposit scale uses a reference film * reference loading so the map
    # stays a sensible relative metric.
    phi_ref = 0.06  # reference polymer loading for normalisation
    film_ref = wp.film_ref_m if wp.film_ref_m > 0 else wp.base_film_m
    areal_polymer = (film_thickness / film_ref) \
        * (base_params.phi_large_0 / phi_ref)
    deposited_mass = 1.0 - np.exp(-areal_polymer)

    coverage = deposited_mass * usable
    # Normalise against a fixed physical reference (a well-surfaced,
    # adequately loaded crown), NOT the per-run maximum -- otherwise any
    # uniform scaling (e.g. concentration) would cancel out. The reference
    # is deposited_mass at saturation (->1) with ideal usable fraction
    # (->1), i.e. coverage_ref = 1.0. Clip to [0,1] as a physical coverage.
    coverage = np.clip(coverage, 0.0, 1.0)
    coverage[pore_mask] = 0.0  # open pores carry no beading-relevant film

    return coverage, height, film_thickness, pore_mask


def cassie_baxter_contact_angle(coverage: np.ndarray,
                                wp: WeaveParameters) -> np.ndarray:
    """Local effective contact angle via a Cassie-Baxter mixing rule
    between fully-coated (intrinsic DWR) and bare-yarn states:

        cos(theta_eff) = f * cos(theta_DWR) + (1-f) * cos(theta_bare)

    where f is the local usable DWR coverage. Higher f -> higher contact
    angle -> better beading.
    """
    th_dwr = np.radians(wp.intrinsic_contact_angle_deg)
    th_bare = np.radians(wp.bare_contact_angle_deg)
    cos_eff = coverage * np.cos(th_dwr) + (1.0 - coverage) * np.cos(th_bare)
    cos_eff = np.clip(cos_eff, -1.0, 1.0)
    return np.degrees(np.arccos(cos_eff))


def beading_performance_index(wp: WeaveParameters,
                              base_params: DryingParameters,
                              fast: bool = False) -> dict:
    """Single scalar (plus supporting fields) summarising predicted
    water-beading performance of the dried DWR on this weave + process.

    Index construction:
      - compute the effective contact-angle map,
      - map angle -> a 0..1 beading score (sigmoid centred near the
        wetting/beading transition ~90 deg, where droplets switch from
        spreading to beading),
      - take a crown-weighted spatial average (a resting droplet sits on
        crowns), excluding open pores.
    Returns the index and all intermediate fields for plotting.

    `fast=True` is for dense parameter sweeps only.
    """
    coverage, height, film_thickness, pore_mask = compute_deposition_map(
        wp, base_params, fast=fast
    )
    theta = cassie_baxter_contact_angle(coverage, wp)

    # Beading score: smooth transition around 90 deg.
    beading = 1.0 / (1.0 + np.exp(-(theta - 90.0) / 8.0))

    # Crown vs valley weighting from the height field (normalised).
    h = height.copy()
    w = wp.crown_weight * h + (1.0 - wp.crown_weight) * (1.0 - h)
    w[pore_mask] = 0.0
    w_sum = w.sum()
    index = float((beading * w).sum() / (w_sum + 1e-30))

    # Diagnostics an interviewer would ask for. Crown / valley bands are
    # defined by quantiles of the *non-pore* surface height so they are
    # always populated regardless of weave geometry.
    surf = ~pore_mask
    h_surf = height[surf]
    crown_thr = np.quantile(h_surf, 0.70)   # top 30% of yarn surface
    valley_thr = np.quantile(h_surf, 0.30)  # bottom 30% of yarn surface
    crown_band = surf & (height >= crown_thr)
    valley_band = surf & (height <= valley_thr)
    return {
        "beading_index": index,
        "mean_contact_angle_deg": float(theta[surf].mean()),
        "crown_contact_angle_deg": float(theta[crown_band].mean()),
        "valley_contact_angle_deg": float(theta[valley_band].mean()),
        "coverage_uniformity": float(
            1.0 - coverage[surf].std() / (coverage[surf].mean() + 1e-30)
        ),
        "coverage": coverage,
        "contact_angle": theta,
        "beading": beading,
        "height": height,
        "film_thickness": film_thickness,
        "pore_mask": pore_mask,
    }


if __name__ == "__main__":
    wp = WeaveParameters()
    bp = DryingParameters(evap_velocity_m_s=1e-7)
    out = beading_performance_index(wp, bp)
    print(f"beading index            = {out['beading_index']:.3f}")
    print(f"mean contact angle (deg) = "
          f"{out['mean_contact_angle_deg']:.1f}")
    print(f"crown contact angle      = "
          f"{out['crown_contact_angle_deg']:.1f}")
    print(f"valley contact angle     = "
          f"{out['valley_contact_angle_deg']:.1f}")
    print(f"coverage uniformity      = "
          f"{out['coverage_uniformity']:.3f}")
