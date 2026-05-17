"""
drying_1d.py
============
One-dimensional drying model for a binary colloidal dispersion in an
evaporating film. This is the physics core of the DWR re-proofer model.

Physical picture
----------------
A DWR (durable water repellent) wash-in re-treatment is a colloidal
dispersion: water-repellent polymer particles (the "large" / functional
phase) plus a smaller carrier/surfactant phase, suspended in water. After
the garment is washed in the treatment and removed, water evaporates from
the wetted textile. As the air-liquid interface recedes it sweeps particles
toward the drying surface; diffusion opposes this. The competition is
governed by the film Peclet number,

        Pe = v_evap * H0 / D

where v_evap is the interface recession velocity, H0 the initial film
thickness, and D the particle diffusion coefficient (Stokes-Einstein).

For a *binary* dispersion the two species have different D (size-dependent),
so they stratify: under the right conditions the smaller particles enrich
the top surface and the larger ones are left below (or vice-versa,
depending on the regime). This is the same stratification physics studied
for drying colloidal films; here the deposit it leaves behind is the DWR
coating, and its uniformity controls water-beading performance.

Model
-----
We use the conservative areal-density variable q_i(x,t) = h(t) * phi_i,
where x = z / h(t) maps the shrinking physical film z in [0, h(t)] onto a
fixed computational domain x in [0, 1]. The physical diffusion equation
d(phi_i)/dt|_z = d/dz[ D_i d(phi_i)/dz ] transforms (verified symbolically
with sympy in the test suite) into the conservative form

    d(q_i)/dt|_x = d/dx [ (D_i / h^2) d(q_i)/dx  +  (h'/h) * x * q_i ]

The first term is diffusion with the mapped coefficient D_i/h^2. The
second is an advective concentration term arising from the moving frame:
with h' = dh/dt < 0 (the film recedes) it sweeps particles toward the
drying surface x = 1. This advective term is the physical origin of
surface enrichment and, because the two species have different diffusion
coefficients (different Peclet numbers), of binary stratification. Both
end faces carry zero total flux (only solvent leaves; particles are
retained), so integral(q dx) -- the total particle amount per unit area --
is conserved exactly. Recovering phi = q / h shows the film concentrating
as h shrinks, faster for the higher-Pe species.

A conservative finite-volume discretisation with closed end faces makes
integral(q dx) constant to the linear-solver tolerance, which is the
headline correctness check for the solver.

References (qualitative validation targets)
-------------------------------------------
- Routh & Russel, evaporation/Peclet framework for drying colloidal films.
- Fortini et al., model stratification in drying binary colloidal mixtures
  (small-on-top enrichment at high Pe).
- Standard Stokes-Einstein diffusion for hard-sphere colloids.

This module is intentionally dependency-light (numpy + scipy) and documented
so it can be read end-to-end in an interview.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.linalg import solve_banded

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant, J/K


def stokes_einstein_D(radius_m: float, temperature_K: float,
                      viscosity_Pa_s: float = 1.0e-3) -> float:
    """Stokes-Einstein diffusion coefficient for a sphere.

    D = k_B T / (6 pi mu r)

    Defaults to the viscosity of water at ~20 C (1.0e-3 Pa.s).
    """
    return K_B * temperature_K / (6.0 * np.pi * viscosity_Pa_s * radius_m)


@dataclass
class DryingParameters:
    """All inputs for a 1D binary drying run.

    Defaults correspond to a plausible consumer DWR wash-in re-treatment
    drying at room temperature: a sub-micron functional polymer particle
    and a smaller carrier/surfactant species, in a thin residual liquid
    film clinging to the textile after the spin cycle.
    """
    # Film / process
    H0_m: float = 50.0e-6              # initial residual film thickness (m)
    evap_velocity_m_s: float = 2.0e-8  # interface recession velocity (m/s)
    temperature_K: float = 293.15      # drying temperature (K)
    viscosity_Pa_s: float = 1.0e-3     # carrier (water) viscosity

    # Binary colloid: species "L" (large, functional repellent polymer)
    # and species "S" (small, carrier/surfactant phase)
    r_large_m: float = 120.0e-9        # large particle radius (m)
    r_small_m: float = 25.0e-9         # small particle radius (m)
    phi_large_0: float = 0.06          # initial volume fraction, large
    phi_small_0: float = 0.04          # initial volume fraction, small

    # Numerics
    n_cells: int = 240                 # finite-volume cells in x in [0,1]
    t_end_fraction: float = 0.92       # stop before full film collapse
                                       # (avoids singular 1/h as h->0)

    def diffusion_coeffs(self) -> tuple[float, float]:
        """Return (D_large, D_small) from Stokes-Einstein."""
        D_L = stokes_einstein_D(self.r_large_m, self.temperature_K,
                                self.viscosity_Pa_s)
        D_S = stokes_einstein_D(self.r_small_m, self.temperature_K,
                                self.viscosity_Pa_s)
        return D_L, D_S

    def peclet_numbers(self) -> tuple[float, float]:
        """Film Peclet numbers Pe_i = v_evap * H0 / D_i for each species."""
        D_L, D_S = self.diffusion_coeffs()
        Pe_L = self.evap_velocity_m_s * self.H0_m / D_L
        Pe_S = self.evap_velocity_m_s * self.H0_m / D_S
        return Pe_L, Pe_S


@dataclass
class DryingResult:
    """Container for the output of a drying run."""
    x: np.ndarray                      # cell-centre coords in [0,1]
    t: np.ndarray                      # time samples (s)
    h: np.ndarray                      # film thickness vs time (m)
    phi_large: np.ndarray              # phi_L[t, x]
    phi_small: np.ndarray              # phi_S[t, x]
    params: DryingParameters
    mass_error_large: float = 0.0      # relative mass conservation error
    mass_error_small: float = 0.0
    Pe_large: float = 0.0
    Pe_small: float = 0.0

    @property
    def final_phi_large(self) -> np.ndarray:
        return self.phi_large[-1]

    @property
    def final_phi_small(self) -> np.ndarray:
        return self.phi_small[-1]

    def stratification_index(self) -> float:
        """A scalar summarising vertical segregation of the two species.

        Defined as the difference in normalised centre-of-mass height
        between small and large species at the end of drying:

            SI = z_cm(small) - z_cm(large)   (in units of final film height)

        SI > 0  => small species enriched toward the drying (top) surface
                   relative to large  (classic "small-on-top").
        SI ~ 0  => well-mixed / uniform.
        SI < 0  => large species enriched toward the top.
        """
        x = self.x
        pL = self.final_phi_large
        pS = self.final_phi_small
        zL = np.trapz(x * pL, x) / np.trapz(pL, x)
        zS = np.trapz(x * pS, x) / np.trapz(pS, x)
        return float(zS - zL)

    def surface_enrichment(self, species: str = "large",
                           top_frac: float = 0.15) -> float:
        """Ratio of mean volume fraction in the top `top_frac` of the film
        to the initial uniform value. >1 means surface enrichment.
        """
        x = self.x
        if species == "large":
            phi = self.final_phi_large
            phi0 = self.params.phi_large_0
        else:
            phi = self.final_phi_small
            phi0 = self.params.phi_small_0
        mask = x >= (1.0 - top_frac)
        return float(np.mean(phi[mask]) / phi0)


def _step_implicit(q: np.ndarray, Dmap: float, adv_coeff: float,
                   x_faces: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Advance the conservative variable q one step.

    Solves, fully implicitly (backward Euler),

        dq/dt = d/dx [ (D/h^2) dq/dx  +  (h'/h) * x * q ]

    which is the *exact* conservative form of the drying problem in the
    mapped coordinate x = z/h (derived symbolically; see module docstring).
    The first term is diffusion with mapped coefficient Dmap = D/h^2; the
    second is the advective concentration term with adv_coeff = h'/h < 0
    (h shrinks), which sweeps particles toward the drying surface x=1 and
    is what produces surface enrichment and binary stratification.

    Discretisation: conservative finite volume on cell centres. Diffusive
    face flux is central; advective face flux uses first-order upwinding of
    the face value of (adv_coeff * x_face) for stability. Both end faces
    carry zero total flux (closed: only solvent leaves the film, particles
    are retained), which makes the scheme conserve sum(q)*dx exactly and
    hence the total particle amount, independent of dt. Backward Euler is
    unconditionally stable so dt is set by accuracy, not by the D/h^2
    stiffness that diverges as h -> 0.
    """
    n = q.size
    lower = np.zeros(n)   # sub-diagonal  (coupling to i-1)
    diag = np.ones(n)     # main diagonal
    upper = np.zeros(n)   # super-diagonal (coupling to i+1)

    Dc = Dmap / (dx * dx)          # diffusion divergence coefficient
    # Advective face "velocity" in mapped space: w(x) = adv_coeff * x.
    # adv_coeff = h'/h is negative, so w is negative (flux toward x=1 when
    # we account for the divergence sign), giving accumulation at the top.
    w_faces = adv_coeff * x_faces  # length n+1

    # Interior faces f = 1 .. n-1 separate cells (f-1) [left] and f [right].
    for f in range(1, n):
        iL = f - 1
        iR = f

        # --- Diffusion (central): flux_f = -Dmap (q_iR - q_iL)/dx ---
        # Divergence contribution: cell iL gets -(flux_right)/dx, etc.
        diag[iL] += dt * Dc
        upper[iL] += -dt * Dc
        diag[iR] += dt * Dc
        lower[iR] += -dt * Dc

        # --- Advection (first-order upwind on w_faces) ---
        wf = w_faces[f]
        if wf >= 0.0:
            # upwind cell is iL: flux_f = wf * q_iL
            a_self_L = wf / dx     # contribution of q_iL
            a_cross_R = 0.0
        else:
            a_self_L = 0.0
            a_cross_R = wf / dx    # contribution of q_iR

        # Cell iL loses this face flux (its right face): div += +flux/dx
        diag[iL] += dt * a_self_L
        upper[iL] += dt * a_cross_R
        # Cell iR gains it (its left face): div += -flux/dx
        lower[iR] += -dt * a_self_L
        diag[iR] += -dt * a_cross_R

    # End faces (f=0 and f=n): closed. x_faces[0]=0 makes the advective
    # flux there identically zero; we leave the diffusive end fluxes out
    # (no-flux) by construction. At f=n, x_faces[n]=1 but the face is the
    # receding interface: enforce zero total flux (particles retained), so
    # no terms are added for f=0 or f=n -> closed, exactly conservative.

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, q)


def solve_binary_drying(params: DryingParameters,
                        n_time_samples: int = 60,
                        n_steps: int = 1200) -> DryingResult:
    """Integrate the 1D binary drying problem.

    Conservative finite-volume scheme on a fixed grid x in [0,1] with a
    shrinking physical thickness h(t). Diffusion in physical space scales
    as D / h^2 in the mapped coordinate; the moving-frame advection is
    upwinded. Backward-Euler implicit time stepping is used so the step
    size is set by accuracy rather than the (otherwise crippling)
    diffusive stability limit as h -> 0.

    Returns a DryingResult including a mass-conservation diagnostic that
    should be ~machine precision -- the headline correctness check.
    """
    n = params.n_cells
    D_L, D_S = params.diffusion_coeffs()
    Pe_L, Pe_S = params.peclet_numbers()

    # Fixed computational grid (cell centres) on [0, 1]
    dx = 1.0 / n
    x_centres = (np.arange(n) + 0.5) * dx
    x_faces = np.arange(n + 1) * dx  # n+1 face positions in [0,1]

    # Initial conditions: uniform volume fractions
    phi_L = np.full(n, params.phi_large_0, dtype=float)
    phi_S = np.full(n, params.phi_small_0, dtype=float)

    # Film thickness evolution: linear recession (constant evaporative flux
    # regime, the standard assumption for the falling-rate-free period).
    H0 = params.H0_m
    v = params.evap_velocity_m_s
    h_min = H0 * (1.0 - params.t_end_fraction)
    t_end = (H0 - h_min) / v
    dhdt = -v  # constant

    # Conservative variable q = h * phi (areal density per unit x).
    # integral(q dx) == total particle amount per unit area, invariant.
    q_L = H0 * phi_L
    q_S = H0 * phi_S
    M_L0 = np.sum(q_L) * dx
    M_S0 = np.sum(q_S) * dx

    # Uniform time stepping (implicit scheme -> stable for any dt).
    dt = t_end / n_steps
    sample_stride = max(1, n_steps // n_time_samples)

    rec_t, rec_h = [0.0], [H0]
    rec_L, rec_S = [phi_L.copy()], [phi_S.copy()]

    t = 0.0
    h = H0
    for step in range(1, n_steps + 1):
        # March thickness to end of step (consistent with implicit scheme).
        h_new = h + dhdt * dt
        Dmap_L = D_L / (h_new * h_new)
        Dmap_S = D_S / (h_new * h_new)
        # Advection coefficient h'/h evaluated at end of step.
        adv_coeff = dhdt / h_new

        q_L = _step_implicit(q_L, Dmap_L, adv_coeff, x_faces, dx, dt)
        q_S = _step_implicit(q_S, Dmap_S, adv_coeff, x_faces, dx, dt)
        np.clip(q_L, 0.0, None, out=q_L)
        np.clip(q_S, 0.0, None, out=q_S)

        t += dt
        h = h_new

        if step % sample_stride == 0:
            rec_t.append(t)
            rec_h.append(h)
            rec_L.append((q_L / h).copy())
            rec_S.append((q_S / h).copy())

    # Ensure final state captured
    rec_t.append(t)
    rec_h.append(h)
    rec_L.append((q_L / h).copy())
    rec_S.append((q_S / h).copy())

    phi_L_arr = np.array(rec_L)
    phi_S_arr = np.array(rec_S)
    h_arr = np.array(rec_h)
    t_arr = np.array(rec_t)

    # Mass conservation diagnostic. The conserved quantity is the total
    # particle amount per unit area = integral(q dx) = h * integral(phi dx).
    # Evaluate from the final phi profile and final thickness.
    M_Lf = np.sum(phi_L_arr[-1]) * dx * h_arr[-1]
    M_Sf = np.sum(phi_S_arr[-1]) * dx * h_arr[-1]
    err_L = abs(M_Lf - M_L0) / M_L0
    err_S = abs(M_Sf - M_S0) / M_S0

    return DryingResult(
        x=x_centres, t=t_arr, h=h_arr,
        phi_large=phi_L_arr, phi_small=phi_S_arr,
        params=params,
        mass_error_large=err_L, mass_error_small=err_S,
        Pe_large=Pe_L, Pe_small=Pe_S,
    )


if __name__ == "__main__":
    # Smoke test: run defaults and print key diagnostics.
    p = DryingParameters()
    Pe_L, Pe_S = p.peclet_numbers()
    D_L, D_S = p.diffusion_coeffs()
    print(f"D_large  = {D_L:.3e} m^2/s")
    print(f"D_small  = {D_S:.3e} m^2/s")
    print(f"Pe_large = {Pe_L:.3f}")
    print(f"Pe_small = {Pe_S:.3f}")
    res = solve_binary_drying(p)
    print(f"mass error (large) = {res.mass_error_large:.2e}")
    print(f"mass error (small) = {res.mass_error_small:.2e}")
    print(f"stratification index = {res.stratification_index():+.4f}")
    print(f"surface enrichment (large) = "
          f"{res.surface_enrichment('large'):.3f}")
    print(f"surface enrichment (small) = "
          f"{res.surface_enrichment('small'):.3f}")
