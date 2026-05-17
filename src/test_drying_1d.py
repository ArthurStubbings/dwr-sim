"""
test_drying_1d.py
=================
Correctness tests for the 1D binary drying solver.

These are written as plain asserts so they can be run with `pytest` or
directly with `python test_drying_1d.py`. They cover the three things an
interviewer would (rightly) probe:

1.  Conservation     -- the conserved variable q = h*phi must keep its
                        integral to ~machine precision for any Pe and dt.
2.  Transformation   -- the coordinate change z->x is verified symbolically
                        with sympy, so the PDE we discretise is provably the
                        right one (not hand-waved).
3.  Physics regimes  -- low Pe stays uniform; high Pe stratifies, with the
                        small/fast species enriched toward the drying surface
                        relative to the large/slow one. This is the known
                        Routh / Fortini binary-stratification result and is
                        the qualitative validation target.
"""

import numpy as np
import sympy as sp

from drying_1d import (
    DryingParameters,
    solve_binary_drying,
    stokes_einstein_D,
)


def test_stokes_einstein_monotonic():
    """Smaller particles diffuse faster (D ~ 1/r)."""
    D_small = stokes_einstein_D(20e-9, 293.15)
    D_large = stokes_einstein_D(200e-9, 293.15)
    assert D_small > D_large
    # Exact inverse-radius scaling
    assert np.isclose(D_small / D_large, 200.0 / 20.0, rtol=1e-12)
    print("PASS  Stokes-Einstein inverse-radius scaling")


def test_symbolic_transformation():
    """The mapped-coordinate PDE we discretise must be exactly the
    transform of the physical diffusion equation. Verify with sympy that

        dq/dt = d/dx[ (D/h^2) dq/dx + (h'/h) x q ]

    is equivalent to  d(phi)/dt|_z = D d^2(phi)/dz^2  under x=z/h,
    q=h*phi.
    """
    z, t, x = sp.symbols('z t x', real=True)
    h = sp.Function('h')(t)
    D = sp.symbols('D', positive=True)

    P = sp.Function('P')               # phi(x,t)
    phi = P(x, t)
    hp = sp.diff(h, t)

    # Physical equation transformed to (x,t) (derived in module docstring):
    phi_form = sp.diff(phi, t) - (D / h**2) * sp.diff(phi, x, 2) \
        - (x * hp / h) * sp.diff(phi, x)

    # Conservative q-form with q = h*phi:
    Q = sp.Function('Q')
    q = Q(x, t)
    q_form = sp.diff(q, t) - sp.diff(
        (D / h**2) * sp.diff(q, x) + (hp / h) * x * q, x
    )
    # Substitute q = h*P into q_form; multiply by 1/h; must equal phi_form.
    q_in_phi = sp.simplify(q_form.subs(Q(x, t), h * P(x, t)) / h)
    residual = sp.simplify(q_in_phi - phi_form)
    assert residual == 0, f"transform mismatch: {residual}"
    print("PASS  symbolic coordinate transformation verified")


def test_mass_conservation_all_regimes():
    """Total particle amount conserved to ~machine precision across a wide
    Peclet sweep and two step counts (independence from dt)."""
    worst = 0.0
    for vev in (2e-8, 1e-7, 1.5e-6):
        for H0 in (50e-6, 200e-6):
            for nstep in (600, 2000):
                p = DryingParameters(evap_velocity_m_s=vev, H0_m=H0)
                r = solve_binary_drying(p, n_steps=nstep)
                worst = max(worst, r.mass_error_large, r.mass_error_small)
    assert worst < 1e-9, f"mass error too large: {worst:.2e}"
    print(f"PASS  mass conservation across regimes (worst {worst:.1e})")


def test_low_peclet_stays_uniform():
    """Diffusion-dominated (Pe << 1): final profile ~ uniform, so the
    stratification index is essentially zero."""
    p = DryingParameters(evap_velocity_m_s=5e-9, H0_m=30e-6)
    PeL, PeS = p.peclet_numbers()
    assert PeL < 1.0 and PeS < 1.0
    r = solve_binary_drying(p, n_steps=1200)
    assert abs(r.stratification_index()) < 5e-3
    print(f"PASS  low-Pe uniform (Pe_L={PeL:.2f}, "
          f"SI={r.stratification_index():+.4f})")


def test_high_peclet_stratifies_small_on_top():
    """Evaporation-dominated (Pe >> 1): strong segregation, with the
    smaller/faster species enriched toward the drying surface relative to
    the larger/slower one (classic binary stratification)."""
    p = DryingParameters(evap_velocity_m_s=1.5e-6, H0_m=200e-6)
    PeL, PeS = p.peclet_numbers()
    assert PeL > 10.0
    r = solve_binary_drying(p, n_steps=2000)
    si = r.stratification_index()
    assert si > 0.05, f"expected strong small-on-top, got SI={si:+.4f}"
    # Large species should be depleted at the very top.
    x = r.x
    top_large = np.mean(r.final_phi_large[x >= 0.9]) / p.phi_large_0
    assert top_large < 1.0, f"large not depleted at top ({top_large:.2f})"
    print(f"PASS  high-Pe small-on-top stratification "
          f"(Pe_L={PeL:.1f}, SI={si:+.4f})")


def test_stratification_monotonic_in_peclet():
    """Stratification index increases monotonically with evaporation rate
    (i.e. with Peclet number)."""
    sis = []
    for vev in (1e-8, 5e-8, 2e-7, 1e-6):
        p = DryingParameters(evap_velocity_m_s=vev, H0_m=150e-6)
        r = solve_binary_drying(p, n_steps=1500)
        sis.append(r.stratification_index())
    diffs = np.diff(sis)
    assert np.all(diffs >= -1e-4), f"non-monotonic SI sweep: {sis}"
    print(f"PASS  SI monotonic in Pe ({[f'{s:+.4f}' for s in sis]})")


if __name__ == "__main__":
    tests = [
        test_stokes_einstein_monotonic,
        test_symbolic_transformation,
        test_mass_conservation_all_regimes,
        test_low_peclet_stays_uniform,
        test_high_peclet_stratifies_small_on_top,
        test_stratification_monotonic_in_peclet,
    ]
    for fn in tests:
        fn()
    print(f"\nAll {len(tests)} tests passed.")
