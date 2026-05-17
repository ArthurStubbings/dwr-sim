"""
test_weave_cell.py
==================
Tests for the plain-weave unit cell and beading-performance model.

These check the *behavioural* correctness an interviewer would probe:
geometry sanity, monotone response to polymer concentration, the central
"pooling starves the crowns" finding, and that the performance index is a
well-behaved bounded metric.
"""

import numpy as np

from drying_1d import DryingParameters
from weave_cell import (
    WeaveParameters,
    build_unit_cell,
    compute_deposition_map,
    cassie_baxter_contact_angle,
    beading_performance_index,
)


def test_unit_cell_geometry():
    """Height field spans [0,1]; pore fraction is approximately as
    requested; interstitial film is thicker than crown film."""
    wp = WeaveParameters()
    height, film, pore = build_unit_cell(wp)
    assert np.isclose(height.min(), 0.0) and np.isclose(height.max(), 1.0)
    assert abs(pore.mean() - wp.pore_fraction) < 0.05
    # Valleys (low height) must have thicker residual film than crowns.
    assert film[height < 0.2].mean() > film[height > 0.8].mean()
    print(f"PASS  unit-cell geometry (pore frac {pore.mean():.3f})")


def test_cassie_baxter_bounds():
    """Effective contact angle is bounded by the bare and intrinsic
    angles for any coverage in [0,1]."""
    wp = WeaveParameters()
    cov = np.linspace(0.0, 1.0, 50)
    theta = cassie_baxter_contact_angle(cov, wp)
    assert theta.min() >= wp.bare_contact_angle_deg - 1e-6
    assert theta.max() <= wp.intrinsic_contact_angle_deg + 1e-6
    # Monotone increasing in coverage.
    assert np.all(np.diff(theta) >= -1e-9)
    print("PASS  Cassie-Baxter bounded and monotone")


def test_coverage_in_unit_range():
    """Effective coverage map stays a physical fraction in [0,1] and is
    exactly zero in open pores."""
    wp = WeaveParameters()
    bp = DryingParameters(evap_velocity_m_s=1e-7)
    cov, height, film, pore = compute_deposition_map(wp, bp)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert np.all(cov[pore] == 0.0)
    print(f"PASS  coverage in [0,1] (max {cov.max():.3f})")


def test_beading_increases_with_concentration():
    """More functional polymer -> better predicted beading. Also checks
    that an under-loaded dispersion fails to bead (CA < 90)."""
    idx = []
    for phiL in (0.01, 0.03, 0.06, 0.10, 0.18):
        bp = DryingParameters(evap_velocity_m_s=1e-7, phi_large_0=phiL)
        o = beading_performance_index(WeaveParameters(), bp)
        idx.append(o["beading_index"])
    assert np.all(np.diff(idx) > 0), f"not monotone in conc: {idx}"
    # Lowest loading should be sub-beading (mean CA < 90 deg).
    bp_lo = DryingParameters(evap_velocity_m_s=1e-7, phi_large_0=0.01)
    o_lo = beading_performance_index(WeaveParameters(), bp_lo)
    assert o_lo["mean_contact_angle_deg"] < 90.0
    print(f"PASS  beading rises with concentration ({idx[0]:.2f} -> "
          f"{idx[-1]:.2f}); under-loaded fails to bead")


def test_pooling_starves_crowns():
    """Central finding: capillary pooling concentrates film in the
    interstices, so valleys end up with a HIGHER effective contact angle
    than the crowns -- i.e. the surfaces a droplet rests on are the
    under-protected ones."""
    bp = DryingParameters(evap_velocity_m_s=1e-7, phi_large_0=0.05)
    o = beading_performance_index(WeaveParameters(), bp)
    assert o["valley_contact_angle_deg"] > o["crown_contact_angle_deg"], (
        f"expected valley CA > crown CA, got "
        f"{o['valley_contact_angle_deg']:.1f} vs "
        f"{o['crown_contact_angle_deg']:.1f}"
    )
    print(f"PASS  pooling starves crowns "
          f"(crown {o['crown_contact_angle_deg']:.1f} deg < "
          f"valley {o['valley_contact_angle_deg']:.1f} deg)")


def test_index_bounded():
    """The beading index is always a clean number in [0,1]."""
    for vev in (1e-8, 1e-7, 1e-6):
        bp = DryingParameters(evap_velocity_m_s=vev)
        o = beading_performance_index(WeaveParameters(), bp)
        assert 0.0 <= o["beading_index"] <= 1.0
        assert np.isfinite(o["mean_contact_angle_deg"])
    print("PASS  beading index bounded in [0,1] and finite")


if __name__ == "__main__":
    tests = [
        test_unit_cell_geometry,
        test_cassie_baxter_bounds,
        test_coverage_in_unit_range,
        test_beading_increases_with_concentration,
        test_pooling_starves_crowns,
        test_index_bounded,
    ]
    for fn in tests:
        fn()
    print(f"\nAll {len(tests)} tests passed.")
