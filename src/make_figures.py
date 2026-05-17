"""
make_figures.py
===============
Generates the publication-style figures used in the technical report and
the GitHub README. Run from src/:  python make_figures.py

Outputs (../figures):
  fig1_drying_profiles.png    -- phi(z) evolution, low vs high Peclet
  fig2_stratification_map.png -- stratification index vs Peclet number
  fig3_unit_cell.png          -- weave geometry + capillary film map
  fig4_deposition.png         -- DWR coverage + contact-angle map
  fig5_process_window.png     -- beading index over the process window
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from drying_1d import DryingParameters, solve_binary_drying
from weave_cell import (
    WeaveParameters,
    beading_performance_index,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Consistent, restrained styling (no default-matplotlib look).
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})

INK = "#1b2a4a"
ACCENT = "#c44536"
ACCENT2 = "#2a7f8e"


def fig1_drying_profiles():
    """Final phi(z) for the large (functional) species, low vs high Pe,
    showing the transition from uniform to strongly surface-segregated."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    cases = [
        ("Diffusion-limited (low Pe)", 8e-9, 30e-6, axes[0]),
        ("Evaporation-limited (high Pe)", 1.2e-6, 200e-6, axes[1]),
    ]
    for title, vev, H0, ax in cases:
        p = DryingParameters(evap_velocity_m_s=vev, H0_m=H0)
        PeL, PeS = p.peclet_numbers()
        r = solve_binary_drying(p, n_steps=1600, n_time_samples=5)
        x = r.x
        # Show a few times: early, mid, final.
        idxs = [0, len(r.t) // 3, 2 * len(r.t) // 3, -1]
        for j, k in enumerate(idxs):
            shade = 0.25 + 0.65 * j / (len(idxs) - 1)
            ax.plot(r.phi_large[k] / p.phi_large_0, x,
                    color=ACCENT, alpha=shade, lw=1.8,
                    label=f"t/t_dry = {r.t[k] / r.t[-1]:.2f}")
        ax.set_title(f"{title}\nPe(large) = {PeL:.2f}")
        ax.set_xlabel(r"normalised volume fraction  $\phi/\phi_0$")
        ax.set_ylabel("height in film  (0 = substrate, 1 = air side)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_ylim(0, 1)

    fig.suptitle("Drying-induced redistribution of functional polymer "
                 "through the residual film", y=1.02, fontweight="bold",
                 color=INK)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_drying_profiles.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig2_stratification_map():
    """Stratification index vs film Peclet number (parameter sweep),
    the quantitative validation curve."""
    vevs = np.geomspace(5e-9, 3e-6, 22)
    H0 = 150e-6
    Pe, SI = [], []
    for v in vevs:
        p = DryingParameters(evap_velocity_m_s=v, H0_m=H0)
        r = solve_binary_drying(p, n_steps=1400, n_time_samples=3)
        Pe.append(p.peclet_numbers()[0])
        SI.append(r.stratification_index())

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(Pe, SI, "o-", color=INK, mfc=ACCENT, mec=INK, ms=5, lw=1.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvspan(0, 1, color=ACCENT2, alpha=0.08)
    ax.text(0.16, max(SI) * 0.9, "diffusion-limited\n(near-uniform)",
            fontsize=8.5, color=ACCENT2, ha="left", va="top")
    ax.text(20, max(SI) * 0.35, "evaporation-limited\n(strong "
            "stratification)", fontsize=8.5, color=ACCENT, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("film Peclet number  Pe (large species)")
    ax.set_ylabel("stratification index  (small-on-top, +ve)")
    ax.set_title("Binary stratification grows monotonically with Pe",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_stratification_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig3_unit_cell():
    """Plain-weave unit-cell geometry and the capillary residual-film
    map that drives non-uniform deposition."""
    wp = WeaveParameters()
    o = beading_performance_index(WeaveParameters(),
                                  DryingParameters(evap_velocity_m_s=1e-7))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    im0 = axes[0].imshow(o["height"], cmap="bone", origin="lower")
    axes[0].set_title("Plain-weave surface elevation\n"
                      "(bright = yarn crowns)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                 label="relative height")

    im1 = axes[1].imshow(o["film_thickness"] * 1e6, cmap="viridis",
                         origin="lower")
    axes[1].set_title("Residual liquid film after spin\n"
                      "(capillary pooling in interstices)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                 label=r"film thickness ($\mu$m)")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)

    fig.suptitle("Idealised woven unit cell: geometry sets where the "
                 "re-proofer pools", y=1.02, fontweight="bold", color=INK)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_unit_cell.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig4_deposition():
    """Effective DWR coverage and the resulting contact-angle map, with
    the crown-starvation finding annotated."""
    bp = DryingParameters(evap_velocity_m_s=1e-7, phi_large_0=0.045)
    o = beading_performance_index(WeaveParameters(), bp)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    cov = np.ma.masked_where(o["pore_mask"], o["coverage"])
    im0 = axes[0].imshow(cov, cmap="magma", origin="lower",
                         vmin=0, vmax=1)
    axes[0].set_title("Effective usable DWR coverage\n"
                      "(dark interstitial lines = pooled, buried polymer)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04,
                 label="coverage fraction")

    ca = np.ma.masked_where(o["pore_mask"], o["contact_angle"])
    im1 = axes[1].imshow(ca, cmap="coolwarm_r", origin="lower")
    axes[1].set_title("Predicted local contact angle\n"
                      f"crowns {o['crown_contact_angle_deg']:.0f} deg < "
                      f"valleys {o['valley_contact_angle_deg']:.0f} deg")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04,
                 label="contact angle (deg)")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)

    fig.suptitle("Capillary pooling starves the yarn crowns -- the "
                 "surfaces a droplet actually rests on", y=1.02,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_deposition.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig5_process_window():
    """Beading index as a 2D process window over polymer concentration
    and drying rate -- the actionable, materials-development output.

    Uses a moderate grid; each cell is a full drying+weave evaluation, so
    this is the most expensive figure.
    """
    phis = np.linspace(0.015, 0.16, 9)
    vevs = np.geomspace(2e-8, 1.2e-6, 9)
    Z = np.zeros((len(phis), len(vevs)))
    for i, ph in enumerate(phis):
        for j, v in enumerate(vevs):
            bp = DryingParameters(evap_velocity_m_s=v, phi_large_0=ph)
            o = beading_performance_index(WeaveParameters(), bp, fast=True)
            Z[i, j] = o["beading_index"]
        print(f"  process-window row {i + 1}/{len(phis)}")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    cf = ax.contourf(vevs, phis, Z, levels=14, cmap="YlGnBu")
    cs = ax.contour(vevs, phis, Z, levels=[0.4, 0.6, 0.75],
                    colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%.2f", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("evaporation rate  (m/s)  ~ drying temperature")
    ax.set_ylabel(r"functional polymer loading  $\phi_0$")
    ax.set_title("Predicted beading-performance process window",
                 fontweight="bold", color=INK)
    fig.colorbar(cf, ax=ax, label="beading index (0-1)")
    ax.grid(False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig5_process_window.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig1_drying_profiles()
    fig2_stratification_map()
    fig3_unit_cell()
    fig4_deposition()
    fig5_process_window()
    print("All figures written to", os.path.abspath(FIGDIR))
