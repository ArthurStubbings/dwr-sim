"""
make_dashboard.py
=================
Precomputes a parameter sweep over the DWR coating model and bakes the
results into a single self-contained interactive HTML file.

Run from src/:  python make_dashboard.py
Output:         ../index.html   (open in any browser; no server needed)

Sweep grid: 4 DWR chemistries × 6 polymer loadings × 6 drying rates × 3
yarn crimps = 432 model evaluations. Each stores a 32×32 coverage map
and contact-angle map encoded as uint8 base64. Total HTML file ~1-2 MB.

The HTML file works offline, requires no running server, and is hostable
on GitHub Pages. Sliders index the precomputed grid — instant response.
"""

import os
import sys
import json
import time
import base64
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Run from src/
sys.path.insert(0, os.path.dirname(__file__))
from drying_1d import DryingParameters
from weave_cell import WeaveParameters, beading_performance_index, CHEMISTRY_PROFILES

OUTDIR = os.path.join(os.path.dirname(__file__), "..")
OUTFILE = os.path.join(OUTDIR, "index.html")
MAP_GRID = 32   # spatial resolution of each stored heatmap

# ── Parameter grid ────────────────────────────────────────────────────────────

PHI_VALUES = [0.020, 0.040, 0.060, 0.085, 0.105, 0.130]
PHI_LABELS = ["2 %", "4 %", "6 %", "8.5 %", "10.5 %", "13 %"]

EVAP_VALUES = list(np.geomspace(2e-8, 8e-7, 6))
# Approximate temperature equivalents via vapour-pressure scaling (Clausius-Clapeyron)
# relative to 20 °C. Labelled as approximate — real drying rate also depends on
# airflow and garment geometry.
EVAP_LABELS = [
    "~20 °C  (line dry)",
    "~30 °C  (warm air)",
    "~40 °C  (gentle tumble)",
    "~55 °C  (tumble low)",
    "~65 °C  (tumble medium)",
    "~80 °C  (tumble high)",
]

CRIMP_VALUES = [0.30, 0.50, 0.70]
CRIMP_LABELS = ["Low — even film", "Medium — pooling", "High — crown drainage"]

CHEM_KEYS   = list(CHEMISTRY_PROFILES.keys())            # ['c8','c6','silicone','dendrimer']
CHEM_LABELS = [CHEMISTRY_PROFILES[k].label for k in CHEM_KEYS]

WASH_COUNTS = [0, 5, 10, 15, 20, 25, 30]

N_PHI   = len(PHI_VALUES)
N_EVAP  = len(EVAP_VALUES)
N_CRIMP = len(CRIMP_VALUES)
N_CHEM  = len(CHEM_KEYS)
N_WASH  = len(WASH_COUNTS)
N_TOTAL = N_CHEM * N_PHI * N_EVAP * N_CRIMP


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_idx(chi, pi, ei, ci):
    """Chemistry is the outermost dimension."""
    return chi * N_PHI * N_EVAP * N_CRIMP + pi * N_EVAP * N_CRIMP + ei * N_CRIMP + ci


def encode_map(arr2d, vmin, vmax):
    """Scale arr2d to uint8 and return base64 string."""
    scaled = np.clip((arr2d - vmin) / (vmax - vmin + 1e-30), 0, 1)
    uint8 = (scaled * 255).astype(np.uint8)
    return base64.b64encode(uint8.tobytes()).decode("ascii")


def encode_bool_map(bool2d):
    """Encode a boolean mask as uint8 base64 (True → 1)."""
    return base64.b64encode(bool2d.astype(np.uint8).tobytes()).decode("ascii")


# ── Parallel worker (must be module-level for multiprocessing pickling) ───────

def _compute_run(args):
    chem_key, phi, evap, crimp, vtc, base_film, map_grid, wash_counts = args
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from weave_cell import (WeaveParameters, beading_performance_index,
                            CHEMISTRY_PROFILES, wash_beading_curve)
    from drying_1d import DryingParameters
    ch = CHEMISTRY_PROFILES[chem_key]
    wp = WeaveParameters(
        grid=map_grid, yarn_crimp=crimp,
        valley_to_crown_film=vtc, base_film_m=base_film,
        film_ref_m=30e-6,
        intrinsic_contact_angle_deg=ch.intrinsic_ca_deg,
    )
    bp = DryingParameters(
        evap_velocity_m_s=evap, phi_large_0=phi,
        r_large_m=ch.r_large_nm * 1e-9,
    )
    out = beading_performance_index(wp, bp, fast=True)
    wash_data = wash_beading_curve(out, wp, ch, wash_counts)
    return {
        "bi":       round(float(out["beading_index"]), 3),
        "cca":      round(float(out["crown_contact_angle_deg"]), 1),
        "vca":      round(float(out["valley_contact_angle_deg"]), 1),
        "mca":      round(float(out["mean_contact_angle_deg"]), 1),
        "cov":      out["coverage"].copy(),
        "ca":       out["contact_angle"].copy(),
        "bi_wash":  wash_data["bi"],
        "cca_wash": wash_data["cca"],
        "vca_wash": wash_data["vca"],
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    f0, f1 = CRIMP_VALUES[0], CRIMP_VALUES[-1]
    CRIMP_VTC       = {c: 2.5 + (c - f0) / (f1 - f0) * 5.0        for c in CRIMP_VALUES}
    CRIMP_BASE_FILM = {c: 42e-6 - (c - f0) / (f1 - f0) * 21e-6    for c in CRIMP_VALUES}

    # Geometry maps depend only on crimp — compute cheaply upfront.
    pore_b64   = []
    height_b64 = []
    for crimp in CRIMP_VALUES:
        wp0 = WeaveParameters(grid=MAP_GRID, yarn_crimp=crimp,
                              valley_to_crown_film=CRIMP_VTC[crimp],
                              base_film_m=CRIMP_BASE_FILM[crimp], film_ref_m=30e-6)
        out0 = beading_performance_index(wp0, DryingParameters(phi_large_0=0.06), fast=True)
        pore_b64.append(encode_bool_map(out0["pore_mask"]))
        height_b64.append(encode_map(out0["height"], 0.0, 1.0))

    # Build ordered task list matching run_idx(chi, pi, ei, ci).
    tasks, order = [], []
    for chi, chem_key in enumerate(CHEM_KEYS):
        for pi, phi in enumerate(PHI_VALUES):
            for ei, evap in enumerate(EVAP_VALUES):
                for ci, crimp in enumerate(CRIMP_VALUES):
                    tasks.append((chem_key, phi, evap, crimp,
                                  CRIMP_VTC[crimp], CRIMP_BASE_FILM[crimp],
                                  MAP_GRID, WASH_COUNTS))
                    order.append(run_idx(chi, pi, ei, ci))

    print(f"Running {N_TOTAL} evaluations in parallel (fast=True)…")
    t_start = time.time()
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_compute_run, tasks))
    elapsed = time.time() - t_start
    print(f"Sweep complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    beading_index = [0.0] * N_TOTAL
    crown_ca      = [0.0] * N_TOTAL
    valley_ca     = [0.0] * N_TOTAL
    mean_ca       = [0.0] * N_TOTAL
    coverage_b64  = [""] * N_TOTAL
    ca_raw        = [None] * N_TOTAL
    # Wash curves: flat array indexed [run_idx * N_WASH + wash_level]
    bi_by_wash  = [0.0] * (N_TOTAL * N_WASH)
    cca_by_wash = [0.0] * (N_TOTAL * N_WASH)
    vca_by_wash = [0.0] * (N_TOTAL * N_WASH)

    for res, idx in zip(results, order):
        beading_index[idx] = res["bi"]
        crown_ca[idx]      = res["cca"]
        valley_ca[idx]     = res["vca"]
        mean_ca[idx]       = res["mca"]
        coverage_b64[idx]  = encode_map(res["cov"], 0.0, 1.0)
        ca_raw[idx]        = res["ca"]
        for w in range(N_WASH):
            bi_by_wash [idx * N_WASH + w] = res["bi_wash"][w]
            cca_by_wash[idx * N_WASH + w] = res["cca_wash"][w]
            vca_by_wash[idx * N_WASH + w] = res["vca_wash"][w]

    # Global CA range across all chemistries for a consistent colormap.
    all_ca = np.concatenate([m.ravel() for m in ca_raw])
    ca_min = float(5 * np.floor(np.percentile(all_ca, 2)  / 5))
    ca_max = float(5 * np.ceil (np.percentile(all_ca, 98) / 5))
    print(f"CA colormap range: {ca_min:.0f}° – {ca_max:.0f}°")
    ca_b64 = [encode_map(ca_raw[i], ca_min, ca_max) for i in range(N_TOTAL)]

    return {
        "params": {
            "phi_labels":   PHI_LABELS,
            "evap_labels":  EVAP_LABELS,
            "crimp_labels": CRIMP_LABELS,
            "chem_labels":  CHEM_LABELS,
        },
        "chem_info": [
            {"key":                  k,
             "label":                CHEMISTRY_PROFILES[k].label,
             "intrinsic_ca":         CHEMISTRY_PROFILES[k].intrinsic_ca_deg,
             "r_large_nm":           CHEMISTRY_PROFILES[k].r_large_nm,
             "status":               CHEMISTRY_PROFILES[k].status,
             "description":          CHEMISTRY_PROFILES[k].description,
             "wash_durability":      CHEMISTRY_PROFILES[k].wash_durability_factor}
            for k in CHEM_KEYS
        ],
        "grid_size":     MAP_GRID,
        "ca_min":        ca_min,
        "ca_max":        ca_max,
        "n_phi":         N_PHI,
        "n_evap":        N_EVAP,
        "n_crimp":       N_CRIMP,
        "n_chem":        N_CHEM,
        "n_wash":        N_WASH,
        "wash_counts":   WASH_COUNTS,
        "beading_index": beading_index,
        "crown_ca":      crown_ca,
        "valley_ca":     valley_ca,
        "mean_ca":       mean_ca,
        "bi_by_wash":    bi_by_wash,
        "cca_by_wash":   cca_by_wash,
        "vca_by_wash":   vca_by_wash,
        "coverage_b64":  coverage_b64,
        "ca_b64":        ca_b64,
        "pore_b64":      pore_b64,
        "height_b64":    height_b64,
    }


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DWR Re-Proofer Coating Uniformity Simulator</title>
<style>
:root {
  --bg:      #0d1520;
  --panel:   #162030;
  --border:  #243348;
  --text:    #c8d4e6;
  --dim:     #5a7090;
  --accent:  #4a9eff;
  --warn:    #f59e0b;
  --good:    #34d399;
  --bad:     #f87171;
  --white:   #f0f4fa;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; overflow: hidden; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Fira Code', 'Fira Mono', 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.5;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* Header */
header {
  padding: 8px 28px 6px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
  flex-shrink: 0;
}
header h1 { font-size: 14px; font-weight: 700; color: var(--white); letter-spacing: .03em; }
header .subtitle { color: var(--dim); font-size: 10px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
header .subtitle a { color: var(--accent); text-decoration: none; }

/* Sliders */
.sliders-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px 48px;
  padding: 7px 28px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.slider-row { display: flex; align-items: center; gap: 10px; }
.slider-label { min-width: 160px; color: var(--dim); font-size: 11px; }
.slider-row input[type=range] {
  flex: 1;
  accent-color: var(--accent);
  cursor: pointer;
  height: 4px;
}
.slider-value { min-width: 80px; text-align: right; color: var(--white); font-size: 11px; }

/* Chemistry selector */
.chem-row { display: flex; align-items: center; gap: 10px; grid-column: 1 / -1; }
.chem-btns { display: flex; gap: 6px; flex-wrap: wrap; flex: 1; }
.chem-btn {
  font-family: inherit;
  font-size: 10px;
  padding: 3px 10px;
  border-radius: 3px;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--text);
  cursor: pointer;
  white-space: nowrap;
  transition: border-color .1s, background .1s;
}
.chem-btn:hover { border-color: var(--accent); }
.chem-btn.on { border-color: var(--accent); background: rgba(74,158,255,.12); color: var(--white); }
.chem-info { font-size: 9px; color: var(--dim); padding-left: 4px; }

/* Durability panel */
.durability-panel { min-width: 0; }
.durability-panel .map-title { margin-bottom: 5px; }
#durability-canvas {
  display: block;
  width: 100%;
  height: clamp(155px, 26vh, 240px);
  border: 1px solid var(--border);
  border-radius: 2px;
}
.durability-axes {
  font-size: 9px;
  color: var(--dim);
  margin-top: 5px;
  line-height: 1.8;
}
.wash-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 600;
  padding: 1px 7px;
  border-radius: 3px;
  margin-left: 6px;
}
.wash-badge.good { background: rgba(52,211,153,.15); color: var(--good); }
.wash-badge.warn { background: rgba(245,158,11,.15); color: var(--warn); }
.wash-badge.bad  { background: rgba(248,113,113,.15); color: var(--bad); }

/* Main viz row — flex:1 lets it absorb all remaining vertical space */
.main-viz {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: auto auto 1fr;
  gap: 10px 20px;
  padding: 10px 28px 28px;
  align-items: start;
}
.map-panel {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  --map-sz: min(calc(50vw - 165px), calc(100vh - 560px));
}
.map-title {
  font-size: 10px;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: 6px;
}
canvas.heatmap {
  display: block;
  width: var(--map-sz);
  aspect-ratio: 1 / 1;
  image-rendering: pixelated;
  border: 1px solid var(--border);
  border-radius: 2px;
}
.cbar-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  width: var(--map-sz);
}
.cbar-canvas { flex: 1; min-width: 0; height: 7px; border-radius: 2px; }
.cbar-label { font-size: 9px; color: var(--dim); min-width: 24px; text-align: center; flex-shrink: 0; }
.map-note { font-size: 9px; color: var(--dim); margin-top: 3px; width: var(--map-sz); }

/* Metrics panel */
.metrics-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
}
.metric-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 9px 11px;
}
.metric-card-label {
  font-size: 9px;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: 4px;
}
.big-number {
  font-size: 34px;
  font-weight: 700;
  color: var(--white);
  line-height: 1;
  letter-spacing: -.02em;
}
.gauge-track {
  height: 4px;
  background: var(--border);
  border-radius: 3px;
  margin-top: 6px;
  overflow: hidden;
}
.gauge-fill {
  height: 100%;
  border-radius: 3px;
  transition: width .15s, background-color .15s;
}
.ca-row { display: flex; justify-content: space-between; align-items: center; padding: 3px 0; }
.ca-row + .ca-row { border-top: 1px solid var(--border); margin-top: 4px; padding-top: 6px; }
.ca-key { font-size: 10px; color: var(--dim); }
.ca-val { font-size: 13px; font-weight: 600; transition: color .15s; }

.warn-card {
  border: 1px solid var(--warn);
  border-left: 3px solid var(--warn);
  border-radius: 5px;
  padding: 10px 12px;
  background: rgba(245,158,11,.07);
  font-size: 10px;
  color: var(--warn);
  line-height: 1.6;
  display: none;
}
.warn-card.on { display: block; }
.ok-card {
  border: 1px solid var(--good);
  border-left: 3px solid var(--good);
  border-radius: 5px;
  padding: 10px 12px;
  background: rgba(52,211,153,.07);
  font-size: 10px;
  color: var(--good);
  line-height: 1.6;
  display: none;
}
.ok-card.on { display: block; }

/* Process + durability + finding row */
.bottom-row {
  display: grid;
  grid-template-columns: 290px 290px 1fr;
  gap: 20px;
  padding: 24px 28px 10px;
  border-top: 1px solid rgba(255,255,255,0.07);
  align-items: start;
  flex-shrink: 0;
}
.process-panel { min-width: 0; }
.process-panel .map-title { margin-bottom: 5px; }
#process-canvas {
  display: block;
  width: 100%;
  height: clamp(155px, 26vh, 240px);
  border: 1px solid var(--border);
  border-radius: 2px;
}
.process-axes {
  font-size: 9px;
  color: var(--dim);
  margin-top: 5px;
  line-height: 1.8;
}

/* Finding box */
.finding-box {
  background: var(--panel);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 5px;
  padding: 16px 18px;
  align-self: center;
}
.finding-box h4 {
  font-size: 10px;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: .07em;
  margin-bottom: 10px;
}
.finding-box p { font-size: 10px; color: var(--text); line-height: 1.65; }
.finding-box strong { color: var(--white); }

/* Footer */
footer {
  padding: 5px 28px;
  border-top: 1px solid var(--border);
  font-size: 9px;
  color: var(--dim);
  line-height: 1.4;
  flex-shrink: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
</head>
<body>

<header>
  <h1>DWR Re-Proofer Coating Uniformity Simulator</h1>
  <div class="subtitle">
    Physics-based computational model &nbsp;·&nbsp; qualitative predictions only &nbsp;·&nbsp;
    A. Stubbings (MEng Materials Science) &nbsp;·&nbsp;
    <a href="https://github.com/arthurstubbings/dwr-sim" target="_blank">github.com/arthurstubbings/dwr-sim</a>
  </div>
</header>

<div class="sliders-section">
  <div class="slider-row">
    <span class="slider-label">Polymer loading &phi;&#x2080;</span>
    <input type="range" id="sl-phi"   min="0" max="5" step="1" value="2">
    <span class="slider-value" id="lbl-phi">—</span>
  </div>
  <div class="slider-row">
    <span class="slider-label">Yarn crimp</span>
    <input type="range" id="sl-crimp" min="0" max="2" step="1" value="1">
    <span class="slider-value" id="lbl-crimp">—</span>
  </div>
  <div class="slider-row">
    <span class="slider-label">Drying temperature (approx.)</span>
    <input type="range" id="sl-evap"  min="0" max="5" step="1" value="1">
    <span class="slider-value" id="lbl-evap">—</span>
  </div>
  <div class="slider-row">
    <span class="slider-label">Wash cycles applied</span>
    <input type="range" id="sl-wash" min="0" max="30" step="1" value="0">
    <span class="slider-value" id="lbl-wash">0 washes</span>
  </div>
  <div class="chem-row">
    <span class="slider-label">DWR chemistry</span>
    <div class="chem-btns" id="chem-btns"></div>
    <span class="chem-info" id="chem-info"></span>
  </div>
</div>

<div class="main-viz">

  <div class="map-panel">
    <div class="map-title">Effective DWR Coverage</div>
    <canvas id="cov-canvas" class="heatmap" width="32" height="32"></canvas>
    <div class="cbar-row">
      <span class="cbar-label">0</span>
      <canvas id="cov-cbar" class="cbar-canvas" width="180" height="8"></canvas>
      <span class="cbar-label">1</span>
    </div>
    <div class="map-note">Bright = high coverage (valleys pool more polymer) &nbsp;·&nbsp; grey = pore</div>
  </div>

  <div class="map-panel">
    <div class="map-title">Contact Angle Map (Cassie–Baxter)</div>
    <canvas id="ca-canvas" class="heatmap" width="32" height="32"></canvas>
    <div class="cbar-row">
      <span class="cbar-label" id="ca-cmin">—</span>
      <canvas id="ca-cbar" class="cbar-canvas" width="180" height="8"></canvas>
      <span class="cbar-label" id="ca-cmax">—</span>
    </div>
    <div class="map-note">Blue &lt; 90° (wets) &nbsp;·&nbsp; red &gt; 90° (beads) &nbsp;·&nbsp; grey = open pore</div>
  </div>

  <div class="metrics-col">

    <div class="metric-card">
      <div class="metric-card-label">Beading Index</div>
      <div class="big-number" id="bi-val">—</div>
      <div class="gauge-track">
        <div class="gauge-fill" id="bi-gauge"></div>
      </div>
      <div style="font-size:9px;color:var(--dim);margin-top:5px">
        Crown-weighted [0–1] &nbsp;·&nbsp; relative metric
      </div>
    </div>

    <div class="metric-card">
      <div class="metric-card-label">Contact Angles</div>
      <div class="ca-row">
        <span class="ca-key">Crown &nbsp;<em style="font-size:9px">(droplet rests here)</em></span>
        <span class="ca-val" id="crown-val">—</span>
      </div>
      <div class="ca-row">
        <span class="ca-key">Valley &nbsp;<em style="font-size:9px">(interstice)</em></span>
        <span class="ca-val" id="valley-val">—</span>
      </div>
      <div class="ca-row">
        <span class="ca-key">Mean</span>
        <span class="ca-val" id="mean-val" style="font-size:11px;color:var(--dim)">—</span>
      </div>
    </div>

    <div class="warn-card" id="warn-card-severe">
      ⚠&nbsp; <strong>Crown starvation</strong><br>
      Crown CA &lt; 90° — crowns wet<br>
      Polymer pools in valleys,<br>
      starving surfaces a drop rests on
    </div>

    <div class="warn-card" id="warn-card-mild" style="border-color:var(--dim);background:rgba(90,112,144,.07);color:var(--dim)">
      ↕&nbsp; Uneven coating<br>
      Crown CA &lt; Valley CA<br>
      Both bead, but polymer skewed<br>
      toward interstices
    </div>

    <div class="ok-card" id="ok-card">
      ✓&nbsp; Crowns adequately coated<br>
      Crown CA ≥ Valley CA — uniform
    </div>

  </div>
</div>

<div class="bottom-row">

  <div class="process-panel">
    <div class="map-title">Process Window &nbsp;(beading index)</div>
    <canvas id="process-canvas" width="580" height="460"></canvas>
    <div class="process-axes">
      &rarr;&nbsp; Drying temperature  20 °C → 80 °C &nbsp;&nbsp; &times; = current params<br>
      &uarr;&nbsp; Polymer loading (dilute → concentrated)
    </div>
  </div>

  <div class="durability-panel">
    <div class="map-title">Wash Durability &nbsp;<span id="wash-to-reproof-badge"></span></div>
    <canvas id="durability-canvas" width="580" height="460"></canvas>
    <div class="durability-axes">
      &rarr;&nbsp; Wash cycles (0 – 30) &nbsp;&nbsp; — = re-proof threshold<br>
      &uarr;&nbsp; Beading index &nbsp;&nbsp; | = current wash count
    </div>
  </div>

  <div class="finding-box">
    <h4>Key Finding — Pooling Starves the Crowns</h4>
    <p>
      Capillary pressure pulls the re-proofer dispersion into the inter-yarn
      interstices, leaving a locally thicker residual film there than on the
      exposed yarn crowns. Thicker films have a <strong>higher local film
      Péclet number</strong>: the evaporative flux buries the functional polymer
      in the deposit rather than leaving it at the air surface where it lowers
      the contact angle a raindrop feels.
      <br><br>
      The result: <strong>repellent pools where it is not needed and starves the
      yarn crowns</strong> — the very surfaces a raindrop actually rests on.
      Crown CA &lt; 90° means the fabric wets despite carrying DWR.
      <br><br>
      The process window (adequate loading + slow/cool drying) keeps local
      Péclet numbers low enough for diffusion to re-homogenise the deposit before
      the film collapses.
    </p>
  </div>

</div>

<footer>
  Computational model — predictive, not experimental &nbsp;·&nbsp;
  Physics: Routh/Russel Péclet framework &nbsp;·&nbsp; Fortini binary stratification &nbsp;·&nbsp;
  Cassie–Baxter wetting &nbsp;·&nbsp; Stokes–Einstein diffusion
  &nbsp;·&nbsp; Beading index is a relative metric, not an absolute contact-angle prediction
</footer>

<script>
// ── Baked data ────────────────────────────────────────────────────────────────
const DATA = __BAKED_DATA__;

// ── Colormaps (simplified; 5-stop linear interpolation) ──────────────────────
const MAGMA = [
  [0.00, [12,  2,  26]],
  [0.25, [81, 18, 124]],
  [0.50, [176, 65,  78]],
  [0.75, [247,149,  32]],
  [1.00, [252,253, 191]],
];
// Diverging: blue (low CA, wets) → white (90°) → red (high CA, beads)
const DIVBR = [
  [0.00, [ 50,  90, 200]],
  [0.45, [140, 175, 230]],
  [0.50, [230, 230, 230]],
  [0.55, [230, 160, 140]],
  [1.00, [200,  50,  50]],
];
// YlGnBu for process window (low=yellow, high=dark blue)
const YLGNBU = [
  [0.00, [255,255,204]],
  [0.25, [161,218,180]],
  [0.50, [ 65,182,196]],
  [0.75, [ 34, 94,168]],
  [1.00, [  8, 29, 88]],
];

function cmap(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const lo = stops[i-1], hi = stops[i];
      const f = (t - lo[0]) / (hi[0] - lo[0]);
      return [
        Math.round(lo[1][0] + f*(hi[1][0]-lo[1][0])),
        Math.round(lo[1][1] + f*(hi[1][1]-lo[1][1])),
        Math.round(lo[1][2] + f*(hi[1][2]-lo[1][2])),
      ];
    }
  }
  return stops[stops.length-1][1];
}

function b64ToU8(s) {
  const bin = atob(s);
  const a = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) a[i]=bin.charCodeAt(i);
  return a;
}

function paintHeatmap(canvas, u8, colorStops, pore) {
  const ctx = canvas.getContext('2d');
  const n = canvas.width;
  const img = ctx.createImageData(n, n);
  const PORE_RGB = [25, 32, 42];
  for (let i=0;i<n*n;i++) {
    let rgb;
    if (pore && pore[i]) rgb = PORE_RGB;
    else rgb = cmap(colorStops, u8[i]/255);
    img.data[i*4]   = rgb[0];
    img.data[i*4+1] = rgb[1];
    img.data[i*4+2] = rgb[2];
    img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function paintColorbar(canvas, colorStops) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  for (let x=0;x<w;x++) {
    const rgb = cmap(colorStops, x/(w-1));
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(x, 0, 1, h);
  }
}

// ── State ─────────────────────────────────────────────────────────────────────
const NP=DATA.n_phi, NE=DATA.n_evap, NC=DATA.n_crimp, NCH=DATA.n_chem;
const NW=DATA.n_wash, WASH_COUNTS=DATA.wash_counts;
let iP=2, iE=1, iC=1, iCh=0, iWash=0;

function flatIdx(p,e,c,ch){ return ch*NP*NE*NC + p*NE*NC + e*NC + c; }

const STATUS_COLOR = {
  "PFAS — banned / phased out": "#f87171",
  "PFAS — under regulation":    "#fbbf24",
  "PFC-free":                        "#34d399",
};

// ── Wash helpers ─────────────────────────────────────────────────────────────
// Interpolate a precomputed wash curve at any wash value 0-30.
function washInterp(flatCurve, runIdx, washVal) {
  for (let i = 1; i < NW; i++) {
    if (washVal <= WASH_COUNTS[i]) {
      const lo = flatCurve[runIdx * NW + i - 1];
      const hi = flatCurve[runIdx * NW + i];
      const f  = (washVal - WASH_COUNTS[i-1]) / (WASH_COUNTS[i] - WASH_COUNTS[i-1]);
      return lo + f * (hi - lo);
    }
  }
  return flatCurve[runIdx * NW + NW - 1];
}

// Washes at which BI first drops below threshold (linear interp between stored points).
// Returns null if never drops below, -1 if already below at wash=0.
function washesToThreshold(runIdx, threshold) {
  if (DATA.bi_by_wash[runIdx * NW] < threshold) return -1;
  for (let i = 1; i < NW; i++) {
    const bi = DATA.bi_by_wash[runIdx * NW + i];
    if (bi < threshold) {
      const bi_prev = DATA.bi_by_wash[runIdx * NW + i - 1];
      const f = (threshold - bi_prev) / (bi - bi_prev);
      return WASH_COUNTS[i-1] + f * (WASH_COUNTS[i] - WASH_COUNTS[i-1]);
    }
  }
  return null;
}

// Apply wash degradation to a coverage uint8 array (client-side, per-pixel).
const CROWN_ABRASION = 1.5;
const BARE_CA_DEG    = 65.0;
const BARE_CA_COS    = Math.cos(BARE_CA_DEG * Math.PI / 180);

function applyWashToCoverage(covU8, heightU8, poreU8, durability, nWashes) {
  const n = covU8.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (poreU8[i]) { out[i] = 0; continue; }
    const h = heightU8[i] / 255;
    const loss = durability * (1.0 + CROWN_ABRASION * h);
    const retained = Math.pow(Math.max(0, 1.0 - Math.min(loss, 0.99)), nWashes);
    out[i] = Math.round(covU8[i] * retained);
  }
  return out;
}

function recomputeCAMap(covU8, poreU8, intrinsicCA, caMin, caMax) {
  const cosDWR = Math.cos(intrinsicCA * Math.PI / 180);
  const n = covU8.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (poreU8[i]) { out[i] = 0; continue; }
    const cov = covU8[i] / 255;
    const cosEff = Math.max(-1, Math.min(1, cov * cosDWR + (1 - cov) * BARE_CA_COS));
    const ca = Math.acos(cosEff) * 180 / Math.PI;
    out[i] = Math.round(255 * Math.max(0, Math.min(1, (ca - caMin) / (caMax - caMin))));
  }
  return out;
}

// ── Durability sparkline ──────────────────────────────────────────────────────
function paintDurabilitySparkline(runIdx, currentWash) {
  const canvas = document.getElementById('durability-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const THRESHOLD = 0.4;
  const maxWash = WASH_COUNTS[NW - 1];

  // Background
  ctx.fillStyle = 'rgba(22,32,48,0.7)';
  ctx.fillRect(0, 0, W, H);

  // Threshold line
  const ty = H * (1 - THRESHOLD);
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = 'rgba(245,158,11,0.6)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
  ctx.setLineDash([]);

  // Threshold label
  ctx.fillStyle = 'rgba(245,158,11,0.7)';
  ctx.font = '8px monospace';
  ctx.fillText('re-proof', 2, ty - 3);

  // BI curve
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = '#4a9eff';
  ctx.beginPath();
  for (let i = 0; i < NW; i++) {
    const bi = DATA.bi_by_wash[runIdx * NW + i];
    const x = (WASH_COUNTS[i] / maxWash) * W;
    const y = H * (1 - bi);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Current wash marker
  const cx = (currentWash / maxWash) * W;
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();

  // Dot at current BI
  const curBI = washInterp(DATA.bi_by_wash, runIdx, currentWash);
  const cy = H * (1 - curBI);
  ctx.beginPath();
  ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
  ctx.fillStyle = curBI >= 0.65 ? '#34d399' : curBI >= 0.4 ? '#f59e0b' : '#f87171';
  ctx.fill();

  // Axis ticks
  ctx.fillStyle = 'rgba(90,112,144,0.8)';
  ctx.font = '8px monospace';
  ctx.fillText('0', 2, H - 2);
  ctx.fillText('30', W - 16, H - 2);
  ctx.fillText('1.0', 2, 10);
  ctx.fillText('0', 2, H * 0.95);
}

// ── Process window ────────────────────────────────────────────────────────────
// Bilinear interpolation to a finer grid for a smoother look.
function paintProcessWindow(iP, iE, iC, iCh) {
  const canvas = document.getElementById('process-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const img = ctx.createImageData(W, H);

  for (let py=0; py<H; py++) {
    for (let px=0; px<W; px++) {
      // Map pixel to fractional grid position
      const fx = px / (W-1) * (NE-1);  // evap axis
      const fy = py / (H-1) * (NP-1);  // phi axis (0=top in canvas=highest phi)
      const phiFrac = (NP-1) - fy;     // invert so high phi is at top

      const e0 = Math.max(0, Math.min(NE-2, Math.floor(fx)));
      const e1 = e0 + 1;
      const p0 = Math.max(0, Math.min(NP-2, Math.floor(phiFrac)));
      const p1 = p0 + 1;
      const ef = fx - e0;
      const pf = phiFrac - p0;

      const v00 = DATA.beading_index[flatIdx(p0,e0,iC,iCh)];
      const v10 = DATA.beading_index[flatIdx(p1,e0,iC,iCh)];
      const v01 = DATA.beading_index[flatIdx(p0,e1,iC,iCh)];
      const v11 = DATA.beading_index[flatIdx(p1,e1,iC,iCh)];
      const val = v00*(1-ef)*(1-pf) + v01*ef*(1-pf) + v10*(1-ef)*pf + v11*ef*pf;

      // Viridis-like: purple (low) → teal → green → yellow (high)
      const rgb = cmap(YLGNBU, 1 - val);
      const i4 = (py*W + px)*4;
      img.data[i4]   = rgb[0];
      img.data[i4+1] = rgb[1];
      img.data[i4+2] = rgb[2];
      img.data[i4+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Cursor at current (iP, iE)
  const cx = iE/(NE-1)*W;
  const cy = (1 - iP/(NP-1))*H;
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  const s = 5;
  ctx.beginPath();
  ctx.moveTo(cx-s, cy); ctx.lineTo(cx+s, cy);
  ctx.moveTo(cx, cy-s); ctx.lineTo(cx, cy+s);
  ctx.stroke();
  // Circle around cursor
  ctx.beginPath();
  ctx.arc(cx, cy, 4.5, 0, 2*Math.PI);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ── Update ────────────────────────────────────────────────────────────────────
function update() {
  const idx = flatIdx(iP, iE, iC, iCh);
  // Use wash-interpolated metrics when wash > 0.
  const bi  = iWash === 0 ? DATA.beading_index[idx] : washInterp(DATA.bi_by_wash,  idx, iWash);
  const cca = iWash === 0 ? DATA.crown_ca[idx]      : washInterp(DATA.cca_by_wash, idx, iWash);
  const vca = iWash === 0 ? DATA.valley_ca[idx]     : washInterp(DATA.vca_by_wash, idx, iWash);
  const mca = DATA.mean_ca[idx];

  // Metrics
  document.getElementById('bi-val').textContent = bi.toFixed(2);
  const fill = document.getElementById('bi-gauge');
  fill.style.width = (bi*100)+'%';
  fill.style.backgroundColor = bi>=0.65 ? 'var(--good)' : bi>=0.35 ? 'var(--warn)' : 'var(--bad)';

  const crownEl  = document.getElementById('crown-val');
  const valleyEl = document.getElementById('valley-val');
  const meanEl   = document.getElementById('mean-val');
  crownEl.textContent  = cca.toFixed(1)+'°';
  valleyEl.textContent = vca.toFixed(1)+'°';
  meanEl.textContent   = mca.toFixed(1)+'°';
  crownEl.style.color  = cca>=90 ? 'var(--good)' : 'var(--bad)';
  valleyEl.style.color = vca>=90 ? 'var(--good)' : 'var(--bad)';

  // Warning cards — three-tier logic
  const crownBeads    = cca >= 90;
  const unevenCoating = crownBeads && (cca < vca - 1.5);
  const severeStarve  = !crownBeads && (cca < vca - 1.0);
  const uniform       = crownBeads && !unevenCoating;
  document.getElementById('warn-card-severe').className = 'warn-card' + (severeStarve  ? ' on' : '');
  document.getElementById('warn-card-mild').className   = 'warn-card' + (unevenCoating ? ' on' : '');
  document.getElementById('ok-card').className          = 'ok-card'   + (uniform       ? ' on' : '');

  // Slider labels
  document.getElementById('lbl-phi').textContent   = DATA.params.phi_labels[iP];
  document.getElementById('lbl-evap').textContent  = DATA.params.evap_labels[iE];
  document.getElementById('lbl-crimp').textContent = DATA.params.crimp_labels[iC];
  document.getElementById('lbl-wash').textContent  = iWash === 1 ? '1 wash' : `${iWash} washes`;

  // Chemistry info badge
  const ch = DATA.chem_info[iCh];
  const sc = STATUS_COLOR[ch.status] || 'var(--dim)';
  const infoEl = document.getElementById('chem-info');
  infoEl.textContent = `θ = ${ch.intrinsic_ca.toFixed(0)}°  ·  rₗ = ${ch.r_large_nm.toFixed(0)} nm  ·  ${ch.status}`;
  infoEl.style.color = sc;

  // CA colorbar labels
  document.getElementById('ca-cmin').textContent = DATA.ca_min.toFixed(0)+'°';
  document.getElementById('ca-cmax').textContent = DATA.ca_max.toFixed(0)+'°';

  // Maps — apply wash degradation client-side
  const poreU8   = b64ToU8(DATA.pore_b64[iC]);
  const heightU8 = b64ToU8(DATA.height_b64[iC]);
  let covU8 = b64ToU8(DATA.coverage_b64[idx]);
  let caU8  = b64ToU8(DATA.ca_b64[idx]);
  if (iWash > 0) {
    covU8 = applyWashToCoverage(covU8, heightU8, poreU8, ch.wash_durability, iWash);
    caU8  = recomputeCAMap(covU8, poreU8, ch.intrinsic_ca, DATA.ca_min, DATA.ca_max);
  }
  paintHeatmap(document.getElementById('cov-canvas'), covU8, MAGMA, poreU8);
  paintHeatmap(document.getElementById('ca-canvas'),  caU8,  DIVBR, poreU8);

  // Durability sparkline + "washes to re-proof" badge
  const reproof = washesToThreshold(idx, 0.4);
  const badgeEl = document.getElementById('wash-to-reproof-badge');
  if (reproof === null) {
    badgeEl.textContent = '> 30 washes';
    badgeEl.className = 'wash-badge good';
  } else if (reproof < 0) {
    badgeEl.textContent = 'poor initial coverage';
    badgeEl.className = 'wash-badge bad';
  } else {
    const n = Math.round(reproof);
    badgeEl.textContent = `re-proof after ~${n} wash${n===1?'':'es'}`;
    badgeEl.className = 'wash-badge ' + (n >= 20 ? 'good' : n >= 10 ? 'warn' : 'bad');
  }
  paintDurabilitySparkline(idx, iWash);

  paintProcessWindow(iP, iE, iC, iCh);
}

// ── Wire sliders ──────────────────────────────────────────────────────────────
document.getElementById('sl-phi').addEventListener('input',   function(){ iP=+this.value; update(); });
document.getElementById('sl-evap').addEventListener('input',  function(){ iE=+this.value; update(); });
document.getElementById('sl-crimp').addEventListener('input', function(){ iC=+this.value; update(); });
document.getElementById('sl-wash').addEventListener('input',  function(){ iWash=+this.value; update(); });

// ── Init ──────────────────────────────────────────────────────────────────────
paintColorbar(document.getElementById('cov-cbar'), MAGMA);
paintColorbar(document.getElementById('ca-cbar'),  DIVBR);

// Build chemistry pill buttons
const chemBtnsEl = document.getElementById('chem-btns');
DATA.chem_info.forEach(function(ch, i) {
  const btn = document.createElement('button');
  btn.className = 'chem-btn' + (i === 0 ? ' on' : '');
  btn.textContent = ch.label;
  btn.dataset.idx = i;
  btn.addEventListener('click', function() {
    iCh = +this.dataset.idx;
    document.querySelectorAll('.chem-btn').forEach(function(b){ b.classList.remove('on'); });
    this.classList.add('on');
    update();
  });
  chemBtnsEl.appendChild(btn);
});

update();
</script>
</body>
</html>
"""


# ── Assemble & write ──────────────────────────────────────────────────────────

def build_dashboard():
    data = run_sweep()

    # Serialise with compact float formatting.
    data_json = json.dumps(data, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__BAKED_DATA__", data_json)

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTFILE) / 1024
    print(f"\nWrote {OUTFILE}  ({size_kb:.0f} KB)")
    print("Open in a browser — no server needed.")


if __name__ == "__main__":
    build_dashboard()
