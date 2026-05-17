"""
make_dashboard.py
=================
Precomputes a parameter sweep over the DWR coating model and bakes the
results into a single self-contained interactive HTML file.

Run from src/:  python make_dashboard.py
Output:         ../dashboard.html   (open in any browser; no server needed)

Sweep grid: 6 polymer loadings × 6 drying rates × 3 yarn crimps × 3
particle sizes = 324 model evaluations. Each stores a 32×32 coverage map
and contact-angle map encoded as uint8 base64. Total HTML file ~2-3 MB.

The HTML file works offline, requires no running server, and is hostable
on GitHub Pages. Sliders index the precomputed grid — instant response.
"""

import os
import sys
import json
import time
import base64
import numpy as np

# Run from src/
sys.path.insert(0, os.path.dirname(__file__))
from drying_1d import DryingParameters
from weave_cell import WeaveParameters, beading_performance_index

OUTDIR = os.path.join(os.path.dirname(__file__), "..")
OUTFILE = os.path.join(OUTDIR, "dashboard.html")
MAP_GRID = 32   # spatial resolution of each stored heatmap

# ── Parameter grid ────────────────────────────────────────────────────────────

PHI_VALUES = [0.020, 0.040, 0.060, 0.085, 0.105, 0.130]
PHI_LABELS = ["2 %", "4 %", "6 %", "8.5 %", "10.5 %", "13 %"]

EVAP_VALUES = list(np.geomspace(2e-8, 8e-7, 6))
EVAP_LABELS = [f"{v*1e8:.1f}×10⁻⁸" for v in EVAP_VALUES]
# More readable labels
EVAP_LABELS = ["2e-8 m/s\n(very slow)", "6e-8 m/s", "2e-7 m/s",
               "6e-7 m/s", "1e-6 m/s", "8e-7 m/s\n(fast)"]
# Recompute cleanly
EVAP_LABELS = []
for v in EVAP_VALUES:
    if v < 1e-7:
        EVAP_LABELS.append(f"{v*1e8:.1f} ×10⁻⁸ m/s")
    else:
        EVAP_LABELS.append(f"{v*1e7:.1f} ×10⁻⁷ m/s")

CRIMP_VALUES = [0.30, 0.50, 0.70]
CRIMP_LABELS = ["Low — even film", "Medium — pooling", "High — crown drainage"]

RADIUS_VALUES = [70e-9, 110e-9, 160e-9]   # r_large_m
RADIUS_LABELS = ["70 nm", "110 nm", "160 nm"]

N_PHI   = len(PHI_VALUES)
N_EVAP  = len(EVAP_VALUES)
N_CRIMP = len(CRIMP_VALUES)
N_RAD   = len(RADIUS_VALUES)
N_TOTAL = N_PHI * N_EVAP * N_CRIMP * N_RAD


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_idx(pi, ei, ci, ri):
    return pi * N_EVAP * N_CRIMP * N_RAD + ei * N_CRIMP * N_RAD + ci * N_RAD + ri


def encode_map(arr2d, vmin, vmax):
    """Scale arr2d to uint8 and return base64 string."""
    scaled = np.clip((arr2d - vmin) / (vmax - vmin + 1e-30), 0, 1)
    uint8 = (scaled * 255).astype(np.uint8)
    return base64.b64encode(uint8.tobytes()).decode("ascii")


def encode_bool_map(bool2d):
    """Encode a boolean mask as uint8 base64 (True → 1)."""
    return base64.b64encode(bool2d.astype(np.uint8).tobytes()).decode("ascii")


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep():
    print(f"Running {N_TOTAL} model evaluations (grid={MAP_GRID}, fast=True)…")
    t_start = time.time()

    beading_index = [0.0] * N_TOTAL
    crown_ca      = [0.0] * N_TOTAL
    valley_ca     = [0.0] * N_TOTAL
    mean_ca       = [0.0] * N_TOTAL
    coverage_b64  = [""] * N_TOTAL
    ca_b64        = [""] * N_TOTAL

    # Pore mask and height map depend only on crimp (geometry).
    pore_b64   = [""] * N_CRIMP
    height_b64 = [""] * N_CRIMP

    # We need a global CA min/max for a consistent colormap. Collect CA maps
    # first as float arrays, then encode after computing the global range.
    ca_raw = [None] * N_TOTAL

    # Higher crimp = deeper channels = more pooling AND more crown drainage.
    # valley_to_crown_film: 2.5 (low) → 7.5 (high) — deeper interstices hold more.
    # base_film_m: 42µm (low) → 21µm (high) — polymer drains away from crowns.
    # Together they conserve approximate total fluid (mean film thickness ≈ const).
    f0, f1 = CRIMP_VALUES[0], CRIMP_VALUES[-1]
    CRIMP_VTC      = {c: 2.5 + (c - f0) / (f1 - f0) * 5.0 for c in CRIMP_VALUES}
    CRIMP_BASE_FILM = {c: (42e-6 - (c - f0) / (f1 - f0) * 21e-6) for c in CRIMP_VALUES}
    # Results: vtc=2.5/base=42µm (low), vtc=5.0/base=31.5µm (mid), vtc=7.5/base=21µm (high)

    count = 0
    for ci, crimp in enumerate(CRIMP_VALUES):
        for ri, r_large in enumerate(RADIUS_VALUES):
            for pi, phi in enumerate(PHI_VALUES):
                for ei, evap in enumerate(EVAP_VALUES):
                    wp = WeaveParameters(grid=MAP_GRID, yarn_crimp=crimp,
                                        valley_to_crown_film=CRIMP_VTC[crimp],
                                        base_film_m=CRIMP_BASE_FILM[crimp],
                                        film_ref_m=30e-6)
                    bp = DryingParameters(
                        evap_velocity_m_s=evap,
                        phi_large_0=phi,
                        r_large_m=r_large,
                    )
                    out = beading_performance_index(wp, bp, fast=True)

                    idx = run_idx(pi, ei, ci, ri)
                    beading_index[idx] = round(float(out["beading_index"]), 3)
                    crown_ca[idx]      = round(float(out["crown_contact_angle_deg"]), 1)
                    valley_ca[idx]     = round(float(out["valley_contact_angle_deg"]), 1)
                    mean_ca[idx]       = round(float(out["mean_contact_angle_deg"]), 1)
                    coverage_b64[idx]  = encode_map(out["coverage"], 0.0, 1.0)
                    ca_raw[idx]        = out["contact_angle"].copy()

                    # Store geometry once per crimp (at any phi/evap/radius).
                    if pi == 0 and ei == 0 and ri == 0:
                        pore_b64[ci]   = encode_bool_map(out["pore_mask"])
                        height_b64[ci] = encode_map(out["height"], 0.0, 1.0)

                    count += 1
                    if count % 36 == 0:
                        elapsed = time.time() - t_start
                        rate = count / elapsed
                        eta = (N_TOTAL - count) / rate
                        print(f"  {count}/{N_TOTAL}  ({elapsed:.0f}s elapsed, "
                              f"~{eta:.0f}s remaining)")

    # Global CA range for consistent colormap.
    valid_ca = [m.ravel() for m in ca_raw if m is not None]
    all_ca = np.concatenate(valid_ca)
    ca_min = float(np.percentile(all_ca, 2))
    ca_max = float(np.percentile(all_ca, 98))
    # Round to nearest 5° for cleaner labels.
    ca_min = 5 * np.floor(ca_min / 5)
    ca_max = 5 * np.ceil(ca_max / 5)
    print(f"CA colormap range: {ca_min:.0f}° – {ca_max:.0f}°")

    for idx in range(N_TOTAL):
        if ca_raw[idx] is not None:
            ca_b64[idx] = encode_map(ca_raw[idx], ca_min, ca_max)

    elapsed = time.time() - t_start
    print(f"Sweep complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return {
        "params": {
            "phi_values":   PHI_VALUES,
            "phi_labels":   PHI_LABELS,
            "evap_values":  EVAP_VALUES,
            "evap_labels":  EVAP_LABELS,
            "crimp_values": CRIMP_VALUES,
            "crimp_labels": CRIMP_LABELS,
            "radius_values": [float(v) for v in RADIUS_VALUES],
            "radius_labels": RADIUS_LABELS,
        },
        "grid_size":     MAP_GRID,
        "ca_min":        ca_min,
        "ca_max":        ca_max,
        "n_phi":         N_PHI,
        "n_evap":        N_EVAP,
        "n_crimp":       N_CRIMP,
        "n_rad":         N_RAD,
        "beading_index": beading_index,
        "crown_ca":      crown_ca,
        "valley_ca":     valley_ca,
        "mean_ca":       mean_ca,
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

/* Process + finding row */
.bottom-row {
  display: grid;
  grid-template-columns: 290px 1fr;
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
    <span class="slider-label">Drying rate v<sub>evap</sub></span>
    <input type="range" id="sl-evap"  min="0" max="5" step="1" value="1">
    <span class="slider-value" id="lbl-evap">—</span>
  </div>
  <div class="slider-row">
    <span class="slider-label">Polymer particle radius r<sub>L</sub></span>
    <input type="range" id="sl-rad"   min="0" max="2" step="1" value="1">
    <span class="slider-value" id="lbl-rad">—</span>
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
    <canvas id="process-canvas" width="60" height="40"></canvas>
    <div class="process-axes">
      &rarr;&nbsp; Drying rate (slow → fast) &nbsp;&nbsp; &times; = current params<br>
      &uarr;&nbsp; Polymer loading (dilute → concentrated)
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
const NP=DATA.n_phi, NE=DATA.n_evap, NC=DATA.n_crimp, NR=DATA.n_rad;
let iP=2, iE=1, iC=1, iR=1;

function flatIdx(p,e,c,r){ return p*NE*NC*NR + e*NC*NR + c*NR + r; }

// ── Process window ────────────────────────────────────────────────────────────
// Bilinear interpolation to a finer grid for a smoother look.
function paintProcessWindow(iP, iE, iC, iR) {
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

      const v00 = DATA.beading_index[flatIdx(p0,e0,iC,iR)];
      const v10 = DATA.beading_index[flatIdx(p1,e0,iC,iR)];
      const v01 = DATA.beading_index[flatIdx(p0,e1,iC,iR)];
      const v11 = DATA.beading_index[flatIdx(p1,e1,iC,iR)];
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
  const idx = flatIdx(iP, iE, iC, iR);
  const bi  = DATA.beading_index[idx];
  const cca = DATA.crown_ca[idx];
  const vca = DATA.valley_ca[idx];
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
  document.getElementById('lbl-rad').textContent   = DATA.params.radius_labels[iR];

  // CA colorbar labels
  document.getElementById('ca-cmin').textContent = DATA.ca_min.toFixed(0)+'°';
  document.getElementById('ca-cmax').textContent = DATA.ca_max.toFixed(0)+'°';

  // Maps
  const pore = b64ToU8(DATA.pore_b64[iC]);
  paintHeatmap(document.getElementById('cov-canvas'), b64ToU8(DATA.coverage_b64[idx]), MAGMA, pore);
  paintHeatmap(document.getElementById('ca-canvas'),  b64ToU8(DATA.ca_b64[idx]),       DIVBR, pore);

  paintProcessWindow(iP, iE, iC, iR);
}

// ── Wire sliders ──────────────────────────────────────────────────────────────
document.getElementById('sl-phi').addEventListener('input',   function(){ iP=+this.value; update(); });
document.getElementById('sl-evap').addEventListener('input',  function(){ iE=+this.value; update(); });
document.getElementById('sl-crimp').addEventListener('input', function(){ iC=+this.value; update(); });
document.getElementById('sl-rad').addEventListener('input',   function(){ iR=+this.value; update(); });

// ── Init ──────────────────────────────────────────────────────────────────────
paintColorbar(document.getElementById('cov-cbar'), MAGMA);
paintColorbar(document.getElementById('ca-cbar'),  DIVBR);
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
