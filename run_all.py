"""run_all.py — run tests then regenerate all figures from the repo root."""
import subprocess, sys, pathlib

root = pathlib.Path(__file__).parent
src  = root / "src"

steps = [
    ("Tests: drying_1d",  [sys.executable, "test_drying_1d.py"],  src),
    ("Tests: weave_cell", [sys.executable, "test_weave_cell.py"],  src),
    ("Figures (fig1–5)",  [sys.executable, "make_figures.py"],     src),
]

for label, cmd, cwd in steps:
    print(f"\n── {label} ──")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("\n✓ All steps complete. Open index.html in a browser for the dashboard.")
