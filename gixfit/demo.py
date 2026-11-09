"""Generate a self-contained synthetic tutorial, not experimental validation.

Usage: python -m gixfit.demo output_folder [--run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from .crystal import Cell, Structure, write_cif
from .engine import Settings, Session, check_cif, export
from .pattern import fiber_pattern, group_reflections, group_intensity


def create_demo(out_dir: str) -> tuple[str, str, Settings]:
    folder = Path(out_dir).resolve() / "gixfit_synthetic_tutorial"
    folder.mkdir(parents=True, exist_ok=True)
    s = Settings(x0=490, y0=640, alpha_i_deg=0.12, q_min=0.35, q_max=1.85,
                 planes_maxidx=1, n_null=0, workers=1)
    rng = np.random.default_rng(42)
    # Invented P1 model, with no external structure/data dependencies.
    model = Structure(Cell(9.1, 11.3, 13.2, 82, 98, 104), ["C"] * 12,
                      rng.random((12, 3)), name="Synthetic tutorial")
    cif = folder / "candidate.cif"
    write_cif(model, str(cif), "Synthetic tutorial")
    geometry = s.geometry()
    pattern = fiber_pattern(model, (1, 0, 0), qmax=1.95)
    col, row = geometry.q_to_pixel(pattern.qxy, pattern.qz)
    visible = (np.isfinite(col) & (col > 30) & (col < 950) & (row > 30) & (row < 590)
               & (pattern.q > s.q_min) & (pattern.q < s.q_max))
    groups = group_reflections(pattern, 0.008, 0.003, visible)
    intensity = group_intensity(pattern, groups)
    image = np.full((700, 981), 100.0)
    rr, cc = np.mgrid[:image.shape[0], :image.shape[1]]
    for j in np.argsort(-intensity)[:40]:
        qx, qz = groups[j].position(pattern)
        x, y = geometry.q_to_pixel(qx, qz)
        amp = 1000 + 5000 * intensity[j] / intensity.max()
        image += amp * np.exp(-((cc - x) ** 2 + (rr - y) ** 2) / (2 * 1.7 ** 2))
    image = rng.poisson(image).astype(np.float32)
    image_path = folder / "synthetic_th0.120_image.tiff"
    tifffile.imwrite(image_path, image)
    s.to_toml(folder / "settings.toml", image_path.name)
    (folder / "README.txt").write_text(
        "SYNTHETIC TUTORIAL: invented P1 carbon model, generated with gixfit.\n"
        "Not experimental data or independent validation.\n"
        "Load settings.toml, Find spots, Check one CIF with candidate.cif and contact plane auto.\n"
        "The generating plane is (1 0 0). Compare fitted lengths with 9.1, 11.3, 13.2 A.\n"
        "Use Export report to save a portable image/model/recipe bundle.\n", encoding="utf-8")
    return str(image_path), str(cif), s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    image, cif, settings = create_demo(args.output)
    print(f"Tutorial created beside {image}")
    if args.run:
        session = Session(image, settings, log=print)
        result = check_cif(session, cif, log=print)
        export(session, [result], [], str(Path(image).parent / "report"), log=print)


if __name__ == "__main__":
    main()
