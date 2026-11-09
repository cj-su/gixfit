"""Portable report recipes. Run: python -m gixfit.replay report/run_manifest.json output_folder."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shutil

import numpy as np

from . import __version__
from .crystal import Cell, Structure


def checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_record(struct: Structure) -> dict:
    return dict(cell=struct.cell.array().tolist(), species=list(struct.species),
                frac=struct.frac.tolist(), occ=None if struct.occ is None else struct.occ.tolist(),
                b_iso=struct.b_iso, name=struct.name,
                symops=None if struct.symops is None else [[r.tolist(), t.tolist()] for r, t in struct.symops])


def restore_structure(record: dict) -> Structure:
    r = dict(record)
    r["cell"] = Cell.from_array(r["cell"])
    r["frac"] = np.asarray(r["frac"], dtype=float).reshape(-1, 3)
    if r["occ"] is not None:
        r["occ"] = np.asarray(r["occ"], dtype=float)
    if r["symops"] is not None:
        r["symops"] = [(np.asarray(a), np.asarray(b)) for a, b in r["symops"]]
    return Structure(**r)


def environment() -> dict:
    deps = {}
    for package in ("numpy", "scipy", "pymatgen", "tifffile", "matplotlib", "pandas", "openpyxl"):
        try:
            deps[package] = version(package)
        except PackageNotFoundError:
            deps[package] = "unavailable"
    return dict(gixfit=__version__, python=platform.python_version(), platform=platform.platform(), packages=deps)


def write_manifest(session, evaluations, flags, out_dir: str) -> Path:
    """Copy the image and capture the exact in-memory candidate models and fitting recipes."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    for ev in evaluations:
        if not ev.recipe:
            raise ValueError("This result predates replay support; rerun the fit before exporting.")
        candidates.append(dict(model=structure_record(ev.struct), recipe=ev.recipe,
                               source_path=getattr(ev.struct, "path", ""), expected_fom=ev.fom))
    source = Path(session.path).resolve()
    image_hash = checksum(source)
    if image_hash != session.image_sha256:
        raise ValueError("Image changed since spot detection. Find spots and fit again before exporting.")
    inputs = folder / "inputs"
    inputs.mkdir(exist_ok=True)
    target = inputs / f"{image_hash}{source.suffix}"
    if source != target.resolve():
        shutil.copy2(source, target)
    if checksum(target) != image_hash:
        raise ValueError("Image changed while exporting; rerun and export again.")
    record = dict(schema_version=1, environment=environment(),
                  image=dict(path=target.relative_to(folder).as_posix(), sha256=image_hash),
                  session_settings=asdict(session.s), geometry=asdict(session.geom),
                  candidates=candidates, flags=flags, ranking="descending FOM; diagnostic outcomes are annotations",
                  limitations="Local curvature errors are not calibrated confidence intervals. "
                               "This bundle replays supplied candidates, not a rescan of a changing CIF library.")
    destination = folder / "run_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)
    # Handy standalone geometry/spot settings; the manifest carries candidate-specific fit settings.
    session.s.to_toml(folder / "settings.toml", record["image"]["path"])
    return destination


def replay(manifest_path: str, out_dir: str, log=print):
    from .engine import Settings, Session, evaluate, export, free_cell_for, _rank
    record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest version.")
    folder = Path(manifest_path).resolve().parent
    if Path(out_dir).resolve() == folder:
        raise ValueError("Choose a different output folder to preserve the original report.")
    image = (folder / record["image"]["path"]).resolve()
    if checksum(image) != record["image"]["sha256"]:
        raise ValueError("Input image checksum mismatch; the report input has changed.")
    if record["environment"] != environment():
        log("Environment differs from the recorded run; inspect numerical differences before relying on replay.")
    settings = Settings(**record["session_settings"])
    session = Session(str(image), settings, log=log)
    evaluations = []
    for item in record["candidates"]:
        struct = restore_structure(item["model"])
        recipe = item["recipe"]
        s = Settings(**recipe["settings"])
        fc = replace(s.fit_config(), free_cell=free_cell_for(struct))
        mc = s.multi_config(**({"max_components": recipe["max_textures"]} if recipe["max_textures"] else {}))
        # Refuse silent changes in defaults across releases. JSON normalizes tuple/list differences.
        if (json.dumps(asdict(fc), sort_keys=True) != json.dumps(recipe["fit_config"], sort_keys=True)
                or json.dumps(asdict(mc), sort_keys=True) != json.dumps(recipe["multi_config"], sort_keys=True)):
            raise ValueError("Fitting defaults differ from this report. Use its recorded gixfit version.")
        ev = evaluate(session, struct, s, recipe["contact"], recipe["with_null"], log, recipe["max_textures"])
        log(f"Replay FOM difference for {ev.name}: {ev.fom - item['expected_fom']:+.6g}")
        evaluations.append(ev)
    evaluations = _rank(evaluations)
    export(session, evaluations, record["flags"], out_dir, log=log)
    return session, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    replay(args.manifest, args.output)


if __name__ == "__main__":
    main()
