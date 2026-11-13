"""ESRF Data Format (.edf) images, as written by laboratory GIWAXS systems such as the Xenocs Xeuss
(Dectris EIGER/PILATUS).  Reads the header and the image, and turns the header into gixfit geometry.

Caveat for the Xeuss: Center_1 / Center_2 in the header are NOT updated when the detector is translated
(det0x_dial / det0z_dial).  With a reference (beam centre measured at known motor positions) the centre is shifted
by the motor offset / pixel size; without one, the header centre is used and should be checked (e.g. the left/right
symmetry of a powder ring, or the specular rod at q_xy = 0).
"""
from __future__ import annotations

import os

import numpy as np

_DTYPES = {
    "floatvalue": "f4", "float": "f4", "float32": "f4", "doublevalue": "f8", "double": "f8", "float64": "f8",
    "unsignedbyte": "u1", "unsignedchar": "u1", "unsignedshort": "u2", "unsignedshortinteger": "u2",
    "unsignedinteger": "u4", "unsignedint": "u4", "unsignedlong": "u4", "unsigned32": "u4",
    "signedbyte": "i1", "signedshort": "i2", "signedinteger": "i4", "signedint": "i4", "signedlong": "i4",
    "signed32": "i4",
}


def is_edf(path) -> bool:
    return str(path).lower().endswith(".edf")


def read_header(path) -> dict:
    """the key = value pairs of the (first) EDF header block."""
    with open(path, "rb") as fh:
        head = fh.read(65536)
    start, end = head.find(b"{"), head.find(b"}")
    if start < 0 or end < 0:
        raise ValueError(f"{os.path.basename(path)}: no EDF header found")
    h = {}
    for part in head[start + 1:end].decode("latin1").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            h[k.strip()] = v.strip()
    h["_header_bytes"] = str(_header_size(head, end))
    return h


def _header_size(head, end):
    """header block length: EDF_HeaderSize if given, else up to the end of the line after '}' (multiple of 512)."""
    txt = head[:end].decode("latin1")
    if "EDF_HeaderSize" in txt:
        return int(txt.split("EDF_HeaderSize")[1].split("=")[1].split(";")[0])
    nl = head.find(b"\n", end)
    n = (nl + 1) if nl >= 0 else end + 1
    return int(np.ceil(n / 512.0) * 512)


def read_edf(path):
    """(header, image as float64 array of shape (Dim_2, Dim_1))."""
    h = read_header(path)
    n1, n2 = int(h["Dim_1"]), int(h["Dim_2"])
    dt = _DTYPES.get(h.get("DataType", "FloatValue").replace(" ", "").lower())
    if dt is None:
        raise ValueError(f"{os.path.basename(path)}: unsupported EDF DataType {h.get('DataType')!r}")
    order = "<" if h.get("ByteOrder", "LowByteFirst").lower().startswith("low") else ">"
    hs = int(h["_header_bytes"])
    with open(path, "rb") as fh:
        fh.seek(hs)
        data = np.frombuffer(fh.read(n1 * n2 * np.dtype(dt).itemsize), dtype=order + dt)
    if data.size != n1 * n2:
        raise ValueError(f"{os.path.basename(path)}: expected {n1 * n2} pixels, found {data.size}")
    return h, data.reshape(n2, n1).astype(np.float64)


def _f(h, *keys):
    for k in keys:
        if k in h:
            try:
                return float(h[k].split()[0])
            except ValueError:
                pass
    return None


def geometry_from_header(h, reference=None) -> dict:
    """gixfit geometry from an EDF header: wavelength (A), sdd_mm, pixel_mm, x0, y0 (px) and alpha_i_deg.

    reference = (x0, y0, det0x, det0z): beam centre measured at those detector-motor positions (mm).  When given and
    the header has det0x_dial / det0z_dial, x0/y0 are shifted by the motor offset / pixel size (Xeuss convention:
    x0 grows with det0x, y0 falls with det0z).  Returns the dict plus a list of human-readable notes."""
    lam = _f(h, "WaveLength", "Wavelength", "wavelength")
    sdd = _f(h, "SampleDistance", "Distance", "sample_distance")
    pix = _f(h, "PSize_1", "PixelSize", "pixel_size")
    out, notes = {}, []
    if lam:
        out["wavelength"] = lam * 1e10 if lam < 1e-6 else lam                    # header in m -> A
    if sdd:
        out["sdd_mm"] = sdd * 1e3 if sdd < 10 else sdd                          # header in m -> mm
    if pix:
        out["pixel_mm"] = pix * 1e3 if pix < 0.01 else pix                      # header in m -> mm
    c1, c2 = _f(h, "Center_1"), _f(h, "Center_2")
    if c1 is not None and c2 is not None:
        out["x0"], out["y0"] = c1, c2
    om = _f(h, "om", "Omega", "incident_angle")
    if om is not None:
        out["alpha_i_deg"] = round(om, 4)
    dx, dz = _f(h, "det0x_dial"), _f(h, "det0z_dial")
    if reference is not None and dx is not None and dz is not None and out.get("pixel_mm"):
        rx0, ry0, rdx, rdz = (float(v) for v in reference)
        out["x0"] = rx0 + (dx - rdx) / out["pixel_mm"]
        out["y0"] = ry0 - (dz - rdz) / out["pixel_mm"]
        notes.append(f"beam centre from the reference ({rx0:g}, {ry0:g} at det0x {rdx:g}, det0z {rdz:g}) shifted by "
                     f"the detector motors: x0 {out['x0']:.2f}, y0 {out['y0']:.2f} (header {c1}, {c2})")
    elif dx is not None and dz is not None:
        notes.append(f"beam centre taken from the header ({c1}, {c2}); the header is not updated when the detector is "
                     f"moved (det0x {dx:g}, det0z {dz:g}) - check it, or give a reference to correct it")
    return out, notes
