"""Unit cells, atomic structures and structure factors.

* ``Cell``       a, b, c (A), alpha, beta, gamma (deg) -> direct / reciprocal matrices (2*pi convention).
* ``Structure``  a Cell + atoms in FRACTIONAL coordinates (full P1 list).  Fractional coordinates are
                 kept fixed when the cell is refined, so |F(hkl)|^2 follows the refined cell.
* Readers: CIF (via pymatgen, symmetry-expanded), FHI-aims geometry.in (lattice_vector / atom lines).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np

PARAM_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")


@dataclass
class Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @classmethod
    def from_array(cls, p):
        return cls(*map(float, p))

    def array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.alpha, self.beta, self.gamma], float)

    def direct_matrix(self) -> np.ndarray:
        """rows = a, b, c in Cartesian A (a along x, b in the xy plane)."""
        al, be, ga = np.radians([self.alpha, self.beta, self.gamma])
        ax = np.array([self.a, 0.0, 0.0])
        bx = np.array([self.b * np.cos(ga), self.b * np.sin(ga), 0.0])
        cx = self.c * np.cos(be)
        cy = self.c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz2 = self.c ** 2 - cx ** 2 - cy ** 2
        if cz2 <= 0:
            raise ValueError(f"impossible cell angles {self.alpha, self.beta, self.gamma}")
        return np.vstack([ax, bx, [cx, cy, np.sqrt(cz2)]])

    def reciprocal_matrix(self) -> np.ndarray:
        """rows = a*, b*, c* with a.a* = 2*pi."""
        return 2.0 * np.pi * np.linalg.inv(self.direct_matrix()).T

    def volume(self) -> float:
        return float(abs(np.linalg.det(self.direct_matrix())))

    def g(self, hkl) -> np.ndarray:
        return np.asarray(hkl, float) @ self.reciprocal_matrix()

    def d(self, hkl) -> np.ndarray:
        return 2.0 * np.pi / np.linalg.norm(self.g(hkl), axis=-1)

    def __str__(self):
        return (f"a={self.a:.4f} b={self.b:.4f} c={self.c:.4f} A  "
                f"alpha={self.alpha:.3f} beta={self.beta:.3f} gamma={self.gamma:.3f} deg  V={self.volume():.1f} A^3")


# ---------------------------------------------------------------------- scattering factors
_FF_CACHE: dict = {}


def form_factor(element: str, s):
    """f(s), s = sin(theta)/lambda = |q|/(4 pi).  pymatgen's 4-Gaussian table (Mott-Bethe form)."""
    from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
    from pymatgen.core import Element
    if element not in _FF_CACHE:
        p = np.array(ATOMIC_SCATTERING_PARAMS[element])
        _FF_CACHE[element] = (Element(element).Z, p[:, 0], p[:, 1])
    Z, a, b = _FF_CACHE[element]
    s = np.asarray(s, float)
    s2 = s[..., None] ** 2
    return Z - 41.78214 * s ** 2 * np.sum(a * np.exp(-b * s2), axis=-1)


@dataclass
class Structure:
    cell: Cell
    species: list = field(default_factory=list)          # element symbols
    frac: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    occ: np.ndarray | None = None
    b_iso: float = 4.0                                    # A^2, typical for molecular crystals at RT
    name: str = ""
    symops: list | None = None                            # (R, t) pairs; used only for cell-only structures

    @property
    def has_atoms(self) -> bool:
        return len(self.species) > 0

    def with_cell(self, cell: Cell) -> "Structure":
        return replace(self, cell=cell)

    def f_squared(self, hkl) -> np.ndarray:
        """|F(hkl)|^2 (electrons^2).  Cell-only input (no atoms): 1 for every reflection allowed by the space
        group, 0 for systematic absences (if symmetry operators are known), i.e. positions only, no intensities."""
        hkl = np.atleast_2d(np.asarray(hkl, float))
        if not self.has_atoms:
            return self._allowed(hkl)
        q = np.linalg.norm(self.cell.g(hkl), axis=1)
        s = q / (4 * np.pi)
        occ = np.ones(len(self.species)) if self.occ is None else self.occ
        F = np.zeros(len(hkl), complex)
        for el in sorted(set(self.species)):
            idx = [i for i, e in enumerate(self.species) if e == el]
            phase = np.exp(2j * np.pi * hkl @ self.frac[idx].T)        # (n_hkl, n_atoms)
            F += form_factor(el, s) * (phase @ occ[idx])
        F *= np.exp(-self.b_iso * s ** 2)
        return np.abs(F) ** 2

    def _allowed(self, hkl) -> np.ndarray:
        """1 = allowed, 0 = systematically absent: |sum over symmetry copies of a random general position|^2
        vanishes exactly for absences (centring, screw axes, glides)."""
        if not self.symops:
            return np.ones(len(hkl))
        rng = np.random.default_rng(0)
        tot = np.zeros(len(hkl))
        for _ in range(3):
            x = rng.random(3)
            pos = np.array([R @ x + t for R, t in self.symops])
            tot += np.abs(np.exp(2j * np.pi * hkl @ pos.T).sum(1)) ** 2
        return (tot / (3 * len(self.symops) ** 2) > 1e-8).astype(float)


# ---------------------------------------------------------------------- readers / writers
def _parse_symop(op: str):
    """'1/2-x,y,1/2+z' -> (R 3x3, t 3).  Small parser (never eval: CIFs are untrusted input)."""
    import re
    from fractions import Fraction
    R = np.zeros((3, 3)); t = np.zeros(3)
    parts = op.lower().replace(" ", "").split(",")
    if len(parts) != 3:
        raise ValueError(op)
    for i, expr in enumerate(parts):
        if expr[0] not in "+-":
            expr = "+" + expr
        for term in re.findall(r"[+-][^+-]+", expr):
            sign = -1.0 if term[0] == "-" else 1.0
            body = term[1:]
            if body[-1] in "xyz":
                coef = body[:-1].rstrip("*")
                R[i, "xyz".index(body[-1])] += sign * (float(Fraction(coef)) if coef else 1.0)
            else:
                t[i] += sign * float(Fraction(body))
    return R, t


def read_cif_cell_only(path, b_iso=4.0) -> Structure:
    """fallback for CIFs without atomic coordinates (e.g. CSD entries 'No coordinates were deposited'):
    cell + symmetry operators only -> peak POSITIONS and systematic absences, no intensities."""
    import re
    txt = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    num = lambda s: float(re.sub(r"\(.*?\)", "", s))
    vals = {}
    for line in txt:
        t = line.split()
        if len(t) >= 2 and t[0] in ("_cell_length_a", "_cell_length_b", "_cell_length_c",
                                    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"):
            vals[t[0]] = num(t[1])
    cell = Cell(vals["_cell_length_a"], vals["_cell_length_b"], vals["_cell_length_c"],
                vals["_cell_angle_alpha"], vals["_cell_angle_beta"], vals["_cell_angle_gamma"])
    ops, in_loop, cols, xyz_col = [], False, [], None
    for line in txt:
        s = line.strip()
        if s == "loop_":
            in_loop, cols, xyz_col = True, [], None
            continue
        if in_loop and s.startswith("_"):
            cols.append(s.split()[0])
            if s.split()[0] in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz"):
                xyz_col = len(cols) - 1
            continue
        if in_loop and xyz_col is not None and s and not s.startswith("#"):
            if s.startswith(("loop_", "data_")) or s.startswith("_"):
                in_loop = False
                continue
            m = re.findall(r"'([^']*)'|\"([^\"]*)\"|(\S+)", s)
            toks = [a or b or c for a, b, c in m]
            if len(toks) == 1:            # unquoted op containing no spaces
                expr = toks[0]
            elif len(cols) > 1 and len(toks) >= len(cols):
                expr = toks[xyz_col]
            else:
                expr = "".join(toks[1:]) if toks[0].isdigit() else "".join(toks)
            try:
                ops.append(_parse_symop(expr))
            except Exception:
                in_loop = False
        elif in_loop and not s.startswith("_"):
            in_loop = False if xyz_col is None and s else in_loop
    return Structure(cell, [], np.zeros((0, 3)), None, b_iso, name=str(path), symops=ops or None)


def read_cif(path, b_iso=4.0) -> Structure:
    import warnings
    from pymatgen.core import Structure as PmgStructure
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = PmgStructure.from_file(str(path))
    except Exception:
        return read_cif_cell_only(path, b_iso)
    lat = s.lattice
    cell = Cell(*lat.abc, *lat.angles)
    species, frac, occ = [], [], []
    for site in s:
        for sp, o in site.species.items():
            species.append(sp.symbol)
            frac.append(site.frac_coords)
            occ.append(o)
    return Structure(cell, species, np.array(frac), np.array(occ), b_iso, name=str(path))


def read_aims(path, drop=(), b_iso=4.0) -> Structure:
    """FHI-aims geometry.in.  ``drop`` = elements to remove (e.g. ('Mo','O') to strip a dopant)."""
    L, cart, frac, sp_c, sp_f = [], [], [], [], []
    for line in open(path):
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "lattice_vector":
            L.append([float(v) for v in t[1:4]])
        elif t[0] == "atom":
            cart.append([float(v) for v in t[1:4]]); sp_c.append(t[4])
        elif t[0] == "atom_frac":
            frac.append([float(v) for v in t[1:4]]); sp_f.append(t[4])
    A = np.array(L)
    fr = list(np.array(cart) @ np.linalg.inv(A)) if cart else []
    fr += frac
    sp = sp_c + sp_f
    keep = [i for i, e in enumerate(sp) if e not in drop]
    a, b, c = np.linalg.norm(A, axis=1)
    ang = lambda u, v: np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    cell = Cell(a, b, c, ang(A[1], A[2]), ang(A[0], A[2]), ang(A[0], A[1]))
    # NOTE: fractional coordinates are invariant to the Cartesian orientation, so re-expressing the cell in
    # the a-along-x convention keeps the structure identical.
    return Structure(cell, [sp[i] for i in keep], np.array(fr)[keep] % 1.0, None, b_iso, name=str(path))


def write_cif(struct: Structure, path, title="gixfit"):
    c = struct.cell
    lines = [f"data_{title.replace(' ', '_')}",
             f"_cell_length_a {c.a:.5f}", f"_cell_length_b {c.b:.5f}", f"_cell_length_c {c.c:.5f}",
             f"_cell_angle_alpha {c.alpha:.4f}", f"_cell_angle_beta {c.beta:.4f}", f"_cell_angle_gamma {c.gamma:.4f}",
             "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1",
             "loop_", "_symmetry_equiv_pos_as_xyz", "  'x, y, z'",
             "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z", "_atom_site_occupancy"]
    occ = np.ones(len(struct.species)) if struct.occ is None else struct.occ
    count: dict = {}
    for el, f, o in zip(struct.species, struct.frac, occ):
        count[el] = count.get(el, 0) + 1
        lines.append(f"  {el}{count[el]} {el} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f} {o:.3f}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def load_structure(spec: dict, base_dir=".") -> Structure:
    """Candidate entry from the config -> Structure.

    keys: cif | aims (path), drop_elements, cell (6 numbers, overrides the file cell, keeps fractional
    coordinates), b_iso, name.  A candidate with only ``cell`` is a cell-only model (no intensities).
    """
    import os
    b_iso = spec.get("b_iso", 4.0)
    if "cif" in spec:
        st = read_cif(os.path.join(base_dir, spec["cif"]), b_iso)
    elif "aims" in spec:
        st = read_aims(os.path.join(base_dir, spec["aims"]), tuple(spec.get("drop_elements", ())), b_iso)
    else:
        st = Structure(Cell(*spec["cell"]), b_iso=b_iso)
    if "cell" in spec:
        st = st.with_cell(Cell(*spec["cell"]))
    st.name = spec.get("name", st.name)
    return st
