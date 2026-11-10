"""gixfit desktop app (Tkinter).  Start:  python -m gixfit.app   (or double-click gixfit_app.bat)

1  Find spots       load a GIWAXS image, find the spots, show d-spacings + structure hints (no CIF needed)
2  Identify         search a folder of CIFs (your library) for the structure that explains the pattern
3  Check one CIF    how well ONE CIF matches: per-spot error vs the CIF, refined cell, surrogate diagnostics
4  Compare CIFs     rank several CIFs (polymorphs, rac vs enantiopure, ...)
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
from copy import deepcopy

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm

from . import plots                      # noqa: F401  (sets the Times New Roman house style)
from .engine import Settings, Session, check_cif, compare_cifs, identify, export
from .peaks import to_qgrid
from .crystal import PARAM_NAMES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBS = [p for p in (
    os.path.join(HERE, "library"),
) if os.path.isdir(p)]

FIELDS = [  # (attribute, label, kind)
    ("wavelength", "Wavelength (A)", float), ("sdd_mm", "Sample-detector (mm)", float),
    ("pixel_mm", "Pixel size (mm)", float), ("x0", "Beam centre column x0", float),
    ("y0", "Beam centre row y0", float), ("alpha_i_deg", "Incident angle (deg, blank = file name)", float),
    ("qz_min", "Mask below q_z (horizon)", float), ("threshold", "Spot threshold (sigma)", float),
    ("q_min", "q min used (A^-1)", float), ("q_max", "q max used (A^-1)", float),
]
class _Canvas(FigureCanvasTkAgg):
    def _update_device_pixel_ratio(self, event=None):      # see App._build: the figure dpi already matches
        pass


CHANCE = {"Off (fast fit)": 0, "Preview (6; unstable)": 6, "Standard (100)": 100, "Extended (200)": 200}


class App:
    def __init__(self, root):
        self.root = root
        root.title("gixfit - GIWAXS structure matching")
        self.q = queue.Queue()
        self.session = None
        self.evals, self.flags = [], []
        self.busy = False
        self._qgrid = None
        self.settings = Settings()
        self.vars = {}
        self._build()
        self._fill_fields(self.settings)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=6)
        right = ttk.Frame(pw, padding=4)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        # --- 1 image + detector
        f1 = ttk.LabelFrame(left, text="1  Image and detector", padding=6)
        f1.pack(fill="x")
        self.v_image = tk.StringVar()
        ttk.Label(f1, text="GIWAXS image (.tif)").grid(row=0, column=0, sticky="w")
        e = ttk.Entry(f1, textvariable=self.v_image, width=38)
        e.grid(row=1, column=0, columnspan=2, sticky="we")
        ttk.Button(f1, text="Browse...", command=self._browse_image).grid(row=1, column=2, padx=2)
        fr = ttk.Frame(f1)
        fr.grid(row=2, column=0, columnspan=3, sticky="we", pady=(4, 2))
        ttk.Button(fr, text="Load settings...", command=self._load_settings).pack(side="left")
        ttk.Button(fr, text="Save settings...", command=self._save_settings).pack(side="left", padx=4)
        ttk.Button(fr, text="Try demo...", command=self._open_demo).pack(side="left")
        self.v_variant = tk.StringVar()
        self.cb_variant = ttk.Combobox(fr, textvariable=self.v_variant, width=18, state="readonly")
        self.cb_variant.pack(side="left")
        self.cb_variant.bind("<<ComboboxSelected>>", lambda *_: self._reload_variant())
        for i, (attr, label, _) in enumerate(FIELDS):
            ttk.Label(f1, text=label).grid(row=3 + i, column=0, sticky="w")
            v = tk.StringVar()
            ttk.Entry(f1, textvariable=v, width=12).grid(row=3 + i, column=1, columnspan=2, sticky="w")
            self.vars[attr] = v
        self.b_find = ttk.Button(f1, text="Find spots", command=self._run_find)
        self.b_find.grid(row=4 + len(FIELDS), column=0, columnspan=3, sticky="we", pady=(6, 0))

        # --- options
        fo = ttk.LabelFrame(left, text="Options", padding=6)
        fo.pack(fill="x", pady=6)
        ttk.Label(fo, text="Max textures per structure").grid(row=0, column=0, sticky="w")
        self.v_tex = tk.StringVar(value="1")
        ttk.Spinbox(fo, from_=1, to=3, textvariable=self.v_tex, width=5).grid(row=0, column=1, sticky="w")
        ttk.Label(fo, text="Contact planes up to index").grid(row=1, column=0, sticky="w")
        self.v_maxidx = tk.StringVar(value="2")
        ttk.Spinbox(fo, from_=1, to=3, textvariable=self.v_maxidx, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(fo, text="Optional diagnostics").grid(row=2, column=0, sticky="w")
        self.v_chance = tk.StringVar(value="Off (fast fit)")
        ttk.Combobox(fo, textvariable=self.v_chance, values=list(CHANCE), state="readonly",
                     width=20).grid(row=2, column=1, sticky="w")

        # --- 2 tasks
        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        # identify
        t1 = ttk.Frame(nb, padding=6)
        nb.add(t1, text="Search library")
        ttk.Label(t1, text="CIF library folders (searched recursively)").pack(anchor="w")
        self.lb_lib = tk.Listbox(t1, height=4)
        self.lb_lib.pack(fill="x")
        for p in DEFAULT_LIBS:
            self.lb_lib.insert("end", p)
        r = ttk.Frame(t1); r.pack(fill="x", pady=2)
        ttk.Button(r, text="Add folder...", command=self._add_lib).pack(side="left")
        ttk.Button(r, text="Remove", command=lambda: self._lb_remove(self.lb_lib)).pack(side="left", padx=4)
        ttk.Label(t1, text="Skip files matching (regex)").pack(anchor="w")
        self.v_excl = tk.StringVar(value="slab|vacuum")
        ttk.Entry(t1, textvariable=self.v_excl).pack(fill="x")
        ttk.Label(t1, text="Top N receive optional diagnostics").pack(anchor="w")
        self.v_top = tk.StringVar(value="3")
        ttk.Spinbox(t1, from_=1, to=20, textvariable=self.v_top, width=5).pack(anchor="w")
        self.b_ident = ttk.Button(t1, text="Rank candidate CIFs", command=self._run_identify)
        self.b_ident.pack(fill="x", pady=6)
        ttk.Label(t1, wraplength=330, foreground="#555", justify="left",
                  text="Ranks supplied candidates by fit quality. A top-ranked candidate is not a confirmed "
                       "structure identification.").pack(anchor="w")
        # check one
        t2 = ttk.Frame(nb, padding=6)
        nb.add(t2, text="Check one CIF")
        self.v_cif = tk.StringVar()
        ttk.Label(t2, text="CIF file").pack(anchor="w")
        r = ttk.Frame(t2); r.pack(fill="x")
        ttk.Entry(r, textvariable=self.v_cif).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="Browse...", command=lambda: self._browse_cif(self.v_cif)).pack(side="left", padx=2)
        ttk.Label(t2, text="Contact plane (h k l), or auto").pack(anchor="w", pady=(6, 0))
        self.v_contact = tk.StringVar(value="auto")
        ttk.Entry(t2, textvariable=self.v_contact, width=12).pack(anchor="w")
        self.b_check = ttk.Button(t2, text="Check match", command=self._run_check)
        self.b_check.pack(fill="x", pady=6)
        ttk.Label(t2, wraplength=330, foreground="#555", justify="left",
                  text="Gives the error of every indexed spot against the CIF, the refined cell (a, b, c, alpha, "
                       "beta, gamma) and which surrogate diagnostics are passed (supplementary; not an identification).").pack(anchor="w")
        # compare
        t3 = ttk.Frame(nb, padding=6)
        nb.add(t3, text="Compare CIFs")
        ttk.Label(t3, text="CIF files").pack(anchor="w")
        self.lb_cifs = tk.Listbox(t3, height=7, selectmode="extended")
        self.lb_cifs.pack(fill="both", expand=True)
        r = ttk.Frame(t3); r.pack(fill="x", pady=2)
        ttk.Button(r, text="Add...", command=self._add_cifs).pack(side="left")
        ttk.Button(r, text="Remove", command=lambda: self._lb_remove(self.lb_cifs)).pack(side="left", padx=4)
        ttk.Button(r, text="Clear", command=lambda: self.lb_cifs.delete(0, "end")).pack(side="left")
        self.b_comp = ttk.Button(t3, text="Compare", command=self._run_compare)
        self.b_comp.pack(fill="x", pady=6)
        ttk.Label(t3, wraplength=330, foreground="#555", justify="left",
                  text="R and S enantiomers diffract identically; such pairs are flagged, not ranked.").pack(anchor="w")

        # --- bottom of left
        fb = ttk.Frame(left); fb.pack(fill="x", pady=4)
        ttk.Button(fb, text="Export report...", command=self._export).pack(fill="x")
        ttk.Button(fb, text="Export figure (preview, intensity bar)...",
                   command=self._export_figure).pack(fill="x", pady=(3, 0))
        self.pb = ttk.Progressbar(fb, mode="indeterminate")
        self.pb.pack(fill="x", pady=4)
        self.v_status = tk.StringVar(value="Load an image and press 'Find spots'.")
        ttk.Label(fb, textvariable=self.v_status, wraplength=340, foreground="#1f4e79").pack(anchor="w")

        # --- right: figure + tables
        vp = ttk.PanedWindow(right, orient="vertical")
        vp.pack(fill="both", expand=True)
        ff = ttk.Frame(vp)
        vp.add(ff, weight=3)
        # the process is DPI-aware (sizes are physical pixels), so give the figure the screen's DPI and switch off
        # matplotlib's own device-pixel-ratio scaling (otherwise the figure is drawn 2x too large and clipped)
        scale = max(1.0, float(self.root.tk.call("tk", "scaling")) / (96 / 72))
        self.fig = Figure(figsize=(4, 3), dpi=100 * scale)  # small request: the canvas then grows to fill its pane
        self.ax = self.fig.add_subplot(111)
        self.canvas = _Canvas(self.fig, master=ff)
        NavigationToolbar2Tk(self.canvas, ff).update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.nb_out = ttk.Notebook(vp)
        vp.add(self.nb_out, weight=2)
        self.tv_res = self._tree(self.nb_out, "Results",
                                 [("rank", 40), ("structure", 220), ("surrogate diagnostics", 330), ("fom", 60), ("z_chi", 60),
                                  ("z_q", 60), ("spots", 70), ("textures", 260), ("cell change", 300)])
        self.tv_res.bind("<<TreeviewSelect>>", lambda *_: self._show_selected())
        self.tv_err = self._tree(self.nb_out, "Spot errors vs CIF",
                                 [("texture", 150), ("hkl", 140), ("q obs", 70), ("q CIF", 70), ("dq CIF %", 75),
                                  ("q refined", 75), ("dq refined %", 85), ("chi obs", 65), ("chi calc", 65),
                                  ("d obs (A)", 70), ("intensity", 85)])
        self.tv_spots = self._tree(self.nb_out, "Spots",
                                   [("id", 40), ("q (A^-1)", 75), ("d (A)", 70), ("chi (deg)", 70), ("q_xy", 70),
                                    ("q_z", 70), ("intensity", 90), ("SNR", 70), ("used", 50)])
        self.root.after(300, lambda: vp.sashpos(0, int(vp.winfo_height() * 0.62)))   # map gets ~60 % height
        fl = ttk.Frame(self.nb_out)
        self.nb_out.add(fl, text="Details / log")
        self.txt = tk.Text(fl, wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(fl, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(fill="both", expand=True)
        self._draw()

    def _tree(self, nb, title, cols):
        fr = ttk.Frame(nb)
        nb.add(fr, text=title)
        tv = ttk.Treeview(fr, columns=[c for c, _ in cols], show="headings")
        for c, w in cols:
            tv.heading(c, text=c, command=lambda c=c, tv=tv: self._sort(tv, c))
            tv.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(fr, command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tv.pack(fill="both", expand=True)
        return tv

    def _sort(self, tv, col):
        rows = [(tv.set(k, col), k) for k in tv.get_children("")]
        def key(v):
            try:
                return (0, float(v[0]))
            except ValueError:
                return (1, v[0])
        rev = getattr(tv, "_rev", False)
        for i, (_, k) in enumerate(sorted(rows, key=key, reverse=rev)):
            tv.move(k, "", i)
        tv._rev = not rev

    # ------------------------------------------------------------------ settings fields
    def _fill_fields(self, s: Settings):
        for attr, _, _ in FIELDS:
            v = getattr(s, attr)
            self.vars[attr].set("" if v is None else f"{v:g}")
        self.v_tex.set(str(s.max_textures))
        self.v_maxidx.set(str(s.planes_maxidx))
        self.v_top.set(str(s.identify_null_top))
        label = next((k for k, v in CHANCE.items() if v == s.n_null), None)
        if label is None:
            label = f"Custom ({s.n_null})"
            CHANCE[label] = s.n_null
        self.v_chance.set(label)

    def _read_fields(self) -> Settings:
        s = deepcopy(self.settings)
        for attr, label, kind in FIELDS:
            txt = self.vars[attr].get().strip()
            if txt == "" and attr in ("alpha_i_deg",):
                setattr(s, attr, None)
                continue
            try:
                setattr(s, attr, kind(txt))
            except ValueError:
                raise ValueError(f"'{label}' is not a number: {txt!r}")
        s.max_textures = int(self.v_tex.get())
        s.planes_maxidx = int(self.v_maxidx.get())
        s.n_null = CHANCE[self.v_chance.get()]
        s.identify_null_top = int(self.v_top.get())
        s.validate()
        return s

    def _open_demo(self):
        if self.busy:
            messagebox.showinfo("gixfit", "Wait for the current task to finish.")
            return
        folder = filedialog.askdirectory(title="Choose a folder for the synthetic tutorial")
        if not folder:
            return
        from .demo import create_demo
        image, cif, settings = create_demo(folder)
        self.settings = settings
        self._fill_fields(settings)
        self.v_image.set(image)
        self.v_cif.set(cif)
        self.v_contact.set("auto")
        self.v_status.set("Synthetic tutorial loaded. Press Find spots, then Check one CIF > Check match.")
        self._log("Synthetic tutorial only; not experimental validation. Geometry is supplied for this demo.")

    def _browse_image(self):
        p = filedialog.askopenfilename(title="GIWAXS image", filetypes=[("TIFF", "*.tif *.tiff"), ("all", "*.*")])
        if p:
            self.v_image.set(p)

    def _browse_cif(self, var):
        p = filedialog.askopenfilename(title="CIF file", filetypes=[("CIF", "*.cif"), ("all", "*.*")])
        if p:
            var.set(p)

    def _add_cifs(self):
        for p in filedialog.askopenfilenames(title="CIF files", filetypes=[("CIF", "*.cif"), ("all", "*.*")]):
            self.lb_cifs.insert("end", p)

    def _add_lib(self):
        p = filedialog.askdirectory(title="CIF library folder")
        if p:
            self.lb_lib.insert("end", p)

    @staticmethod
    def _lb_remove(lb):
        for i in reversed(lb.curselection()):
            lb.delete(i)

    def _load_settings(self):
        p = filedialog.askopenfilename(title="Settings / gixfit config", filetypes=[("TOML", "*.toml"), ("all", "*.*")])
        if not p:
            return
        self._settings_file = p
        s, img, variants = Settings.from_toml(p)
        self.cb_variant["values"] = variants
        self.v_variant.set(variants[0] if variants else "")
        if variants:                                   # apply the first geometry variant
            s, img, _ = Settings.from_toml(p, variant=0)
        self.settings = s
        self._fill_fields(s)
        if img:
            self.v_image.set(img)
        self._log(f"Settings loaded from {p}" + (f" (geometry variant: {variants[0]})" if variants else ""))

    def _reload_variant(self):
        p = getattr(self, "_settings_file", None)
        if not p:
            return
        i = list(self.cb_variant["values"]).index(self.v_variant.get())
        s, _, _ = Settings.from_toml(p, variant=i)
        self.settings = s
        self._fill_fields(s)
        self._log(f"Geometry variant: {self.v_variant.get()}  (x0 {s.x0}, y0 {s.y0}, SDD {s.sdd_mm})")

    def _save_settings(self):
        try:
            s = self._read_fields()
        except ValueError as e:
            messagebox.showerror("gixfit", str(e)); return
        p = filedialog.asksaveasfilename(defaultextension=".toml", filetypes=[("TOML", "*.toml")])
        if p:
            s.to_toml(p, self.v_image.get())
            self._log(f"Settings saved to {p}")

    # ------------------------------------------------------------------ background jobs
    def _start(self, name, fn):
        if self.busy:
            messagebox.showinfo("gixfit", "Still working on the previous task."); return
        self.busy = True
        self.pb.start(12)
        self.v_status.set(f"{name} ... (see 'Details / log')")
        self.nb_out.select(3)

        def run():
            try:
                self.q.put(("done", name, fn()))
            except Exception:                                   # noqa: BLE001
                self.q.put(("error", name, traceback.format_exc()))
        threading.Thread(target=run, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "log":
                    self._log(a)
                elif kind == "done":
                    self.busy = False; self.pb.stop()
                    self._finish(a, b)
                elif kind == "error":
                    self.busy = False; self.pb.stop()
                    self.v_status.set(f"{a} failed - see log")
                    self._log(b)
                    messagebox.showerror("gixfit", f"{a} failed:\n\n{b.strip().splitlines()[-1]}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _logq(self, msg):
        self.q.put(("log", str(msg), None))

    def _log(self, msg):
        self.txt.insert("end", str(msg) + "\n")
        self.txt.see("end")

    def _need_session(self):
        if self.session is None:
            messagebox.showinfo("gixfit", "First load an image and press 'Find spots'.")
            return False
        return True

    def _run_find(self):
        img = self.v_image.get().strip()
        if not os.path.isfile(img):
            messagebox.showerror("gixfit", "Choose an existing image file first."); return
        try:
            s = self._read_fields()
        except ValueError as e:
            messagebox.showerror("gixfit", str(e)); return
        self.settings = deepcopy(s)
        self._start("Find spots", lambda: Session(img, s, log=self._logq))

    def _apply_options(self):
        try:
            s = self._read_fields()
        except ValueError as e:
            messagebox.showerror("gixfit", str(e)); return None
        # geometry / spot settings changed since 'Find spots'?  -> the spots must be found again
        if not self.session.matches_inputs(self.v_image.get().strip(), s):
            self.v_status.set("Results are stale: image or detection settings changed. Find spots again.")
            messagebox.showinfo("gixfit", "Image, geometry or spot settings changed: press 'Find spots' again.")
            return None
        self.settings = deepcopy(s)
        return s

    def _run_check(self):
        if not self._need_session():
            return
        cif = self.v_cif.get().strip()
        if not os.path.isfile(cif):
            messagebox.showerror("gixfit", "Choose a CIF file."); return
        s = self._apply_options()
        if s is None:
            return
        txt = self.v_contact.get().strip().lower()
        contact = None
        if txt and txt != "auto":
            try:
                contact = tuple(int(v) for v in txt.replace(",", " ").replace("(", " ").replace(")", " ").split())
                assert len(contact) == 3 and any(contact)
            except (ValueError, AssertionError):
                messagebox.showerror("gixfit", "Contact plane: three integers, e.g. 1 0 1, or 'auto'."); return
        ses = self.session
        self._start("Check CIF", lambda: ([check_cif(ses, cif, s, contact, log=self._logq)], []))

    def _run_compare(self):
        if not self._need_session():
            return
        cifs = list(self.lb_cifs.get(0, "end"))
        if len(cifs) < 2:
            messagebox.showinfo("gixfit", "Add at least two CIF files."); return
        s = self._apply_options()
        if s is None:
            return
        ses = self.session
        self._start("Compare CIFs", lambda: compare_cifs(ses, cifs, s, log=self._logq))

    def _run_identify(self):
        if not self._need_session():
            return
        libs = list(self.lb_lib.get(0, "end"))
        if not libs:
            messagebox.showinfo("gixfit", "Add a CIF library folder."); return
        s = self._apply_options()
        if s is None:
            return
        ses, excl = self.session, self.v_excl.get().strip()
        self._start("Identify", lambda: identify(ses, libs, s, excl, log=self._logq)[:2])

    def _export(self):
        if not self.session:
            messagebox.showinfo("gixfit", "Nothing to export yet."); return
        s = self._apply_options()
        if s is None:
            return
        from dataclasses import asdict
        if any(e.recipe.get("settings") != asdict(s) for e in self.evals):
            messagebox.showinfo("gixfit", "Fit options changed. Run the fit again before exporting.")
            return
        d = filedialog.askdirectory(title="Folder for the report")
        if d:
            ses, evs, flags = self.session, list(self.evals), list(self.flags)
            self._start("Export", lambda: (export(ses, evs, flags, d, log=self._logq), d)[1])

    # ------------------------------------------------------------------ results
    def _finish(self, name, res):
        if name == "Find spots":
            self.session, self.evals, self.flags = res, [], []
            self._qgrid = None
            self._fill_spots()
            self._fill_results()
            self._log("Structure hints (no CIF needed):")
            for h in res.hints():
                self._log("  " + h)
            self.v_status.set(f"{len(res.peaks)} spots. Check one CIF, compare CIFs, or search your library.")
            self._draw()
            self.nb_out.select(2)
        elif name == "Export":
            self.v_status.set(f"Report written to {res}")
        else:
            self.evals, self.flags = res
            self._fill_results()
            if self.flags:
                self._log("Same diffraction pattern (cannot be told apart): " +
                          "; ".join(f"{a} = {b}" for a, b, _ in self.flags))
            best = self.evals[0] if self.evals else None
            self.v_status.set(f"{name} done. Best: {best.name} - {best.verdict}" if best else f"{name} done.")
            if self.evals:
                first = self.tv_res.get_children("")[0]
                self.tv_res.selection_set(first)
            self.nb_out.select(0)

    def _fill_spots(self):
        tv = self.tv_spots
        tv.delete(*tv.get_children(""))
        for r in self.session.peak_rows():
            tv.insert("", "end", values=(r["id"], f"{r['q']:.4f}", f"{r['d']:.3f}", f"{r['chi']:.1f}",
                                         f"{r['qxy']:.4f}", f"{r['qz']:.4f}", f"{r['intensity']:.3g}",
                                         f"{r['snr']:.0f}", "yes" if r["used"] else ""))

    def _fill_results(self):
        tv = self.tv_res
        tv.delete(*tv.get_children(""))
        for i, e in enumerate(self.evals):
            m = e.best.metrics if e.best else {}
            change = ""
            if e.best and e.best.cells[0] is not None:
                c0, c1 = e.struct.cell.array(), e.best.cells[0].array()
                change = " ".join(f"{n}{100 * (c1[j] / c0[j] - 1):+.1f}%" for j, n in enumerate("abc")) + "  " + \
                         " ".join(f"{n}{c1[j + 3] - c0[j + 3]:+.1f}" for j, n in enumerate(("al", "be", "ga")))
            name = e.name + ("  [cell only]" if e.cell_only else "")
            tv.insert("", "end", iid=str(i), values=(
                i + 1, name, e.verdict, f"{e.fom:.3f}", _f(e.z_chi), _f(e.z_q),
                f"{m.get('n_match', 0)}/{m.get('n_obs', 0)}", e.best.label().replace(e.name, "") if e.best else "-",
                change))

    def _show_selected(self):
        sel = self.tv_res.selection()
        if not sel:
            return
        e = self.evals[int(sel[0])]
        self._draw(e)
        tv = self.tv_err
        tv.delete(*tv.get_children(""))
        if e.best:
            for r in sorted(e.best.pairs, key=lambda r: r["q_obs"]):
                tv.insert("", "end", values=(
                    f"{r['contact']}", r["label"], f"{r['q_obs']:.4f}", f"{r['q_cif']:.4f}", f"{r['dq_cif_pct']:+.2f}",
                    f"{r['q_fit']:.4f}", f"{r['dq_fit_pct']:+.2f}", f"{r['chi_obs']:.1f}", f"{r['chi_fit']:.1f}",
                    f"{r['d_obs']:.3f}", f"{r['area_obs']:.3g}"))
        self._log("")
        self._log(f"=== {e.name}: {e.verdict}")
        if e.best:
            m = e.best.metrics
            self._log(f"  textures: {e.best.label()}   (textures passing both diagnostics: {e.k_sig})")
            self._log(f"  spots indexed {m['n_match']}/{m['n_obs']}, strongest predicted spots missing "
                      f"{m['n_strong_missing']}/{m['n_strong_pred']}, fom {m['fom']:.3f}, "
                      f"intensity rank corr {m['spearman_I_pooled']:.2f}")
            self._log(f"  rms error of spot q vs CIF {m['rms_dq_cif_pct']:.2f}%  -> after refining the cell "
                      f"{m['rms_dq_fit_pct']:.2f}%")
            c0, c1, ce = e.struct.cell.array(), e.best.cells[0].array(), e.best.cell_errs[0]
            self._log("  " + "  ".join(f"{n:>6s}" for n in PARAM_NAMES))
            self._log("  CIF     " + " ".join(f"{v:8.3f}" for v in c0))
            self._log("  refined " + " ".join(f"{v:8.3f}" for v in c1))
            self._log("  local curvature errors " + " ".join(f"{v:8.3f}" for v in ce))
            self._log("  Errors condition on assignments, priors and geometry; calibration uncertainty is not included.")
            for mode, per_k in (e.null or {}).items():
                for d in per_k:
                    self._log(f"  chance [{mode}] k={d['k']}: fom {d['fom']:.3f} vs {d['null_mean']:.3f} +- "
                              f"{d['null_sd']:.3f}  (z {d['z']:.1f})")
        if e.cell_only:
            self._log("  NOTE: this CIF has no atomic coordinates - only positions and systematic absences were "
                      "tested; intensities could not be used.")

    # ------------------------------------------------------------------ figure
    def _export_figure(self):
        """the map currently shown (with its fit, if a result is selected) -> preview dialog -> file."""
        if not self.session:
            messagebox.showinfo("gixfit", "Find spots first; then select a result to include its fit."); return
        from .figexport import open_dialog
        self.fig_dialog = open_dialog(self.root, self.session, self._shown_ev, self._qgrid,
                                      initial_dir=os.path.dirname(self.session.path))

    def _draw(self, ev=None):
        self._shown_ev = ev
        ax = self.ax
        ax.clear()
        if self.session is None:
            ax.text(0.5, 0.5, "Load an image and press 'Find spots'", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_axis_off()
            self.canvas.draw_idle()
            return
        ax.set_axis_on()
        ses = self.session
        qlims = ((-0.6, max(2.0, ses.s.q_max)), (0.0, max(2.0, ses.s.q_max)))
        if self._qgrid is None:
            self._qgrid = to_qgrid(ses.D, ses.mask, ses.geom, qlims[0], qlims[1], 0.004)
        qxy, qz, img = self._qgrid
        v = img[np.isfinite(img) & (img > 0)]
        lo, hi = (np.percentile(v, 30), np.percentile(v, 99.7)) if v.size else (1, 10)
        ax.imshow(np.clip(img, lo, None), origin="lower", extent=[qxy[0], qxy[-1], qz[0], qz[-1]],
                  cmap="magma", norm=LogNorm(lo, hi), aspect="equal", interpolation="nearest")
        ax.set_facecolor("#bdbdbd")
        ax.set_xlabel(r"$q_{xy}$ ($\AA^{-1}$)"); ax.set_ylabel(r"$q_z$ ($\AA^{-1}$)")
        ax.set_xlim(qlims[0]); ax.set_ylim(qlims[1])
        if ev is None or ev.best is None:
            px, pz = ses.geom.pixel_to_q(ses.peaks.col, ses.peaks.row)
            ax.scatter(px, pz, s=40, facecolors="none", edgecolors="cyan", lw=1.0)
            ax.set_title(f"{os.path.basename(ses.path)}: {len(ses.peaks)} spots", fontsize=11)
        else:
            plots.draw_multi(ax, ev.best, ses.peaks, qlims[0], qlims[1], title=False)
            ax.set_title(f"{ev.name}: {ev.verdict}", fontsize=11)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def _f(x):
    return "" if x is None or not np.isfinite(x) else f"{x:.1f}"


def dpi_aware():
    """sharp text on high-DPI Windows screens (call before creating the Tk root)."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:                                   # noqa: BLE001 - not Windows / older Windows
        pass


def make_root():
    dpi_aware()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    from tkinter import font as tkfont
    line = tkfont.nametofont("TkDefaultFont").metrics("linespace")
    ttk.Style().configure("Treeview", rowheight=int(line * 1.35))        # high-DPI: rows were clipped
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = int(sw * 0.92), int(sh * 0.88)
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
    return root


def main():
    root = make_root()
    App(root)
    root.mainloop()


if __name__ == "__main__":       # guard: chance-test worker processes re-import this module on Windows
    main()
