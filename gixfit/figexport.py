"""Export the fitted q-map as a figure with an adjustable intensity (colour) bar.

Desktop: "Export figure..." opens a dialog with a live preview; the saved file is drawn from the same settings
at the chosen size and resolution.  Scripts:  save_map(path, session, evaluation, FigureOptions(...)).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from . import plots                     # noqa: F401  (sets the house style: Times New Roman, stix mathtext)
from .peaks import to_qgrid

CMAPS = ("magma", "inferno", "viridis", "cividis", "turbo", "gray", "gray_r")
FORMATS = (("PNG image", "*.png"), ("TIFF image", "*.tif"), ("PDF (vector overlays)", "*.pdf"),
           ("SVG (vector overlays)", "*.svg"))


@dataclass
class FigureOptions:
    vmin: float | None = None           # intensity limits of the colour bar (None: from the percentiles below)
    vmax: float | None = None
    lo_pct: float = 30.0
    hi_pct: float = 99.7
    scale: str = "log"                  # "log" or "linear"
    cmap: str = "magma"
    overlay: bool = True                # predicted / indexed / not-indexed spots of the fit
    labels: bool = True                 # hkl labels of the strongest indexed spots
    n_labels: int | None = 8            # how many indexed spots get an hkl label (strongest first); None = all
    legend: bool = True
    colorbar: bool = True
    title: str = ""
    cbar_label: str = "Intensity (counts)"
    qxy_lim: tuple | None = None        # None: the detector's own coverage on both sides (auto_qlims)
    qz_lim: tuple | None = None
    width_in: float = 7.5
    height_in: float = 5.0
    dpi: int = 300


def auto_qlims(session, qmax=None, step=0.1):
    """(qxy_lim, qz_lim) covering every usable pixel up to |q| = qmax (default max(2, q_max)), on BOTH sides of the
    beam, rounded outwards to `step`.  Fixed limits such as q_xy -0.6..2 hide the negative-q_xy half of a detector
    whose beam centre is near the middle, together with the spots indexed there."""
    cached = getattr(session, "_auto_qlims", None)
    if qmax is None and cached is not None:
        return cached
    qm = qmax if qmax is not None else max(2.0, float(getattr(session.s, "q_max", 2.0)))
    rows, cols = np.nonzero(session.mask[::4, ::4])
    qxy, qz = session.geom.pixel_to_q(cols * 4.0, rows * 4.0)
    ok = np.isfinite(qxy) & np.isfinite(qz) & (np.hypot(qxy, qz) <= qm)
    if not ok.any():
        lims = ((-0.6, 2.0), (0.0, 2.0))
    else:
        lo, hi = float(qxy[ok].min()), float(qxy[ok].max())
        x0 = min(np.floor(lo / step) * step, 0.0); x1 = max(np.ceil(hi / step) * step, 0.0)
        z1 = np.ceil(float(qz[ok].max()) / step) * step
        lims = ((round(x0, 6), round(x1, 6)), (0.0, round(max(z1, step), 6)))
    if qmax is None:
        try:
            session._auto_qlims = lims
        except AttributeError:
            pass
    return lims


def resolve(opts: FigureOptions, session) -> FigureOptions:
    """fill automatic q limits from the detector coverage."""
    if opts.qxy_lim is not None and opts.qz_lim is not None:
        return opts
    ax, az = auto_qlims(session)
    return replace(opts, qxy_lim=opts.qxy_lim or ax, qz_lim=opts.qz_lim or az)


def qgrid_for(session, opts: FigureOptions, cache=None):
    """image resampled on a regular (q_xy, q_z) grid; reuse `cache` when it covers the same limits."""
    opts = resolve(opts, session)
    if cache is not None:
        qxy, qz, _ = cache
        if qxy[0] <= opts.qxy_lim[0] + 1e-9 and qxy[-1] >= opts.qxy_lim[1] - 0.01 and \
                qz[0] <= opts.qz_lim[0] + 1e-9 and qz[-1] >= opts.qz_lim[1] - 0.01:
            return cache
    return to_qgrid(session.D, session.mask, session.geom, opts.qxy_lim, opts.qz_lim, 0.004)


def positive_values(img):
    v = img[np.isfinite(img)]
    return v[v > 0]


def visible(qgrid, opts: FigureOptions):
    """pixels of the q-grid inside the requested q range (so the contrast never depends on a larger cached grid)."""
    qxy, qz, img = qgrid
    cx = (qxy >= opts.qxy_lim[0]) & (qxy <= opts.qxy_lim[1])
    cz = (qz >= opts.qz_lim[0]) & (qz <= opts.qz_lim[1])
    return img[np.ix_(cz, cx)]


def limits(img, opts: FigureOptions):
    """(vmin, vmax) of the colour bar.  Typed values are used exactly as given; missing ones come from percentiles of
    the positive pixels of `img` (pass only the visible region, see visible()).  Raises ValueError for limits that
    cannot be drawn (non-finite, vmin >= vmax, or <= 0 on a log scale) instead of silently changing them."""
    for name in ("vmin", "vmax", "lo_pct", "hi_pct"):
        x = getattr(opts, name)
        if x is not None and not np.isfinite(x):
            raise ValueError(f"{name} must be a finite number.")
    if not 0 <= opts.lo_pct < opts.hi_pct <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    v = positive_values(img)
    if v.size == 0 and (opts.vmin is None or opts.vmax is None):
        raise ValueError("No positive pixels in this q range; type both colour-bar limits.")
    lo = float(opts.vmin) if opts.vmin is not None else float(np.percentile(v, opts.lo_pct))
    hi = float(opts.vmax) if opts.vmax is not None else float(np.percentile(v, opts.hi_pct))
    if opts.scale == "log" and lo <= 0:
        raise ValueError("A log colour bar needs a minimum above 0.")
    if not hi > lo:
        if opts.vmin is None and opts.vmax is None:        # flat image: widen around the single value
            lo, hi = (lo / 1.1, lo * 1.1) if lo > 0 else (lo - 1, lo + 1)
        else:
            raise ValueError(f"The colour-bar maximum ({hi:.4g}) must exceed the minimum ({lo:.4g}).")
    return lo, hi


def draw_map(fig: Figure, session, evaluation, opts: FigureOptions, qgrid=None):
    """draw the map (+ fit overlay + colour bar) into `fig`; returns the q-grid used (for caching)."""
    fig.clear()
    opts = resolve(opts, session)
    qgrid = qgrid_for(session, opts, qgrid)
    qxy, qz, img = qgrid
    lo, hi = limits(visible(qgrid, opts), opts)
    norm = LogNorm(lo, hi) if opts.scale == "log" else Normalize(lo, hi)
    ax = fig.add_subplot(111)
    data = np.clip(img, lo, None) if opts.scale == "log" else img
    im = ax.imshow(data, origin="lower", extent=[qxy[0], qxy[-1], qz[0], qz[-1]], cmap=opts.cmap, norm=norm,
                   aspect="equal", interpolation="nearest")
    ax.set_facecolor("#bdbdbd")                       # masked / missing-wedge pixels
    ax.set_xlabel(r"$q_{xy}$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(r"$q_z$ ($\mathrm{\AA}^{-1}$)")
    ax.set_xlim(opts.qxy_lim); ax.set_ylim(opts.qz_lim)
    best = getattr(evaluation, "best", None)
    if opts.overlay and best is not None:
        plots.draw_multi(ax, best, session.peaks, opts.qxy_lim, opts.qz_lim,
                         label_top=label_count(opts), title=False)
        leg = ax.get_legend()
        if leg is not None and not opts.legend:
            leg.remove()
    elif opts.overlay:
        px, pz = session.geom.pixel_to_q(session.peaks.col, session.peaks.row)
        ax.scatter(px, pz, s=40, facecolors="none", edgecolors="cyan", lw=1.0)
    if opts.title:
        ax.set_title(opts.title, fontsize=12)
    if opts.colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)   # same height as the map
        cb = fig.colorbar(im, cax=cax)
        if opts.scale == "log":
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
            cb.ax.yaxis.set_major_locator(FixedLocator(log_ticks(lo, hi)))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.set_label(opts.cbar_label)
    fig.tight_layout()
    return qgrid


LABEL_CHOICES = ("none", "8", "20", "all")


def label_count(opts: FigureOptions):
    """label_top for plots.draw_multi: 0 = none, None = every indexed spot."""
    if not opts.labels:
        return 0
    return None if opts.n_labels is None else max(0, int(opts.n_labels))


def label_choice(n):
    """FigureOptions.n_labels / app setting -> combobox text."""
    return "all" if n is None else ("none" if n == 0 else str(n))


def parse_label_choice(text):
    """combobox text ('none', '8', '20', 'all' or any integer) -> n_labels."""
    t = str(text).strip().lower()
    if t == "all":
        return None
    if t == "none":
        return 0
    n = int(t)
    if n < 0:
        raise ValueError("Number of hkl labels must be 0 or more.")
    return n


def log_ticks(lo, hi, n_max=7):
    """readable colour-bar ticks for a log scale: 1-2-5 steps (or 1-10 steps over wide ranges), at least two."""
    k0, k1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    for mant in ((1, 2, 5), (1, 3), (1,)):
        t = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in mant if lo <= m * 10.0 ** k <= hi]
        if 2 <= len(t) <= n_max:
            return t
    if len(t) > n_max:
        return t[::int(np.ceil(len(t) / n_max))]
    # narrow range: geometric steps rounded to the fewest digits that keep them distinct and inside [lo, hi]
    for digits in range(2, 8):
        t = sorted({float(f"{x:.{digits}g}") for x in np.geomspace(lo, hi, 4)})
        t = [x for x in t if lo <= x <= hi]
        if len(t) >= 3:
            return t
    return [lo, hi]


def save_map(path, session, evaluation, opts: FigureOptions = None, qgrid=None):
    """write the figure at opts.width_in x opts.height_in and opts.dpi; returns the path."""
    opts = opts or FigureOptions()
    fig = Figure(figsize=(opts.width_in, opts.height_in), dpi=opts.dpi)
    draw_map(fig, session, evaluation, opts, qgrid)
    ext = os.path.splitext(path)[1].lower()
    kw = dict(dpi=opts.dpi, facecolor="white")
    if ext in (".tif", ".tiff"):
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(path, **kw)
    return path


# ---------------------------------------------------------------------- desktop dialog
def open_dialog(parent, session, evaluation, qgrid=None, initial_dir=None, n_labels=8):
    """modal-less Tk dialog: controls on the left, live preview on the right, Save... writes the file."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image, ImageTk

    name = getattr(evaluation, "name", None) or os.path.splitext(os.path.basename(session.path))[0]
    opts = resolve(FigureOptions(title=name, n_labels=n_labels), session)
    state = {"qgrid": qgrid_for(session, opts, qgrid), "job": None}
    win = tk.Toplevel(parent)
    win.title("Export figure - preview")
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{int(sw * 0.8)}x{int(sh * 0.82)}+{int(sw * 0.1)}+{int(sh * 0.05)}")
    left = ttk.Frame(win, padding=8); left.pack(side="left", fill="y")
    # Save / Close first, so they stay visible in a small window
    rb = ttk.Frame(left); rb.pack(fill="x", pady=(0, 4))
    ttk.Button(rb, text="Save...", command=lambda: save()).pack(side="left", fill="x", expand=True)
    ttk.Button(rb, text="Close", command=win.destroy).pack(side="left", fill="x", expand=True, padx=4)
    status = tk.StringVar(value="")
    ttk.Label(left, textvariable=status, foreground="#1f4e79", wraplength=320).pack(anchor="w", pady=(0, 6))
    right = ttk.Frame(win, padding=4); right.pack(side="left", fill="both", expand=True)
    right.pack_propagate(False)                  # the preview image must not resize the window

    v = dict(lo=tk.DoubleVar(value=opts.lo_pct), hi=tk.DoubleVar(value=opts.hi_pct),
             vmin=tk.StringVar(), vmax=tk.StringVar(), scale=tk.StringVar(value=opts.scale),
             cmap=tk.StringVar(value=opts.cmap), overlay=tk.BooleanVar(value=True),
             nlab=tk.StringVar(value=label_choice(n_labels)), legend=tk.BooleanVar(value=True),
             colorbar=tk.BooleanVar(value=True), title=tk.StringVar(value=opts.title),
             cbar=tk.StringVar(value=opts.cbar_label), w=tk.StringVar(value=f"{opts.width_in:g}"),
             h=tk.StringVar(value=f"{opts.height_in:g}"), dpi=tk.StringVar(value=str(opts.dpi)),
             qx0=tk.StringVar(value=f"{opts.qxy_lim[0]:g}"), qx1=tk.StringVar(value=f"{opts.qxy_lim[1]:g}"),
             qz0=tk.StringVar(value=f"{opts.qz_lim[0]:g}"), qz1=tk.StringVar(value=f"{opts.qz_lim[1]:g}"))

    ttk.Label(left, text="Intensity bar", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    ttk.Label(left, text="Lower limit (percentile of pixels)").pack(anchor="w")
    s_lo = tk.Scale(left, from_=0, to=99.9, resolution=0.1, orient="horizontal", variable=v["lo"], length=260)
    s_lo.pack(anchor="w")
    ttk.Label(left, text="Upper limit (percentile of pixels)").pack(anchor="w")
    s_hi = tk.Scale(left, from_=50, to=100, resolution=0.05, orient="horizontal", variable=v["hi"], length=260)
    s_hi.pack(anchor="w")
    r = ttk.Frame(left); r.pack(anchor="w", pady=2)
    ttk.Label(r, text="min").pack(side="left"); e_min = ttk.Entry(r, textvariable=v["vmin"], width=10)
    e_min.pack(side="left", padx=2)
    ttk.Label(r, text="max").pack(side="left"); e_max = ttk.Entry(r, textvariable=v["vmax"], width=10)
    e_max.pack(side="left", padx=2)
    ttk.Label(left, text="(typed values are used exactly; moving a slider releases them)",
              foreground="#555").pack(anchor="w")
    r = ttk.Frame(left); r.pack(anchor="w", pady=4)
    ttk.Label(r, text="Scale").pack(side="left")
    ttk.Combobox(r, textvariable=v["scale"], values=("log", "linear"), state="readonly", width=7).pack(side="left", padx=3)
    ttk.Label(r, text="Colours").pack(side="left")
    ttk.Combobox(r, textvariable=v["cmap"], values=CMAPS, state="readonly", width=9).pack(side="left", padx=3)
    ttk.Button(left, text="Auto limits (30-99.7 %)", command=lambda: auto()).pack(anchor="w", pady=2)

    ttk.Separator(left).pack(fill="x", pady=6)
    ttk.Label(left, text="Content", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    r = ttk.Frame(left); r.pack(anchor="w")
    ttk.Label(r, text="hkl labels (strongest indexed spots)").pack(side="left")
    ttk.Combobox(r, textvariable=v["nlab"], values=LABEL_CHOICES, width=6).pack(side="left", padx=3)
    for key, text in (("overlay", "Fit overlay (predicted / indexed / not indexed)"),
                      ("legend", "Legend"), ("colorbar", "Colour bar")):
        ttk.Checkbutton(left, text=text, variable=v[key], command=lambda: schedule()).pack(anchor="w")
    for key, text in (("title", "Title"), ("cbar", "Colour-bar label")):
        r = ttk.Frame(left); r.pack(fill="x", pady=1)
        ttk.Label(r, text=text, width=14).pack(side="left")
        ttk.Entry(r, textvariable=v[key], width=26).pack(side="left")
    r = ttk.Frame(left); r.pack(anchor="w", pady=2)
    ttk.Label(r, text="q_xy").pack(side="left")
    for key in ("qx0", "qx1"):
        ttk.Entry(r, textvariable=v[key], width=6).pack(side="left", padx=1)
    ttk.Label(r, text="  q_z").pack(side="left")
    for key in ("qz0", "qz1"):
        ttk.Entry(r, textvariable=v[key], width=6).pack(side="left", padx=1)

    ttk.Separator(left).pack(fill="x", pady=6)
    ttk.Label(left, text="Output", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    r = ttk.Frame(left); r.pack(anchor="w")
    for key, text in (("w", "width (in)"), ("h", "height (in)"), ("dpi", "dpi")):
        ttk.Label(r, text=text).pack(side="left")
        ttk.Entry(r, textvariable=v[key], width=6).pack(side="left", padx=2)

    # preview = the output figure itself (same inches, so the same layout), rasterised at a dpi that fits the pane
    fig = Figure(figsize=(opts.width_in, opts.height_in), dpi=90)
    agg = FigureCanvasAgg(fig)
    pane = tk.Label(right, background="#808080", anchor="center")
    pane.pack(fill="both", expand=True)
    ttk.Label(right, text="Preview is the saved figure scaled to fit this window.", foreground="#555").pack(anchor="w")

    def show_preview(*_):
        w_px, h_px = max(pane.winfo_width() - 10, 100), max(pane.winfo_height() - 10, 100)
        w_in, h_in = fig.get_size_inches()
        fig.set_dpi(min(w_px / w_in, h_px / h_in))
        agg.draw()
        state["photo"] = ImageTk.PhotoImage(Image.frombuffer("RGBA", agg.get_width_height(),
                                                             bytes(agg.buffer_rgba()), "raw", "RGBA", 0, 1))
        pane.configure(image=state["photo"])

    def _num(var, kind=float):
        t = var.get().strip()
        if t == "":
            return None
        x = kind(t)
        if not np.isfinite(x):                          # 'nan', 'inf' are parsed by float() but cannot be drawn
            raise ValueError(f"'{t}' is not a finite number.")
        return x

    def current() -> FigureOptions:
        o = replace(opts, lo_pct=float(v["lo"].get()), hi_pct=float(v["hi"].get()), scale=v["scale"].get(),
                    cmap=v["cmap"].get(), overlay=v["overlay"].get(), labels=True, n_labels=parse_label_choice(v["nlab"].get()),
                    legend=v["legend"].get(), colorbar=v["colorbar"].get(), title=v["title"].get(),
                    cbar_label=v["cbar"].get(), vmin=_num(v["vmin"]), vmax=_num(v["vmax"]),
                    qxy_lim=(_num(v["qx0"]), _num(v["qx1"])), qz_lim=(_num(v["qz0"]), _num(v["qz1"])),
                    width_in=_num(v["w"]), height_in=_num(v["h"]), dpi=_num(v["dpi"], int))
        if None in o.qxy_lim + o.qz_lim or not (o.qxy_lim[0] < o.qxy_lim[1] and o.qz_lim[0] < o.qz_lim[1]):
            raise ValueError("q limits must be numbers with min < max.")
        if not (o.width_in and o.height_in and 0.5 <= o.width_in <= 30 and 0.5 <= o.height_in <= 30):
            raise ValueError("Width and height must be 0.5-30 inches.")
        if not (o.dpi and 50 <= o.dpi <= 1200):
            raise ValueError("dpi must be 50-1200.")
        state["qgrid"] = qgrid_for(session, o, state["qgrid"])
        state["lims"] = limits(visible(state["qgrid"], o), o)    # validates the colour-bar limits before drawing
        return o

    def redraw():
        state["job"] = None
        try:
            o = current()
        except (ValueError, TypeError) as e:
            status.set(f"Not drawn: {e}"); return
        fig.set_size_inches(o.width_in, o.height_in)
        state["qgrid"] = draw_map(fig, session, evaluation, o, state["qgrid"])
        lo, hi = state["lims"]
        status.set(f"Colour bar: {lo:.3g} - {hi:.3g} ({o.scale}).  Output {o.width_in:g} x {o.height_in:g} in, "
                   f"{int(o.width_in * o.dpi)} x {int(o.height_in * o.dpi)} px.")
        state["last"] = o
        show_preview()

    def schedule(*_):
        if state["job"] is not None:
            win.after_cancel(state["job"])
        state["job"] = win.after(150, redraw)

    def fix_typed(*_):
        schedule()

    def auto():
        v["vmin"].set(""); v["vmax"].set(""); v["lo"].set(30.0); v["hi"].set(99.7); v["scale"].set("log")
        schedule()

    def release_min(*_):
        v["vmin"].set(""); schedule()

    def release_max(*_):
        v["vmax"].set(""); schedule()

    def save():
        try:
            o = current()
        except (ValueError, TypeError) as e:
            messagebox.showerror("gixfit", str(e), parent=win); return
        base = (getattr(evaluation, "name", None) or "gixfit_map").replace(" ", "_")
        path = filedialog.asksaveasfilename(parent=win, title="Save figure", defaultextension=".png",
                                            initialdir=initial_dir, initialfile=f"{base}_map.png", filetypes=FORMATS)
        if not path:
            return
        save_map(path, session, evaluation, o, state["qgrid"])
        status.set(f"Saved {path}")
        state["saved"] = path

    s_lo.configure(command=release_min)
    s_hi.configure(command=release_max)
    e_min.bind("<Return>", fix_typed); e_max.bind("<Return>", fix_typed)
    e_min.bind("<FocusOut>", fix_typed); e_max.bind("<FocusOut>", fix_typed)
    for key in ("scale", "cmap", "nlab"):
        v[key].trace_add("write", schedule)
    for key in ("title", "cbar", "w", "h", "dpi", "qx0", "qx1", "qz0", "qz1"):
        v[key].trace_add("write", schedule)
    pane.bind("<Configure>", lambda e: schedule_preview())

    def schedule_preview():
        if state.get("pjob") is not None:
            win.after_cancel(state["pjob"])
        state["pjob"] = win.after(120, lambda: (state.update(pjob=None), show_preview()))

    win.update_idletasks()
    redraw()
    win.state = state                            # for scripted tests
    win.v, win.redraw, win.current, win.status = v, redraw, current, status
    return win
