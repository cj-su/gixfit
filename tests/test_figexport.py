"""Figure export with an adjustable intensity bar (portable: uses the synthetic tutorial).

python tests/test_figexport.py
"""
import os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from gixfit.demo import create_demo
from gixfit.engine import Session, check_cif
from gixfit.figexport import FigureOptions, save_map, limits, log_ticks, draw_map, qgrid_for, visible
from matplotlib.figure import Figure


class FigExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        image, cif, s = create_demo(cls.tmp.name)
        cls.ses = Session(image, s)
        cls.ev = check_cif(cls.ses, cif, contact=(1, 0, 0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_typed_limits_are_used_exactly(self):
        img = np.full((10, 10), 50.0); img[0, 0] = 10; img[9, 9] = 1000
        self.assertEqual(limits(img, FigureOptions(vmin=5, vmax=50)), (5, 50))
        # below the smallest pixel on a log scale: kept, not raised to the image minimum (was 10-10.1)
        self.assertEqual(limits(img, FigureOptions(scale="log", vmin=1, vmax=5)), (1, 5))

    def test_impossible_limits_raise(self):
        img = np.array([[0, 1, 2], [3, np.nan, 1000]], float)
        for o in (FigureOptions(scale="log", vmin=-3, vmax=10), FigureOptions(vmin=9, vmax=9),
                  FigureOptions(vmin=float("nan"), vmax=100), FigureOptions(vmax=float("inf")),
                  FigureOptions(lo_pct=float("nan"))):
            with self.assertRaises(ValueError, msg=str(o)):
                limits(img, o)

    def test_auto_contrast_ignores_cached_area(self):
        """same q range -> same colour limits, whether the cached grid is the exact range or a larger one."""
        o = FigureOptions(qxy_lim=(0.8, 1.4), qz_lim=(0.1, 0.9))
        big = qgrid_for(self.ses, FigureOptions(qxy_lim=(-0.6, 2.0), qz_lim=(0.0, 2.0)))
        exact = qgrid_for(self.ses, o)
        self.assertIs(qgrid_for(self.ses, o, big), big)                    # the larger cache is reused...
        a, b = limits(visible(big, o), o), limits(visible(exact, o), o)    # ...but only visible pixels count
        np.testing.assert_allclose(a, b, rtol=0.05)

    def test_log_ticks(self):
        self.assertEqual(log_ticks(1, 1e5), [1, 10, 100, 1000, 1e4, 1e5])
        for lo, hi in ((30, 60), (101, 102), (1000.5, 1001), (2.2, 2.9)):
            t = log_ticks(lo, hi)
            self.assertGreaterEqual(len(t), 2, (lo, hi))
            self.assertEqual(len(t), len(set(t)), (lo, hi, t))              # no repeated ticks (was 100,100,..)
            self.assertTrue(all(lo <= x <= hi for x in t), (lo, hi, t))

    def test_colour_bar_follows_limits(self):
        fig = Figure(figsize=(6.5, 5.2), dpi=72)
        draw_map(fig, self.ses, self.ev, FigureOptions(scale="linear", vmin=10, vmax=500))
        ims = [a.images[0] for a in fig.axes if a.images]
        self.assertEqual(ims[0].norm.vmin, 10); self.assertEqual(ims[0].norm.vmax, 500)
        self.assertEqual(len(fig.axes), 2)                                                 # map + colour bar
        draw_map(fig, self.ses, self.ev, FigureOptions(colorbar=False))
        self.assertEqual(len(fig.axes), 1)

    def test_number_of_hkl_labels(self):
        from gixfit.figexport import parse_label_choice, label_choice
        n_idx = len(self.ev.best.pairs)
        counts = {}
        for n in (0, 3, 8, None):
            fig = Figure(figsize=(6.5, 5.2), dpi=72)
            draw_map(fig, self.ses, self.ev, FigureOptions(n_labels=n, qxy_lim=(-2, 3), qz_lim=(0, 3)))
            counts[n] = len(fig.axes[0].texts)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[3], min(3, n_idx))
        self.assertEqual(counts[None], n_idx)                      # 'all' labels every indexed spot
        self.assertGreater(counts[None], counts[8])                 # tutorial: 38 indexed > 8
        self.assertEqual([parse_label_choice(t) for t in ("none", "8", "20", "all", " 12 ")], [0, 8, 20, None, 12])
        self.assertEqual([label_choice(n) for n in (0, 8, None)], ["none", "8", "all"])
        with self.assertRaises(ValueError):
            parse_label_choice("-1")

    def test_labels_follow_the_visible_range(self):
        """a cropped view still labels the strongest indexed spots that are INSIDE it."""
        best = self.ev.best
        pos = [(r, *map(float, best.geom.pixel_to_q(r["col"], r["row"]))) for r in best.pairs]
        weakest = min(pos, key=lambda t: t[0]["I_pred"])                  # never among the global top 8
        _, x, z = weakest
        o = FigureOptions(n_labels=8, qxy_lim=(x - 0.02, x + 0.02), qz_lim=(max(0, z - 0.02), z + 0.02))
        fig = Figure(figsize=(4, 3), dpi=72)
        draw_map(fig, self.ses, self.ev, o)
        inside = sum(1 for _, px, pz in pos if o.qxy_lim[0] < px < o.qxy_lim[1] and o.qz_lim[0] < pz < o.qz_lim[1])
        self.assertGreaterEqual(inside, 1)
        self.assertEqual(len(fig.axes[0].texts), min(8, inside))

    def test_label_budget_is_shared_by_components(self):
        """two textures/crystals: '8' means 8 labels in total, the strongest across both."""
        from types import SimpleNamespace
        from matplotlib.figure import Figure as F
        from gixfit import plots
        best = self.ev.best
        pairs = [dict(r, component=k % 2, I_pred=float(len(best.pairs) - k)) for k, r in enumerate(best.pairs)]
        res = SimpleNamespace(group_pos=[([], [], []), ([], [], [])], groups=[[], []], pairs=pairs, geom=best.geom,
                              unmatched_obs=[], metrics=best.metrics,
                              comp_metrics=[dict(phase=p, contact=(1, 0, 0), n_match=0, n_strong_missing=0,
                                                 n_strong_pred=20) for p in ("A", "B")])
        for n, want in ((8, 8), (3, 3), (None, len(pairs)), (0, 0)):
            ax = F().add_subplot(111)
            self.assertEqual(plots.draw_multi(ax, res, self.ses.peaks, (-5, 5), (0, 5), label_top=n, title=False), want)
            self.assertEqual(len(ax.texts), want)
        ax = F().add_subplot(111)
        plots.draw_multi(ax, res, self.ses.peaks, (-5, 5), (0, 5), label_top=4, title=False)
        top4 = sorted(pairs, key=lambda r: -r["I_pred"])[:4]                # strongest overall, from both components
        self.assertEqual(sorted(t.get_text() for t in ax.texts), sorted(r["label"].split("/")[0] for r in top4))
        self.assertEqual({r["component"] for r in top4}, {0, 1})

    def test_matched_peaks_only(self):
        """export only peaks found in both the image and the structure: one circle per indexed spot, no x."""
        def drawn(mo):
            fig = Figure(figsize=(6.5, 5.2), dpi=72)
            draw_map(fig, self.ses, self.ev, FigureOptions(matched_only=mo, qxy_lim=(-2, 3), qz_lim=(0, 3)))
            ax = fig.axes[0]
            circles = sum(len(c.get_offsets()) for c in ax.collections
                          if len(c.get_facecolor()) == 0 or (c.get_facecolor()[:, 3] == 0).all())
            xs = sum(len(l.get_xdata()) for l in ax.lines if l.get_marker() in ("x", "+"))
            return circles, xs
        best = self.ev.best
        all_c, _ = drawn(False)
        only_c, only_x = drawn(True)
        self.assertEqual(only_c, len(best.pairs))                   # one predicted circle per matched spot
        self.assertEqual(only_x, 0)                                  # circles only: no + and no x
        self.assertGreater(all_c, only_c)                            # predicted-only circles were dropped
        self.assertTrue(all("group" in r for r in best.pairs))

    def test_save_formats_and_size(self):
        from PIL import Image
        for ext in ("png", "tif", "pdf", "svg"):
            p = os.path.join(self.tmp.name, f"map.{ext}")
            save_map(p, self.ses, self.ev, FigureOptions(width_in=4, height_in=3, dpi=100))
            self.assertGreater(os.path.getsize(p), 1000, ext)
        with Image.open(os.path.join(self.tmp.name, "map.png")) as im:
            self.assertEqual(im.size, (400, 300))

    def test_dialog_rejects_bad_input(self):
        import tkinter as tk
        from gixfit.figexport import open_dialog
        root = tk.Tk(); root.withdraw()
        try:
            win = open_dialog(root, self.ses, self.ev)
            for vmin, vmax, scale in (("nan", "", "log"), ("1", "5", "log"), ("0", "100", "log"), ("50", "10", "linear")):
                win.v["vmin"].set(vmin); win.v["vmax"].set(vmax); win.v["scale"].set(scale)
                win.redraw(); root.update()
                ok = (vmin, vmax) == ("1", "5")
                self.assertEqual(win.status.get().startswith("Not drawn"), not ok, (vmin, vmax, scale, win.status.get()))
                if ok:
                    self.assertEqual(win.state["lims"], (1.0, 5.0))
            win.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main(verbosity=1)
