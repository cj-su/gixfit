"""Figure export with an adjustable intensity bar (portable: uses the synthetic tutorial).

python tests/test_figexport.py
"""
import os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from gixfit.demo import create_demo
from gixfit.engine import Session, check_cif
from gixfit.figexport import FigureOptions, save_map, limits, log_ticks, draw_map
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

    def test_limits(self):
        img = np.array([[0, 1, 2], [3, np.nan, 1000]], float)
        self.assertEqual(limits(img, FigureOptions(vmin=5, vmax=50)), (5, 50))          # typed values win
        lo, hi = limits(img, FigureOptions(scale="log", vmin=-3))                          # log floor stays > 0
        self.assertGreater(lo, 0); self.assertGreater(hi, lo)
        lo, hi = limits(img, FigureOptions(vmin=9, vmax=9))                                # never vmin >= vmax
        self.assertGreater(hi, lo)

    def test_log_ticks(self):
        self.assertEqual(log_ticks(1, 1e5), [1, 10, 100, 1000, 1e4, 1e5])
        self.assertGreaterEqual(len(log_ticks(30, 60)), 2)

    def test_colour_bar_follows_limits(self):
        fig = Figure(figsize=(6.5, 5.2), dpi=72)
        draw_map(fig, self.ses, self.ev, FigureOptions(scale="linear", vmin=10, vmax=500))
        ims = [a.images[0] for a in fig.axes if a.images]
        self.assertEqual(ims[0].norm.vmin, 10); self.assertEqual(ims[0].norm.vmax, 500)
        self.assertEqual(len(fig.axes), 2)                                                 # map + colour bar
        draw_map(fig, self.ses, self.ev, FigureOptions(colorbar=False))
        self.assertEqual(len(fig.axes), 1)

    def test_save_formats_and_size(self):
        from PIL import Image
        for ext in ("png", "tif", "pdf", "svg"):
            p = os.path.join(self.tmp.name, f"map.{ext}")
            save_map(p, self.ses, self.ev, FigureOptions(width_in=4, height_in=3, dpi=100))
            self.assertGreater(os.path.getsize(p), 1000, ext)
        self.assertEqual(Image.open(os.path.join(self.tmp.name, "map.png")).size, (400, 300))


if __name__ == "__main__":
    unittest.main(verbosity=1)
