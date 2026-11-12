"""Export report: only the chosen parts are written (portable: synthetic tutorial).

python tests/test_export.py
"""
import os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gixfit.demo import create_demo
from gixfit.engine import Session, Settings, check_cif, export, EXPORT_PARTS


class ExportParts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        image, cif, s = create_demo(cls.tmp.name)
        cls.ses = Session(image, s)
        cls.ev = check_cif(cls.ses, cif, contact=(1, 0, 0))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_export(self, name, parts):
        d = os.path.join(self.tmp.name, name)
        written = export(self.ses, [self.ev], [], d, parts=parts)
        return d, sorted(os.listdir(d)), written

    def test_everything_by_default(self):
        d, files, written = self.run_export("all", None)
        for f in ("gixfit_results.xlsx", "summary.txt", "settings.toml", "run_manifest.json", "inputs"):
            self.assertIn(f, files)
        self.assertTrue(any(f.endswith("_map.png") for f in files))
        self.assertTrue(any(f.endswith("_refined.cif") for f in files))
        self.assertEqual(sorted(written), files)

    def test_maps_only(self):
        d, files, _ = self.run_export("maps", {"maps"})
        self.assertTrue(files and all(f.endswith("_map.png") for f in files), files)

    def test_no_refined_cifs_no_replay(self):
        d, files, _ = self.run_export("no_cif", {"workbook", "summary", "maps", "settings"})
        self.assertFalse(any(f.endswith(".cif") for f in files))
        self.assertNotIn("run_manifest.json", files)
        self.assertNotIn("inputs", files)                       # no copy of the image

    def test_settings_alone_reload(self):
        d, files, _ = self.run_export("settings", {"settings"})
        self.assertEqual(files, ["settings.toml"])
        s, img, _ = Settings.from_toml(os.path.join(d, "settings.toml"))
        self.assertTrue(os.path.isfile(img))                     # points at the original image
        self.assertEqual((s.x0, s.y0), (self.ses.s.x0, self.ses.s.y0))

    def test_replay_without_settings_file(self):
        d, files, _ = self.run_export("replay", {"replay"})
        self.assertIn("run_manifest.json", files)
        self.assertNotIn("settings.toml", files)

    def test_unknown_part(self):
        with self.assertRaises(ValueError):
            export(self.ses, [self.ev], [], os.path.join(self.tmp.name, "bad"), parts={"pictures"})
        self.assertEqual(set(EXPORT_PARTS), {"workbook", "summary", "maps", "cifs", "settings", "replay"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
