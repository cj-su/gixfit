"""EDF (laboratory GIWAXS, e.g. Xeuss) reading and header geometry; portable: writes its own EDF files.

python tests/test_edf.py
"""
import os, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from gixfit.edf import read_edf, read_header, geometry_from_header, is_edf
from gixfit.peaks import load_image
from gixfit.engine import Settings


def write_edf(path, img, dtype="<f4", datatype="FloatValue", byteorder="LowByteFirst", extra=None, sized=True):
    keys = {"HeaderID": "EH:000001:000000:000000", "ByteOrder": byteorder, "DataType": datatype,
            "Dim_1": img.shape[1], "Dim_2": img.shape[0], "Size": img.size * np.dtype(dtype).itemsize,
            "Center_1": "602.7015", "Center_2": "899.9432", "PSize_1": "7.5e-05", "PSize_2": "7.5e-05",
            "SampleDistance": "0.0750011", "WaveLength": "1.54189e-10", "om": "0.0998998",
            "det0x_dial": "8.24706", "det0z_dial": "-15.7797"}
    keys.update(extra or {})
    body = "".join(f"{k} = {v} ;\n" for k, v in keys.items())
    hs = 2560
    if sized:
        body = f"EDF_HeaderSize = {hs:5d} ;\n" + body
    head = ("{\n" + body).encode("latin1")
    head = head + b" " * (hs - len(head) - 2) + b"}\n"
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(np.ascontiguousarray(img, dtype=dtype).tobytes())


class EDF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        self.img = rng.poisson(20, (40, 30)).astype(np.float32)
        self.img[10:12] = -1                                  # detector gap pixels

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_float_and_ints(self):
        for dt, name, order in (("<f4", "FloatValue", "LowByteFirst"), ("<u2", "UnsignedShort", "LowByteFirst"),
                                (">i4", "SignedInteger", "HighByteFirst"), ("<f8", "DoubleValue", "LowByteFirst")):
            p = os.path.join(self.tmp.name, f"t_{name}.edf")
            src = np.clip(self.img, 0, None) if dt.endswith("u2") else self.img
            write_edf(p, src, dt, name, order)
            h, img = read_edf(p)
            self.assertEqual(img.shape, (40, 30))
            np.testing.assert_allclose(img, src.astype(np.float64))
        self.assertTrue(is_edf("a/b/HaP.EDF") and not is_edf("x.tiff"))

    def test_header_without_size_key(self):
        p = os.path.join(self.tmp.name, "nosize.edf")
        write_edf(p, self.img, sized=False)
        np.testing.assert_allclose(read_edf(p)[1], self.img)

    def test_load_image_dispatch(self):
        p = os.path.join(self.tmp.name, "img.edf")
        write_edf(p, self.img)
        D = load_image(p, despike=False)
        np.testing.assert_allclose(D, self.img)

    def test_geometry_from_header(self):
        p = os.path.join(self.tmp.name, "g.edf")
        write_edf(p, self.img)
        geo, notes = geometry_from_header(read_header(p))
        self.assertAlmostEqual(geo["wavelength"], 1.54189, 5)
        self.assertAlmostEqual(geo["sdd_mm"], 75.0011, 3)
        self.assertAlmostEqual(geo["pixel_mm"], 0.075, 6)
        self.assertEqual((geo["x0"], geo["y0"]), (602.7015, 899.9432))
        self.assertAlmostEqual(geo["alpha_i_deg"], 0.0999, 4)
        self.assertIn("not updated", notes[0])
        # reference at known motors: centre shifted by motor offset / pixel (Xeuss: x grows, y falls)
        geo2, notes2 = geometry_from_header(read_header(p), [602.69, 899.99, 7.57515, -15.1504])
        self.assertAlmostEqual(geo2["x0"], 602.69 + (8.24706 - 7.57515) / 0.075, 6)
        self.assertAlmostEqual(geo2["y0"], 899.99 - (-15.7797 + 15.1504) / 0.075, 6)

    def test_incident_angle_from_header_when_blank(self):
        p = os.path.join(self.tmp.name, "noangle_in_name.edf")
        write_edf(p, self.img)
        s = Settings(alpha_i_deg=None)
        self.assertAlmostEqual(s.geometry(p).alpha_i_deg, 0.0999, 4)


if __name__ == "__main__":
    unittest.main(verbosity=1)
