"""Two crystals fitted together (engine.joint_fit), portable: two invented P1 crystals, synthetic spot lists.

python tests/test_joint.py
"""
import os, sys, unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from gixfit.crystal import Cell, Structure
from gixfit.engine import Settings, joint_fit
from gixfit.peaks import PeakTable
from test_synthetic import make_peaks

SHAPE = (1043, 981)


def crystal(cell, seed, name):
    rng = np.random.default_rng(seed)
    return Structure(Cell(*cell), ["C"] * 12, rng.random((12, 3)), name=name)


def concat(*tables):
    return PeakTable(**{k: np.concatenate([getattr(t, k) for t in tables]) for k in PeakTable.__dataclass_fields__})


class JointFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A = crystal((9.1, 11.3, 13.2, 82, 98, 104), 42, "crystal A")
        cls.B = crystal((7.4, 8.8, 15.6, 95, 101, 89), 7, "crystal B")
        s = Settings(x0=490, y0=640, alpha_i_deg=0.12, q_min=0.35, q_max=1.85, planes_maxidx=1, n_null=0, workers=1)
        geom = s.geometry()
        mask = np.ones(SHAPE, bool); mask[595:] = False
        pk = concat(make_peaks(cls.A, (1, 0, 0), geom, SHAPE, n_strong=30, n_spurious=0, seed=1),
                    make_peaks(cls.B, (0, 0, 1), geom, SHAPE, n_strong=25, n_spurious=8, seed=2))
        cls.ses = SimpleNamespace(path="synthetic", s=s, geom=geom, peaks=pk, mask=mask,
                                  matches_inputs=lambda path, settings: True)
        cls.jf = joint_fit(cls.ses, [cls.A, cls.B], s)

    def test_both_crystals_and_planes(self):
        comps = {cm["phase"]: cm for cm in self.jf.best.comp_metrics}
        self.assertEqual(set(comps), {"crystal A", "crystal B"})
        self.assertIn(comps["crystal A"]["contact"], ("(1, 0, 0)", "(-1, 0, 0)"))
        self.assertIn(comps["crystal B"]["contact"], ("(0, 0, 1)", "(0, 0, -1)"))
        self.assertGreaterEqual(comps["crystal A"]["n_match"], 20)
        self.assertGreaterEqual(comps["crystal B"]["n_match"], 15)
        self.assertEqual(self.jf.forced, [])

    def test_each_spot_assigned_once(self):
        obs = [r["obs"] for r in self.jf.best.pairs]
        self.assertEqual(len(obs), len(set(obs)))                 # crystals compete: no spot used twice

    def test_phase_map_and_summary(self):
        self.assertEqual(len(self.jf.rows), len({r["spot"] for r in self.jf.rows}))
        labels = {r["label"] for r in self.jf.rows}
        self.assertTrue({"crystal A", "crystal B"} <= labels)
        self.assertIn("phase map: crystal A", self.jf.verdict)
        self.assertIn("crystal B (0, 0, 1)", self.jf.report.replace("(0, 0, -1)", "(0, 0, 1)"))
        self.assertTrue(np.isfinite(self.jf.fom) and 0 < self.jf.fom <= 1)

    def test_selection_order_does_not_matter(self):
        """reversing the order of the crystals gives the same crystals, colours (= phase order) and assignment."""
        jf2 = joint_fit(self.ses, [self.B, self.A], self.ses.s)
        self.assertEqual(jf2.names, self.jf.names)
        self.assertEqual(jf2.best.phase_names, self.jf.best.phase_names)
        key = lambda j: sorted((r["obs"], r["phase"]) for r in j.best.pairs)
        self.assertEqual(key(jf2), key(self.jf))

    def test_phase_map_covers_fitted_q_range_only(self):
        q = [r["q"] for r in self.jf.rows]
        self.assertTrue(all(self.ses.s.q_min <= x <= self.ses.s.q_max for x in q))
        self.assertEqual(self.jf.n_detected, len(self.ses.peaks))
        self.assertIn(f"{len(self.jf.rows)} spots in the fitted q range", self.jf.report)

    def test_needs_two(self):
        with self.assertRaises(ValueError):
            joint_fit(self.ses, [self.A], self.ses.s)


if __name__ == "__main__":
    unittest.main(verbosity=1)
