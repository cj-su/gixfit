"""Spot-to-reflection assignment and the matching-window setting.  python tests/test_matching.py"""
import os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from gixfit.fit import match_spots
from gixfit.engine import Settings


class Matching(unittest.TestCase):
    def test_on_axis_reflection_claims_one_spot_per_wedge_edge(self):
        # an on-axis (side 0) reflection at q 1.31, |chi| 6 deg is seen on both edges of the missing wedge
        q, chi = np.array([1.312, 1.314, 1.316]), np.array([6.0, -6.0, 5.5])
        gq, gchi, gI = np.array([1.313]), np.array([5.8]), np.array([1.0])
        pairs = match_spots(q, chi, gq, gchi, gI, [True], 0.015, 4.0, 0.0)
        self.assertEqual(len(pairs), 2)                               # one per edge, not three
        got = sorted(np.sign(chi[i]) for i, _ in pairs)
        self.assertEqual(got, [-1.0, 1.0])
        self.assertEqual({j for _, j in pairs}, {0})

    def test_off_axis_reflection_is_one_to_one_and_signed(self):
        q, chi = np.array([0.76, 0.76]), np.array([63.5, -63.5])
        pairs = match_spots(q, chi, np.array([0.76]), np.array([63.5]), np.array([1.0]), [False], 0.015, 4.0, 0.0)
        self.assertEqual(pairs, [(0, 0)])                             # the -chi spot is a different reflection

    def test_window_gate(self):
        q, chi = np.array([1.313]), np.array([6.0])
        g = (np.array([1.342]), np.array([5.5]), np.array([1.0]), [True])
        self.assertEqual(match_spots(q, chi, *g, 0.015, 4.0, 0.0), [])           # 0.029 A^-1 off: outside strict
        self.assertEqual(len(match_spots(q, chi, *g, 0.03, 6.0, 0.0)), 1)       # inside the loose window

    def test_settings_window_schedule(self):
        s = Settings()
        self.assertEqual(s.tol_schedule()[-1], (0.015, 4.0))
        s.match_q, s.match_chi = 0.03, 6.0
        self.assertEqual(s.tol_schedule(), ((0.05, 10.0), (0.03, 6.0)))
        s.match_q = 0.5
        with self.assertRaises(ValueError):
            s.validate()


if __name__ == "__main__":
    unittest.main(verbosity=1)
