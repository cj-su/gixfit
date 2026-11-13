"""Portable regression tests for session state, ranking, validation and report replay."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gixfit.demo import create_demo
from gixfit.engine import Settings, Session, check_cif, _rank
from gixfit.replay import replay, write_manifest


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_root = Path(__file__).resolve().parents[1] / "runs"
        test_root.mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=test_root)
        cls.image, cls.cif, cls.settings = create_demo(cls.tmp.name)
        cls.session = Session(cls.image, cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_session_owns_settings_and_detects_hidden_geometry_changes(self):
        s = Settings(**vars(self.settings))
        s.mask_rects = [[0, 10, 0, 10]]
        session = Session(self.image, s)
        original = session.geom.x0
        s.x0 += 2
        s.mask_rects[0][0] = 3
        self.assertEqual(session.geom.x0, original)
        self.assertEqual(session.s.mask_rects[0][0], 0)
        self.assertFalse(session.matches_inputs(self.image, s))
        with self.assertRaisesRegex(ValueError, "find spots again"):
            check_cif(session, self.cif, s)

    def test_changed_image_invalidates_session(self):
        self.assertFalse(self.session.matches_inputs(self.image + ".different", self.settings))

    def test_replaced_image_at_same_path_invalidates_session(self):
        image = Path(self.tmp.name) / "replaceable.tiff"
        shutil.copy2(self.image, image)
        session = Session(str(image), self.settings)
        with image.open("ab") as stream:
            stream.write(b"changed")
        self.assertFalse(session.matches_inputs(str(image), self.settings))
        with self.assertRaisesRegex(ValueError, "find spots again"):
            check_cif(session, self.cif)
        image.unlink()
        self.assertFalse(session.matches_inputs(str(image), self.settings))

    def test_ranking_is_fit_quality_not_diagnostic_grade(self):
        strong = SimpleNamespace(fom=.8, grade="neither")
        weak = SimpleNamespace(fom=.1, grade="both")
        self.assertIs(_rank([weak, strong])[0], strong)
        strong.grade, weak.grade = "untested", "chi"
        self.assertIs(_rank([weak, strong])[0], strong)

    def test_invalid_settings_are_rejected(self):
        for change in ({"sdd_mm": -1}, {"wavelength": float("nan")}, {"q_min": 2, "q_max": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Settings(**change).validate()

    def test_relative_settings_path(self):
        path = Path(self.image).parent / "settings.toml"
        loaded, image, _ = Settings.from_toml(path)
        self.assertEqual(image, self.image)
        self.assertEqual(loaded.n_null, 0)

    def test_portable_replay_and_tamper_detection(self):
        ev = check_cif(self.session, self.cif, contact=(1, 0, 0))
        folder = Path(self.tmp.name) / "report"
        manifest = write_manifest(self.session, [ev], [], str(folder))
        record = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(record["candidates"][0]["recipe"]["contact"], [1, 0, 0])
        moved = Path(self.tmp.name) / "moved"
        shutil.copytree(folder, moved)
        _, again = replay(str(moved / "run_manifest.json"), str(Path(self.tmp.name) / "replayed"), log=lambda _: None)
        self.assertAlmostEqual(ev.fom, again[0].fom, places=12)
        self.assertEqual(ev.best.components, again[0].best.components)
        import numpy as np
        np.testing.assert_allclose(ev.best.cells[0].array(), again[0].best.cells[0].array(), atol=1e-10)
        image = moved / record["image"]["path"]
        with image.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            replay(str(moved / "run_manifest.json"), str(Path(self.tmp.name) / "bad"))


class GuiStateTests(unittest.TestCase):
    def test_field_read_does_not_mutate_settings_on_error(self):
        from gixfit.app import App, FIELDS, WINDOWS
        class Variable:
            def __init__(self, value): self.value = str(value)
            def get(self): return self.value
        original = Settings(alpha_i_deg=.12)
        ui = SimpleNamespace(settings=original,
            vars={name: Variable(getattr(original, name)) for name, _, _ in FIELDS},
            v_tex=Variable(1), v_maxidx=Variable(2), v_chance=Variable("Off (fast fit)"), v_top=Variable(3),
            v_window=Variable(next(iter(WINDOWS))))
        ui.vars["x0"].value = "123"
        ui.vars["y0"].value = "invalid"
        with self.assertRaises(ValueError):
            App._read_fields(ui)
        self.assertEqual(original.x0, 577)
        ui.vars["y0"].value = "456"
        result = App._read_fields(ui)
        self.assertEqual(result.x0, 123)
        self.assertEqual(original.x0, 577)

    def test_stale_gui_fit_is_blocked(self):
        from gixfit.app import App
        s = Settings()
        session = SimpleNamespace(matches_inputs=lambda path, settings: False)
        ui = SimpleNamespace(session=session, _read_fields=lambda: s,
                             v_image=SimpleNamespace(get=lambda: "new.tiff"),
                             v_status=SimpleNamespace(set=lambda value: None))
        with patch("gixfit.app.messagebox.showinfo") as info:
            self.assertIsNone(App._apply_options(ui))
            info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
