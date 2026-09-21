# -*- coding: utf-8 -*-
"""Тесты настроек, общей копии настроек между потоками и папки данных."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from workers import SharedSettings  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "settings.json"
            settings = config.Settings()
            settings.speed = 1.7
            settings.speaker = "aidar"
            settings.save(path)
            loaded = config.Settings.load(path)
            self.assertEqual(loaded.speed, 1.7)
            self.assertEqual(loaded.speaker, "aidar")

    def test_corrupted_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text("{это не json", encoding="utf-8")
            self.assertEqual(config.Settings.load(path).speed, 1.0)

    def test_out_of_range_values_are_clamped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(
                '{"speed": 99, "volume": -5, "capture_fps": 100, "speaker": "нет_такого", "roi": [1, 2]}',
                encoding="utf-8",
            )
            loaded = config.Settings.load(path)
            self.assertEqual(loaded.speed, 2.0)
            self.assertEqual(loaded.volume, 0.0)
            self.assertEqual(loaded.capture_fps, 8.0)
            self.assertEqual(loaded.speaker, config.DEFAULT_SPEAKER)
            self.assertEqual(len(loaded.roi), 4)

    def test_unknown_fields_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text('{"speed": 1.2, "поле_из_будущего": 1}', encoding="utf-8")
            self.assertEqual(config.Settings.load(path).speed, 1.2)


class SharedSettingsTests(unittest.TestCase):
    def test_snapshot_is_isolated_from_original(self):
        shared = SharedSettings(config.Settings())
        snapshot = shared.snapshot()
        snapshot.roi[0] = 99999  # список roi не должен быть общим между потоками
        self.assertNotEqual(shared.snapshot().roi[0], 99999)

    def test_update_is_visible_in_next_snapshot(self):
        shared = SharedSettings(config.Settings())
        shared.update(speed=1.5)
        self.assertEqual(shared.snapshot().speed, 1.5)


class DataDirTests(unittest.TestCase):
    def test_environment_override_wins(self):
        with mock.patch.dict(os.environ, {"NEUROVOX_HOME": "/tmp/nv_override"}):
            self.assertEqual(config._data_dir(), Path("/tmp/nv_override"))

    def test_frozen_windows_uses_localappdata(self):
        env = {"LOCALAPPDATA": "/tmp/local_app_data"}
        with mock.patch.dict(os.environ, env), mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "platform", "win32"):
            os.environ.pop("NEUROVOX_HOME", None)
            self.assertEqual(config._data_dir(), Path("/tmp/local_app_data") / config.APP_NAME)

    def test_source_run_uses_project_folder(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("NEUROVOX_HOME", None)
            self.assertEqual(config._data_dir(), config._base_dir())


if __name__ == "__main__":
    unittest.main()
