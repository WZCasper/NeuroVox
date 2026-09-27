# -*- coding: utf-8 -*-
"""Тесты настроек, общей копии настроек между потоками и папки данных."""

import json
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
                '{"settings_version": 2, "speed": 99, "volume": -5, "capture_fps": 100, '
                '"speaker": "нет_такого", "rois": [[1, 2]], "area_pause": 99}',
                encoding="utf-8",
            )
            loaded = config.Settings.load(path)
            self.assertEqual(loaded.speed, 2.0)
            self.assertEqual(loaded.volume, 0.0)
            self.assertEqual(loaded.capture_fps, config.MAX_CAPTURE_FPS)
            self.assertEqual(loaded.area_pause, config.MAX_AREA_PAUSE)
            self.assertEqual(loaded.speaker, config.DEFAULT_SPEAKER)
            self.assertEqual(len(loaded.rois), 1)          # неверная область заменена областью по умолчанию
            self.assertEqual(len(loaded.rois[0]), 4)

    def test_unknown_fields_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text('{"speed": 1.2, "поле_из_будущего": 1}', encoding="utf-8")
            self.assertEqual(config.Settings.load(path).speed, 1.2)


class AreasAndMigrationTests(unittest.TestCase):
    @staticmethod
    def _load(text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(text, encoding="utf-8")
            return config.Settings.load(path)

    def test_several_areas_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            settings = config.Settings()
            settings.rois = [[10, 20, 300, 60], [10, 120, 300, 60], [500, 700, 800, 90]]
            settings.area_pause = 2.5
            settings.save(path)
            loaded = config.Settings.load(path)
            self.assertEqual(loaded.rois, settings.rois)
            self.assertEqual(loaded.area_pause, 2.5)

    def test_old_single_roi_becomes_first_area(self):
        loaded = self._load('{"roi": [11, 22, 333, 44]}')
        self.assertEqual(loaded.rois, [[11, 22, 333, 44]])
        self.assertFalse(hasattr(loaded, "roi"))

    def test_bad_areas_are_dropped_and_extra_ones_cut(self):
        good = [[i, i, 100, 50] for i in range(config.MAX_AREAS + 3)]
        junk = [[1, 2], "x", None, [1, 2, 3, True], [1, 2, 3, "4"]]
        # Мусорные записи стоят ПЕРЕД верными: их надо отбросить, а верных должно остаться ровно MAX_AREAS.
        loaded = self._load(json.dumps({"settings_version": 2, "rois": junk + good}))
        self.assertEqual(len(loaded.rois), config.MAX_AREAS)
        self.assertTrue(all(len(roi) == 4 for roi in loaded.rois))
        self.assertEqual(loaded.rois[0], [0, 0, 100, 50])       # первая верная область — на первом месте

    def test_no_valid_area_gives_default(self):
        loaded = self._load('{"settings_version": 2, "rois": []}')
        self.assertEqual(loaded.rois, [list(config.DEFAULT_ROI)])

    def test_area_sizes_have_a_minimum(self):
        loaded = self._load('{"settings_version": 2, "rois": [[0, 0, 1, 1]]}')
        self.assertGreaterEqual(loaded.rois[0][2], 40)
        self.assertGreaterEqual(loaded.rois[0][3], 20)

    def test_old_file_gets_fast_capture_and_new_voice_model(self):
        # Файл версии 1: медленный захват 3,5 кадра/с и прежняя модель по умолчанию.
        loaded = self._load('{"capture_fps": 3.5, "tts_model": "v4_ru", "speed": 1.3}')
        self.assertEqual(loaded.capture_fps, config.DEFAULT_CAPTURE_FPS)
        self.assertEqual(loaded.tts_model, config.DEFAULT_TTS_MODEL)
        self.assertEqual(loaded.speed, 1.3)                       # остальное не трогаем
        self.assertEqual(loaded.settings_version, config.SETTINGS_VERSION)

    def test_removed_model_falls_back_to_default(self):
        loaded = self._load('{"settings_version": 2, "tts_model": "v5_ru"}')
        self.assertEqual(loaded.tts_model, config.DEFAULT_TTS_MODEL)

    def test_new_file_keeps_users_choice_of_old_model(self):
        # Уже перенесённые настройки: явный выбор v4_ru пользователем уважаем.
        loaded = self._load('{"settings_version": 2, "tts_model": "v4_ru", "capture_fps": 20}')
        self.assertEqual(loaded.tts_model, "v4_ru")
        self.assertEqual(loaded.capture_fps, 20.0)

    def test_defaults_use_fast_capture_and_latest_model(self):
        settings = config.Settings()
        self.assertGreaterEqual(settings.capture_fps, 10)
        self.assertEqual(settings.tts_model, "v5_5_ru")
        self.assertIn("v5_5_ru", config.SILERO_MODELS)
        self.assertTrue(config.SILERO_MODELS["v5_5_ru"].endswith("/v5_5_ru.pt"))


class SharedSettingsTests(unittest.TestCase):
    def test_snapshot_is_isolated_from_original(self):
        shared = SharedSettings(config.Settings())
        snapshot = shared.snapshot()
        snapshot.rois[0][0] = 99999  # список областей не должен быть общим между потоками
        self.assertNotEqual(shared.snapshot().rois[0][0], 99999)

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
