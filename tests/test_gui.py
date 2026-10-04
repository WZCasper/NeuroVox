# -*- coding: utf-8 -*-
"""
Тесты обновления меню выбора голоса в gui.MainWindow по реальному списку
голосов загруженной модели (NeuroVoxEngine.available_speakers()).

Требуют рабочего графического дисплея (создают настоящее окно CustomTkinter) —
если дисплея нет (TclError при создании Tk), весь модуль пропускается, а не
падает: это не редкий случай для headless Linux-окружений без Xvfb, на
Windows-раннере CI рабочий стол доступен и тесты должны выполняться.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import tkinter

    _probe = tkinter.Tk()
    _probe.destroy()
    _DISPLAY_AVAILABLE = True
except Exception:  # noqa: BLE001 — любая причина недоступности дисплея
    _DISPLAY_AVAILABLE = False

if _DISPLAY_AVAILABLE:
    import gui  # noqa: E402


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class VoiceMenuRefreshTests(unittest.TestCase):
    """
    gui.MainWindow._refresh_voice_menu() должен обновлять меню голосов по
    NeuroVoxEngine.available_speakers(), а не оставаться на статичном
    config.SILERO_SPEAKERS после загрузки модели.
    """

    def setUp(self):
        self.win = gui.MainWindow()
        self.win.update()

    def tearDown(self):
        # Отменяем все запланированные self.after(...) (в т.ч. таймерный цикл
        # _poll_events, который реальное приложение никогда не останавливает
        # само) — иначе они пытаются сработать уже после win.destroy() и
        # засоряют вывод безобидным, но лишним "invalid command name".
        try:
            for after_id in self.win.tk.eval("after info").split():
                self.win.after_cancel(after_id)
        except tkinter.TclError:
            pass
        # win.destroy() напрямую, в обход _on_close(): тот сохраняет
        # settings.json на диск, что в тесте не нужно и не должно происходить.
        try:
            for overlay in self.win.overlays:
                overlay.destroy()
            self.win.destroy()
        except tkinter.TclError:
            pass

    def test_menu_starts_with_static_config_list(self):
        import config
        labels = set(self.win.voice_menu.cget("values"))
        self.assertEqual(labels, set(config.SILERO_SPEAKERS.values()))

    def test_refresh_shows_voice_not_in_static_config(self):
        # Как настоящая v4_ru: пять «именных» голосов плюс random, которого
        # нет в config.SILERO_SPEAKERS.
        self.win.engine.available_speakers = lambda: [
            "aidar", "baya", "kseniya", "xenia", "eugene", "random",
        ]
        self.win._refresh_voice_menu()
        self.win.update()
        values = self.win.voice_menu.cget("values")
        self.assertIn("random", values)
        self.assertEqual(len(values), 6)

    def test_refresh_keeps_current_choice_when_still_available(self):
        self.win.settings.speaker = "baya"
        self.win.voice_menu.set(self.win._voice_label("baya"))
        self.win.engine.available_speakers = lambda: ["aidar", "baya", "kseniya"]
        self.win._refresh_voice_menu()
        self.win.update()
        self.assertEqual(self.win.settings.speaker, "baya")
        self.assertEqual(self.win.voice_menu.get(), self.win._voice_label("baya"))

    def test_refresh_switches_away_from_unavailable_choice(self):
        # Выбранный голос отсутствует в новой модели — интерфейс должен
        # переключиться на тот же DEFAULT_SPEAKER, что молча подставляет и
        # SileroTts.synthesize(), а не продолжать показывать недоступный голос.
        import config
        self.win.settings.speaker = "eugene"
        self.win.voice_menu.set(self.win._voice_label("eugene"))
        self.win.engine.available_speakers = lambda: ["aidar", "baya"]
        self.win._refresh_voice_menu()
        self.win.update()
        self.assertEqual(self.win.settings.speaker, config.DEFAULT_SPEAKER)
        self.assertEqual(self.win.shared.snapshot().speaker, config.DEFAULT_SPEAKER)
        self.assertEqual(self.win.voice_menu.get(), self.win._voice_label(config.DEFAULT_SPEAKER))

    def test_refresh_with_empty_speaker_list_does_not_crash(self):
        self.win.engine.available_speakers = lambda: []
        self.win._refresh_voice_menu()  # не должно бросить исключение
        self.win.update()

    def test_ready_event_triggers_menu_refresh(self):
        # Сквозная проверка: обработчик события "ready" из шины (а не прямой
        # вызов _refresh_voice_menu в отрыве от остального) должен обновить
        # меню — это и есть реальный путь срабатывания после успешного
        # NeuroVoxEngine._prepare_and_start(). _handle_event вызывается
        # напрямую, а не через таймерный self.after(...)-цикл _poll_events —
        # тот предназначен для реального приложения, а не для теста.
        from workers import UiEvent
        self.win.engine.available_speakers = lambda: ["aidar", "baya", "random"]
        self.win._handle_event(UiEvent(kind="ready", text="Готово."))
        self.win.update()
        self.assertIn("random", self.win.voice_menu.cget("values"))


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class RecognitionPreviewButtonTests(unittest.TestCase):
    """
    Кнопка «Проверить распознавание»: вызывает engine.preview_recognition()
    (не preview_voice/TTS) и предварительно синхронизирует положение рамок.
    """

    def setUp(self):
        self.win = gui.MainWindow()
        self.win.update()

    def tearDown(self):
        try:
            for after_id in self.win.tk.eval("after info").split():
                self.win.after_cancel(after_id)
        except tkinter.TclError:
            pass
        try:
            for overlay in self.win.overlays:
                overlay.destroy()
            self.win.destroy()
        except tkinter.TclError:
            pass

    def test_button_exists_with_expected_label(self):
        self.assertIn("Проверить распознавание", self.win.recognition_test_button.cget("text"))

    def test_click_calls_preview_recognition_not_preview_voice(self):
        calls = []
        self.win.engine.preview_recognition = lambda: calls.append("recognition")
        self.win.engine.preview_voice = lambda: calls.append("voice")
        self.win._preview_recognition()
        self.assertEqual(calls, ["recognition"])

    def test_click_syncs_roi_positions_before_preview(self):
        synced = []
        self.win._sync_rois = lambda: synced.append(True)
        self.win.engine.preview_recognition = lambda: None
        self.win._preview_recognition()
        self.assertEqual(synced, [True])


if __name__ == "__main__":
    unittest.main()
