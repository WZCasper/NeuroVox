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
from unittest import mock

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


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class StartupErrorAndHeaderTests(unittest.TestCase):
    """
    Ошибка запуска: причина видна пользователю (диалог и журнал), статус не сменяется
    на «Остановлено». Подзаголовок шапки не сжимается сеткой и не обрезается.
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

    def test_startup_error_shows_dialog_and_keeps_error_status(self):
        from workers import UiEvent
        self.win._starting = True
        with mock.patch.object(gui.messagebox, "showerror") as dialog:
            self.win._handle_event(UiEvent(kind="error", text="Не удалось скачать модель озвучки"))
            # Движок сразу после ошибки шлёт «stopped» — статус ошибки не должен быть перезаписан.
            self.win._handle_event(UiEvent(kind="stopped", text="Остановлено."))
        self.win.update()
        dialog.assert_called_once()
        self.assertEqual(self.win.status_label.cget("text"), "Ошибка запуска")
        self.assertEqual(self.win.start_button.cget("text"), "▶  Запустить")
        self.assertFalse(self.win._starting)

    def test_runtime_error_does_not_show_dialog(self):
        from workers import UiEvent
        self.win._starting = False
        with mock.patch.object(gui.messagebox, "showerror") as dialog:
            self.win._handle_event(UiEvent(kind="error", text="Ошибка воспроизведения звука"))
        dialog.assert_not_called()
        self.assertEqual(self.win.status_label.cget("text"), "Ошибка")

    def test_header_subtitle_is_not_squeezed_at_minimum_width(self):
        self.win.geometry("580x780")
        self.win.update()
        label = self.win.subtitle_label
        self.assertLessEqual(label.winfo_reqwidth(), label.winfo_width())


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class SpeakingOverlayEventsTests(unittest.TestCase):
    """
    "speaking_start"/"speaking_end" из шины событий (workers.TtsPlaybackWorker)
    должны прятать и возвращать рамку ИМЕННО той области, что сейчас звучит —
    не все рамки сразу, и не рамку другой области. "skipped" должен попадать в
    журнал. Сами значения прозрачности — забота overlay.SelectionOverlay.set_reading
    (проверено отдельно); здесь проверяется только маршрутизация события к нужной
    рамке, поэтому set_reading подменяется, а не проверяется по факту отрисовки.
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

    def test_speaking_start_calls_set_reading_true_on_the_matching_overlay(self):
        from workers import UiEvent
        calls = []
        self.win.overlays[0].set_reading = lambda reading: calls.append(reading)
        self.win._handle_event(UiEvent(kind="speaking_start", value=0.0))
        self.assertEqual(calls, [True])

    def test_speaking_end_calls_set_reading_false_on_the_matching_overlay(self):
        from workers import UiEvent
        calls = []
        self.win.overlays[0].set_reading = lambda reading: calls.append(reading)
        self.win._handle_event(UiEvent(kind="speaking_end", value=0.0))
        self.assertEqual(calls, [False])

    def test_out_of_range_area_index_does_not_crash(self):
        from workers import UiEvent
        # По умолчанию существует только одна область (индекс 0) — индекс 5
        # не должен ничего ломать, событие просто тихо игнорируется.
        self.win._handle_event(UiEvent(kind="speaking_start", value=5.0))
        self.win._handle_event(UiEvent(kind="speaking_end", value=5.0))

    def test_skipped_event_is_logged(self):
        from workers import UiEvent
        self.win._handle_event(
            UiEvent(kind="skipped", text="пропущено устаревших реплик: 2 — тест")
        )
        self.win.update()
        content = self.win.log_box.get("1.0", "end")
        self.assertIn("⏭", content)
        self.assertIn("пропущено устаревших реплик: 2 — тест", content)

    def test_stopped_event_resets_reading_state_on_every_overlay(self):
        # Защита на случай, если слежение остановили прямо во время звучания
        # реплики — ни одна рамка не должна остаться невидимой навсегда.
        from workers import UiEvent
        calls = []
        self.win.overlays[0].set_reading = lambda reading: calls.append(reading)
        self.win._handle_event(UiEvent(kind="stopped", text="Остановлено."))
        self.assertIn(False, calls)

    def test_multiple_areas_route_to_the_right_overlay_only(self):
        from workers import UiEvent
        self.win._add_area()
        self.win.update()
        self.assertEqual(len(self.win.overlays), 2)
        calls0, calls1 = [], []
        self.win.overlays[0].set_reading = lambda reading: calls0.append(reading)
        self.win.overlays[1].set_reading = lambda reading: calls1.append(reading)
        self.win._handle_event(UiEvent(kind="speaking_start", value=1.0))
        self.assertEqual(calls0, [])
        self.assertEqual(calls1, [True])
