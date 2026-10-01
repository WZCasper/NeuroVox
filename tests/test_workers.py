# -*- coding: utf-8 -*-
"""
Тесты диагностики «рамка не наведена на текст» (_AreaHealthMonitor) и её
сквозной интеграции в OcrWorker — через реальные потоки на подделках
(pipeline_fakes.run_pipeline), без экрана, моделей и звука.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from workers import _AreaHealthMonitor  # noqa: E402

from tests.pipeline_fakes import run_pipeline  # noqa: E402


class AreaHealthMonitorTests(unittest.TestCase):
    """Прямые тесты логики монитора — без потоков, без реального времени."""

    def test_no_warning_before_threshold(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        self.assertIsNone(monitor.check(0, now=9.9))

    def test_warning_after_threshold(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        warning = monitor.check(0, now=10.0)
        self.assertIsNotNone(warning)
        self.assertIn("Область 1", warning)
        self.assertIn("10 секунд", warning)
        self.assertIn("рамка наведена на игру", warning)

    def test_warning_is_not_repeated_until_new_phrase(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        first = monitor.check(0, now=10.0)
        second = monitor.check(0, now=20.0)  # ситуация не изменилась — молчим
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_phrase_resets_timer_and_allows_future_warning(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        self.assertIsNotNone(monitor.check(0, now=10.0))
        # Фраза распознана — «всё в порядке», таймер и флаг предупреждения сбрасываются.
        monitor.on_phrase_recognized(0, now=15.0)
        self.assertIsNone(monitor.check(0, now=20.0))  # прошло всего 5 секунд с фразы
        self.assertIsNotNone(monitor.check(0, now=25.0))  # а тут уже 10 — можно предупредить снова

    def test_areas_are_tracked_independently(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        monitor.reset(1, now=0.0)
        monitor.on_phrase_recognized(0, now=5.0)  # только область 0 «живая»
        self.assertIsNone(monitor.check(0, now=14.0))       # 9 секунд с последней фразы
        self.assertIsNotNone(monitor.check(1, now=14.0))    # а тут все 14 секунд молчания

    def test_check_without_prior_reset_does_not_warn_immediately(self):
        # Область появилась только что (первый вызов check без явного reset) —
        # не должно быть немедленного ложного предупреждения на пустом месте.
        monitor = _AreaHealthMonitor(warning_after=10.0)
        self.assertIsNone(monitor.check(0, now=1000.0))
        self.assertIsNone(monitor.check(0, now=1005.0))

    def test_forget_missing_clears_removed_areas(self):
        monitor = _AreaHealthMonitor(warning_after=10.0)
        monitor.reset(0, now=0.0)
        monitor.reset(1, now=0.0)
        monitor.forget_missing(range(1))  # оставили только область 0
        # Область 1 «забыта»: следующий check начнёт отсчёт заново, а не выдаст
        # немедленное предупреждение по старому (уже нерелевантному) времени.
        self.assertIsNone(monitor.check(1, now=100.0))


class OcrWorkerHealthIntegrationTests(unittest.TestCase):
    """
    Сквозные тесты через настоящие потоки CaptureWorker/OcrWorker на подделках
    экрана и OCR — проверяют, что диагностика реально работает внутри
    OcrWorker.run(), а не только в изолированном _AreaHealthMonitor.
    """

    def test_area_with_unrecognizable_text_triggers_warning(self):
        # Область «видна» (там регулярно что-то нарисовано), но FakeOcr никогда
        # не узнаёт этот текст (его нет в списке texts) — рамка как будто
        # смотрит на текст, которого OCR не понимает (типичный «мимо цели» кейс).

        def timeline(area, t):
            return "нераспознаваемый текст" if int(t * 2) % 2 == 0 else None

        _, _, _, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=["Привет"],
            duration=1.2, ocr_latency=0.02, area_pause=0.2, capture_fps=20.0,
            no_speech_warning_seconds=0.5,
        )
        events = bus.poll(limit=500)
        warnings = [e.text for e in events if e.kind == "error" and "не обнаруживаются" in e.text]
        self.assertTrue(warnings, "ожидалось предупреждение о ненаведённой рамке")
        self.assertIn("Область 1", warnings[0])

    def test_area_with_no_frames_at_all_triggers_warning(self):
        # Область вообще ничего не показывает (timeline всегда None) — рамка
        # как будто указывает в полностью статичное место без текста.

        def timeline(area, t):
            return None

        _, _, _, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=["Привет"],
            duration=1.2, ocr_latency=0.02, area_pause=0.2, capture_fps=20.0,
            no_speech_warning_seconds=0.5,
        )
        events = bus.poll(limit=500)
        warnings = [e.text for e in events if e.kind == "error" and "не обнаруживаются" in e.text]
        self.assertTrue(warnings, "ожидалось предупреждение даже при полном отсутствии кадров")

    def test_area_with_recognized_speech_does_not_warn(self):
        # Область исправно выдаёт распознаваемые реплики — предупреждения быть
        # не должно (это и есть штатная, правильно настроенная рамка). Текст
        # держится на экране заметно дольше CAPTURE_SETTLE_SECONDS (0.3 c), а
        # пауза между показами — заметно дольше CAPTURE_EMPTY_HOLD_SECONDS
        # (0.8 c): иначе область не успеет «опустеть» между репликами, фильтр
        # повторов не сбросится, и одна и та же фраза повторно не распознается
        # — это ошибка сценария теста, а не диагностики.
        def timeline(area, t):
            cycle = t % 4.0
            return "Привет" if cycle < 1.0 else None

        _, _, _, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=["Привет"],
            duration=6.0, ocr_latency=0.02, area_pause=0.2, capture_fps=20.0,
            no_speech_warning_seconds=6.0,  # заметный запас над периодом повторения (4 c)
        )
        events = bus.poll(limit=500)
        warnings = [e.text for e in events if e.kind == "error" and "не обнаруживаются" in e.text]
        self.assertFalse(warnings, f"не ожидалось предупреждений, но получено: {warnings}")

    def test_short_silence_below_threshold_does_not_warn(self):
        # Реплика прозвучала (и успела «устояться» дольше CAPTURE_SETTLE_SECONDS),
        # затем пауза короче порога предупреждения — это обычная короткая тишина
        # в диалоге, а не признак неправильной рамки.
        def timeline(area, t):
            return "Привет" if t < 0.8 else None

        _, _, _, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=["Привет"],
            duration=1.2, ocr_latency=0.02, area_pause=0.2, capture_fps=20.0,
            no_speech_warning_seconds=5.0,  # заметно дольше всей длительности теста
        )
        events = bus.poll(limit=500)
        warnings = [e.text for e in events if e.kind == "error" and "не обнаруживаются" in e.text]
        self.assertFalse(warnings, f"не ожидалось предупреждений, но получено: {warnings}")


if __name__ == "__main__":
    unittest.main()
