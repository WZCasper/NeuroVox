# -*- coding: utf-8 -*-
"""
Тесты диагностики «рамка не наведена на текст» (_AreaHealthMonitor) и её
сквозной интеграции в OcrWorker — через реальные потоки на подделках
(pipeline_fakes.run_pipeline), без экрана, моделей и звука.
"""

import pathlib
import sys
import threading
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from workers import EventBus, NeuroVoxEngine, SharedSettings, _AreaHealthMonitor  # noqa: E402

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


def _wait_for_thread(name, timeout=5.0):
    """Дожидается завершения фонового потока с данным именем (или что он вообще не появлялся)."""
    deadline = time.monotonic() + timeout
    # Сначала даём потоку шанс стартовать — NeuroVoxEngine.preview_recognition()
    # запускает его асинхронно, сразу после вызова его может ещё не быть в списке.
    thread = None
    while time.monotonic() < deadline:
        thread = next((t for t in threading.enumerate() if t.name == name), None)
        if thread is not None:
            break
        time.sleep(0.01)
    if thread is not None:
        thread.join(timeout=max(0.1, deadline - time.monotonic()))


class _FakeOcr:
    """
    Поддельный OCR-движок: тексты заданы по порядку вызовов recognize() —
    preprocess_for_ocr бинаризует кадр, исходные пиксели-метки в нём не
    сохраняются, поэтому различать кадры по содержимому ненадёжно; порядок
    вызовов же детерминирован (preview_recognition идёт по areas последовательно).
    """

    name = "fake"

    def __init__(self, texts_in_order=(), raise_on_call=frozenset()):
        self._texts = list(texts_in_order)
        self._raise_on_call = raise_on_call
        self.calls = 0

    def load(self):
        pass

    def recognize(self, image):
        call_index = self.calls
        self.calls += 1
        if call_index in self._raise_on_call:
            from ocr_engine import OcrError
            raise OcrError(f"подделанная ошибка распознавания (вызов {call_index})")
        return self._texts[call_index] if call_index < len(self._texts) else ""


class _FakeCapture:
    """Поддельный источник кадров: по каждому вызову grab(roi) отдаёт следующий кадр по порядку."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.closed = False
        self.grabbed_rois = []

    def __call__(self):
        return self

    def grab(self, roi):
        self.grabbed_rois.append(roi)
        index = len(self.grabbed_rois) - 1
        return self._frames[index]

    def close(self):
        self.closed = True


def _frame_with_outlined_text(size=(80, 400)):
    """
    Синтетический кадр, который preprocess_for_ocr гарантированно распознает
    как «похоже на текст» — обведённые белые буквы на сером фоне с лёгким
    градиентом (тот же стиль, что и в test_capture.py).
    """
    import cv2
    import numpy as np

    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        value = int(40 + (x / w) * 120)
        frame[:, x] = (value, value, value)
    cv2.putText(frame, "TEXT", (20, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, "TEXT", (20, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _blank_frame(size=(80, 400)):
    """Совершенно пустой (однотонный) кадр — preprocess_for_ocr должен вернуть None."""
    import numpy as np
    return np.full((size[0], size[1], 3), 60, dtype=np.uint8)


class _FakeSileroTts:
    """Минимальная подделка SileroTts для теста NeuroVoxEngine.available_speakers()."""

    def __init__(self, speakers):
        self._speakers = list(speakers)
        self.model_name = "fake_model"
        self.is_loaded = True

    def available_speakers(self):
        return list(self._speakers)


class NeuroVoxEngineSpeakersTests(unittest.TestCase):
    """
    available_speakers() должен безопасно работать из потока интерфейса: до
    загрузки модели — отдавать статичный список-заглушку, после загрузки —
    список реальных голосов этой конкретной модели (см. tts_engine.SileroTts).
    """

    def _make_engine(self):
        shared = SharedSettings(config.Settings())
        return NeuroVoxEngine(shared, EventBus())

    def test_before_any_model_loaded_returns_config_fallback(self):
        engine = self._make_engine()
        self.assertEqual(set(engine.available_speakers()), set(config.SILERO_SPEAKERS))

    def test_after_model_loaded_returns_models_own_speakers(self):
        engine = self._make_engine()
        # Подставляем уже «загруженную» модель напрямую — как и в тестах
        # SileroTts.synthesize(), без реального torch и реальной загрузки.
        engine._tts = _FakeSileroTts(["aidar", "baya", "kseniya", "xenia", "eugene", "random"])
        self.assertIn("random", engine.available_speakers())
        self.assertEqual(
            set(engine.available_speakers()),
            {"aidar", "baya", "kseniya", "xenia", "eugene", "random"},
        )

    def test_does_not_block_or_touch_threads(self):
        # Чисто функциональная проверка «безопасно вызывать из потока интерфейса»:
        # метод не создаёт и не трогает никакие потоки сам по себе.
        engine = self._make_engine()
        before = threading.active_count()
        engine.available_speakers()
        self.assertEqual(threading.active_count(), before)


class PreviewRecognitionTests(unittest.TestCase):
    """
    NeuroVoxEngine.preview_recognition(): разовая проверка распознавания без
    обращения к TTS/AudioPlayer — ровно то, что требует кнопка «Проверить
    распознавание» в интерфейсе.
    """

    def _make_engine(self, rois, ocr, frames):
        settings = config.Settings()
        settings.rois = rois
        settings.ocr_engine = config.DEFAULT_OCR_ENGINE
        shared = SharedSettings(settings)
        engine = NeuroVoxEngine(shared, EventBus(), capture_factory=_FakeCapture(frames))
        # «Уже загруженный» движок с совпадающим именем — _prepare_ocr не
        # станет создавать настоящий (а настоящий требует easyocr/tesseract).
        engine._ocr = ocr
        engine._ocr_name = config.DEFAULT_OCR_ENGINE
        return engine

    def test_recognized_text_is_logged_raw_and_cleaned_without_tts(self):
        ocr = _FakeOcr(texts_in_order=["Привет,   мир!\n"])
        engine = self._make_engine(
            rois=[[0, 0, 400, 80]], ocr=ocr, frames=[_frame_with_outlined_text()],
        )
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertTrue(any("сырой текст" in t and "Привет" in t for t in texts), texts)
        self.assertTrue(any("очищенный текст" in t and "Привет, мир!" in t for t in texts), texts)
        # Ключевое требование: TTS и плеер не должны были использоваться вообще.
        self.assertIsNone(engine._tts)

    def test_empty_area_reports_no_text_found(self):
        # preprocess_for_ocr возвращает None для однотонного кадра — пользователь
        # должен увидеть явное объяснение, а не тишину в журнале.
        ocr = _FakeOcr()
        engine = self._make_engine(rois=[[0, 0, 400, 80]], ocr=ocr, frames=[_blank_frame()])
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertTrue(any("текст не обнаружен" in t for t in texts), texts)
        self.assertEqual(ocr.calls, 0)  # OCR вообще не вызывался — нечего распознавать

    def test_multiple_areas_are_each_labelled(self):
        ocr = _FakeOcr(texts_in_order=["Первая", "Вторая"])
        engine = self._make_engine(
            rois=[[0, 0, 400, 80], [0, 100, 400, 80]],
            ocr=ocr,
            frames=[_frame_with_outlined_text(), _frame_with_outlined_text()],
        )
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertTrue(any(t.startswith("Область 1:") for t in texts), texts)
        self.assertTrue(any(t.startswith("Область 2:") for t in texts), texts)

    def test_single_area_is_not_labelled_with_number(self):
        # Одна область — подписывать «Область 1:» избыточно, как и у остальной
        # диагностики в проекте (ср. recognized-метки в OcrWorker.run()).
        ocr = _FakeOcr(texts_in_order=["Текст"])
        engine = self._make_engine(
            rois=[[0, 0, 400, 80]], ocr=ocr, frames=[_frame_with_outlined_text()],
        )
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertFalse(any("Область 1" in t for t in texts), texts)

    def test_ocr_error_on_one_area_does_not_stop_the_others(self):
        ocr = _FakeOcr(texts_in_order=["", "Вторая область в порядке"], raise_on_call={0})
        engine = self._make_engine(
            rois=[[0, 0, 400, 80], [0, 100, 400, 80]],
            ocr=ocr,
            frames=[_frame_with_outlined_text(), _frame_with_outlined_text()],
        )
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertTrue(any(e.kind == "error" and "Область 1" in e.text for e in events), texts)
        self.assertTrue(any("Вторая область в порядке" in t for t in texts), texts)

    def test_capture_is_closed_after_use(self):
        ocr = _FakeOcr(texts_in_order=["Текст"])
        capture = _FakeCapture([_frame_with_outlined_text()])
        settings = config.Settings()
        settings.rois = [[0, 0, 400, 80]]
        settings.ocr_engine = config.DEFAULT_OCR_ENGINE
        shared = SharedSettings(settings)
        engine = NeuroVoxEngine(shared, EventBus(), capture_factory=capture)
        engine._ocr = ocr
        engine._ocr_name = config.DEFAULT_OCR_ENGINE

        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")
        self.assertTrue(capture.closed)

    def test_lazily_loads_ocr_engine_when_not_loaded_yet(self):
        # Движок ещё ни разу не загружался (engine._ocr is None) — тот же
        # ленивый путь, что и у preview_voice() для TTS: _prepare_ocr должен
        # быть вызван и загрузить движок перед распознаванием, а не упасть.
        from unittest import mock

        ocr = _FakeOcr(texts_in_order=["Загружено и распознано"])
        settings = config.Settings()
        settings.rois = [[0, 0, 400, 80]]
        settings.ocr_engine = config.DEFAULT_OCR_ENGINE
        shared = SharedSettings(settings)
        capture = _FakeCapture([_frame_with_outlined_text()])
        engine = NeuroVoxEngine(shared, EventBus(), capture_factory=capture)
        self.assertIsNone(engine._ocr)  # движок действительно ещё не загружен

        with mock.patch("workers.create_with_fallback", return_value=ocr) as fake_create:
            engine.preview_recognition()
            _wait_for_thread("RecognitionPreview")

        fake_create.assert_called_once()
        events = engine._bus.poll(limit=50)
        texts = [e.text for e in events]
        self.assertTrue(any("Загружено и распознано" in t for t in texts), texts)

    def test_no_areas_configured_reports_error_without_touching_ocr(self):
        ocr = _FakeOcr()
        engine = self._make_engine(rois=[], ocr=ocr, frames=[])
        engine.preview_recognition()
        _wait_for_thread("RecognitionPreview")

        events = engine._bus.poll(limit=50)
        self.assertTrue(any(e.kind == "error" for e in events))
        self.assertEqual(ocr.calls, 0)


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
