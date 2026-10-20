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
from workers import (  # noqa: E402
    CaptureWorker,
    EventBus,
    NeuroVoxEngine,
    OverlayCaptureGate,
    SharedSettings,
    _AreaHealthMonitor,
)

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


class OverlayCaptureGateTests(unittest.TestCase):
    """
    Прямые тесты моста OverlayCaptureGate — без настоящих окон и без
    потока снимков: только протокол install/request_hide/request_show/poll,
    каким его используют gui.MainWindow и workers.CaptureWorker._grab_area.
    """

    def test_request_hide_without_install_returns_false_immediately(self):
        # Ничего не зарегистрировано (GUI ещё не запустило слежение, либо
        # это тест/preview без окна вообще) — не должно ни блокировать
        # вызывающий поток, ни притворяться, что рамка спрятана.
        gate = OverlayCaptureGate()
        self.assertFalse(gate.request_hide())

    def test_hide_then_poll_confirms_and_calls_hide_fn(self):
        gate = OverlayCaptureGate()
        calls = []
        gate.install(lambda: calls.append("hide"), lambda: calls.append("show"))

        result = {}

        def worker():
            result["confirmed"] = gate.request_hide()

        thread = threading.Thread(target=worker)
        thread.start()
        # Имитируем короткий таймер GUI (gui._poll_gate): рано или поздно он
        # должен увидеть запрос и выполнить hide_fn.
        deadline = time.monotonic() + 2.0
        while "confirmed" not in result and time.monotonic() < deadline:
            gate.poll()
            time.sleep(0.005)
        thread.join(timeout=1.0)

        self.assertEqual(calls, ["hide"])
        self.assertTrue(result.get("confirmed"))

    def test_show_is_called_after_hide_on_next_poll(self):
        gate = OverlayCaptureGate()
        calls = []
        gate.install(lambda: calls.append("hide"), lambda: calls.append("show"))

        def worker():
            gate.request_hide()
            gate.request_show()

        thread = threading.Thread(target=worker)
        thread.start()
        deadline = time.monotonic() + 2.0
        while calls.count("show") == 0 and time.monotonic() < deadline:
            gate.poll()
            time.sleep(0.005)
        thread.join(timeout=1.0)

        self.assertEqual(calls, ["hide", "show"])

    def test_uninstall_releases_any_waiting_request(self):
        # Если GUI закрылось (или слежение остановили) ровно в момент, когда
        # поток снимков ждёт подтверждения hide — он не должен зависнуть
        # навсегда: uninstall() обязан немедленно освободить ожидание.
        gate = OverlayCaptureGate()
        gate.install(lambda: None, lambda: None)

        result = {}

        def worker():
            result["confirmed"] = gate.request_hide()

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.05)   # даём worker-потоку дойти до wait() внутри request_hide
        gate.uninstall()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive(), "поток не должен был зависнуть после uninstall()")

    def test_request_hide_after_uninstall_returns_false(self):
        gate = OverlayCaptureGate()
        gate.install(lambda: None, lambda: None)
        gate.uninstall()
        self.assertFalse(gate.request_hide())

    def test_request_show_without_install_is_a_no_op(self):
        gate = OverlayCaptureGate()
        gate.request_show()  # не должно поднять исключение

    def test_poll_without_pending_request_does_nothing(self):
        gate = OverlayCaptureGate()
        calls = []
        gate.install(lambda: calls.append("hide"), lambda: calls.append("show"))
        gate.poll()
        self.assertEqual(calls, [])


class CaptureWorkerOverlayGateIntegrationTests(unittest.TestCase):
    """
    CaptureWorker.run() — убеждаемся, что при установленном overlay_gate
    рамки прячутся и показываются РОВНО ОДИН РАЗ за весь проход по rois
    в рамках одного цикла снимков, а не отдельно на каждую область.

    До этого теста (и до соответствующего исправления в CaptureWorker)
    hide()/show() вызывались внутри _grab_area() — то есть попарно вокруг
    каждой области по отдельности. При нескольких настроенных областях
    (MAX_AREAS = 6) это означало, что рамки на экране прячутся и
    показываются N раз за один проход цикла снимков вместо одного — отсюда
    заметное мерцание рамок на частоте capture_fps (до 30 раз в секунду)
    и лишняя нагрузка на обмен с главным потоком GUI через
    OverlayCaptureGate. Текущая реализация переносит hide()/show() на
    уровень run(): одна пара запросов оборачивает весь проход по всем
    настроенным областям за цикл — это то, что здесь и проверяется.
    """

    @staticmethod
    def _run_one_cycle(worker, stop_event, timeout=2.0):
        """Пускает worker.run() в отдельном потоке и ждёт, пока он сам не
        остановится (stop_event выставляется подложным grab — см. ниже,
        после ровно одного прохода цикла while)."""
        thread = threading.Thread(target=worker.run, name="CaptureWorkerTest")
        thread.start()
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "CaptureWorker.run() не остановился вовремя"

    def test_hide_and_show_wrap_the_whole_cycle_once_for_multiple_areas(self):
        """Доказательство исправления: 3 области, но только одна пара hide/show."""
        gate = OverlayCaptureGate()
        order = []
        gate.install(lambda: order.append("hide"), lambda: order.append("show"))

        stop_event = threading.Event()
        grabbed_rois = []

        class _Capture:
            def grab(self, roi):
                order.append(f"grab:{roi}")
                grabbed_rois.append(roi)
                if len(grabbed_rois) == 3:
                    # Третий (последний для этого цикла) снимок — останавливаем
                    # worker сразу после текущей итерации while, чтобы run()
                    # вернулся, а не крутился бесконечно в тесте.
                    stop_event.set()
                return _frame_with_outlined_text()

            def close(self):
                pass

        settings = config.Settings()
        settings.rois = [[0, 0, 400, 80], [0, 100, 400, 80], [0, 200, 400, 80]]
        shared = SharedSettings(settings)
        worker = CaptureWorker(
            shared, EventBus(), buffer=_NullBuffer(), stop_event=stop_event,
            capture_factory=_Capture, overlay_gate=gate,
        )

        stop_polling = threading.Event()

        def poller():
            while not stop_polling.is_set():
                gate.poll()
                time.sleep(0.005)

        poll_thread = threading.Thread(target=poller)
        poll_thread.start()
        try:
            self._run_one_cycle(worker, stop_event)
            deadline = time.monotonic() + 2.0
            while order.count("show") < 1 and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            stop_polling.set()
            poll_thread.join(timeout=1.0)

        # Ровно один hide, три снимка между ним и show, ровно один show —
        # не три пары hide/show, как было бы при попарной схеме на область.
        self.assertEqual(order.count("hide"), 1, order)
        self.assertEqual(order.count("show"), 1, order)
        self.assertEqual(order.count("grab"), 0)  # сверяем точные grab:roi ниже
        self.assertEqual(len([e for e in order if e.startswith("grab:")]), 3, order)
        self.assertEqual(order[0], "hide", order)
        self.assertEqual(order[-1], "show", order)

    def test_without_gate_run_just_grabs_every_area(self):
        """Без overlay_gate (как раньше — preview_recognition, тесты) поведение
        снимков не меняется: ни одного обращения к несуществующему gate."""
        stop_event = threading.Event()
        grabbed = []

        class _Capture:
            def grab(self, roi):
                grabbed.append(roi)
                if len(grabbed) == 2:
                    stop_event.set()
                return _frame_with_outlined_text()

            def close(self):
                pass

        settings = config.Settings()
        settings.rois = [[0, 0, 400, 80], [0, 100, 400, 80]]
        shared = SharedSettings(settings)
        worker = CaptureWorker(
            shared, EventBus(), buffer=_NullBuffer(), stop_event=stop_event,
            capture_factory=_Capture, overlay_gate=None,
        )
        self._run_one_cycle(worker, stop_event)
        self.assertEqual(len(grabbed), 2)

    def test_show_is_still_called_even_if_one_area_raises(self):
        """Рамка не должна остаться спрятанной насовсем, если снимок одной
        из областей бросил исключение — ошибка в середине прохода не должна
        пропустить request_show() в конце цикла."""
        gate = OverlayCaptureGate()
        order = []
        gate.install(lambda: order.append("hide"), lambda: order.append("show"))

        stop_event = threading.Event()
        calls = []

        class _Capture:
            def grab(self, roi):
                calls.append(roi)
                if len(calls) == 1:
                    raise RuntimeError("симуляция сбоя снимка одной области")
                if len(calls) == 2:
                    stop_event.set()
                return _frame_with_outlined_text()

            def close(self):
                pass

        settings = config.Settings()
        settings.rois = [[0, 0, 400, 80], [0, 100, 400, 80]]
        shared = SharedSettings(settings)
        worker = CaptureWorker(
            shared, EventBus(), buffer=_NullBuffer(), stop_event=stop_event,
            capture_factory=_Capture, overlay_gate=gate,
        )

        stop_polling = threading.Event()

        def poller():
            while not stop_polling.is_set():
                gate.poll()
                time.sleep(0.005)

        poll_thread = threading.Thread(target=poller)
        poll_thread.start()
        try:
            self._run_one_cycle(worker, stop_event)
            deadline = time.monotonic() + 2.0
            while "show" not in order and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            stop_polling.set()
            poll_thread.join(timeout=1.0)

        self.assertIn("show", order)
        self.assertEqual(len(calls), 2)

    def test_no_areas_configured_does_not_touch_the_gate(self):
        """rois пуст (пользователь ещё ничего не настроил) — gate не должен
        получать ни одного запроса, цикл просто ничего не снимает."""
        gate = OverlayCaptureGate()
        order = []
        gate.install(lambda: order.append("hide"), lambda: order.append("show"))

        stop_event = threading.Event()
        cycles = {"count": 0}

        class _Capture:
            def grab(self, roi):  # pragma: no cover — не должен вызываться
                raise AssertionError("grab не должен вызываться без областей")

            def close(self):
                pass

        settings = config.Settings()
        settings.rois = []
        shared = SharedSettings(settings)
        worker = CaptureWorker(
            shared, EventBus(), buffer=_NullBuffer(), stop_event=stop_event,
            capture_factory=_Capture, overlay_gate=gate,
        )

        # Без снимков нечему выставить stop_event изнутри цикла — останавливаем
        # снаружи, дав ровно одной итерации while отработать.
        def stopper():
            time.sleep(0.2)
            stop_event.set()

        threading.Thread(target=stopper).start()
        self._run_one_cycle(worker, stop_event, timeout=3.0)
        self.assertEqual(order, [])


class _NullBuffer:
    """Подделка FrameBuffer для тестов CaptureWorker.run(), которым сам
    буфер не важен — реальный FrameBuffer требует config для лимитов, а
    здесь интересен только порядок вызовов gate и capture."""

    def put(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    unittest.main()
