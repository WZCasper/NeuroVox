# -*- coding: utf-8 -*-
"""
Рабочие потоки NeuroVox.

Архитектура (главный поток занят только интерфейсом):

    [Поток захвата и OCR] --текст--> [Фильтр] --фраза--> [Очередь] --> [Поток озвучки и воспроизведения]

Потоки общаются через thread-safe очереди и события. Любые сообщения для
интерфейса отправляются в очередь событий, а GUI сам забирает их в своём
потоке — так окно никогда не зависает и не вызывается из чужих потоков.
"""

import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from capture import CaptureError, ScreenCapture, frames_are_similar, preprocess_for_ocr
from ocr_engine import BaseOcr, OcrError, create_with_fallback
from text_filter import SubtitleFilter
from tts_engine import SileroTts, TtsError

logger = logging.getLogger("neurovox.workers")


# ---------------------------------------------------------------------------
# События для интерфейса
# ---------------------------------------------------------------------------

@dataclass
class UiEvent:
    """Сообщение из рабочего потока в интерфейс."""

    kind: str          # "status" | "recognized" | "spoken" | "error" | "progress" | "ready" | "stopped"
    text: str = ""
    value: float = 0.0


class EventBus:
    """Потокобезопасная очередь событий для GUI."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[UiEvent]" = queue.Queue(maxsize=500)

    def emit(self, kind: str, text: str = "", value: float = 0.0) -> None:
        event = UiEvent(kind, text, value)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Интерфейс не успевает — отбрасываем самое старое сообщение.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass

    def poll(self, limit: int = 50):
        """Забирает накопившиеся события (вызывается из потока GUI)."""
        events = []
        for _ in range(limit):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events


# ---------------------------------------------------------------------------
# Воспроизведение звука
# ---------------------------------------------------------------------------

class AudioPlayer:
    """Воспроизводит звук через sounddevice и умеет мгновенно прерываться."""

    def __init__(self, sample_rate: int = config.TTS_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._sd = None
        self._stop_flag = threading.Event()

    def _ensure_backend(self):
        if self._sd is None:
            try:
                import sounddevice as sd  # noqa: WPS433
            except (ImportError, OSError) as exc:
                raise TtsError(
                    "Не удалось подключить звуковую библиотеку sounddevice. "
                    f"Выполните: pip install sounddevice. Подробности: {exc}"
                ) from exc
            self._sd = sd
        return self._sd

    def check_device(self) -> None:
        """Проверяет, что в системе есть устройство вывода звука."""
        sd = self._ensure_backend()
        try:
            sd.query_devices(kind="output")
        except Exception as exc:  # noqa: BLE001
            raise TtsError(
                "Не найдено устройство вывода звука. Подключите колонки или наушники. "
                f"Подробности: {exc}"
            ) from exc

    def play(self, samples: np.ndarray, volume: float = 1.0) -> None:
        """Проигрывает звук и блокирует поток до конца (или до вызова stop())."""
        sd = self._ensure_backend()
        if samples.size == 0:
            return
        self._stop_flag.clear()
        data = np.clip(samples * float(volume), -1.0, 1.0).astype(np.float32)
        try:
            sd.play(data, samplerate=self.sample_rate, blocking=False)
            duration = data.size / float(self.sample_rate)
            deadline = time.monotonic() + duration + 1.0
            while time.monotonic() < deadline:
                if self._stop_flag.wait(timeout=0.03):
                    sd.stop()
                    return
                # sd.get_stream() бывает None после завершения воспроизведения.
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    return
        except Exception as exc:  # noqa: BLE001
            raise TtsError(f"Ошибка воспроизведения звука: {exc}") from exc

    def stop(self) -> None:
        """Мгновенно прерывает текущее воспроизведение."""
        self._stop_flag.set()
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Общий доступ к настройкам между GUI и потоками
# ---------------------------------------------------------------------------

class SharedSettings:
    """Потокобезопасный контейнер настроек, которые можно менять «на лету»."""

    def __init__(self, settings: config.Settings) -> None:
        self._lock = threading.Lock()
        self._settings = settings

    def snapshot(self) -> config.Settings:
        """Возвращает независимую копию настроек (включая вложенный список roi)."""
        with self._lock:
            return copy.deepcopy(self._settings)

    def update(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)


# ---------------------------------------------------------------------------
# Поток захвата и распознавания
# ---------------------------------------------------------------------------

class CaptureOcrWorker(threading.Thread):
    """Циклически захватывает область экрана, распознаёт текст и фильтрует его."""

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        speech_queue: "queue.Queue[str]",
        stop_event: threading.Event,
        ocr: BaseOcr,
    ) -> None:
        super().__init__(name="CaptureOcrWorker", daemon=True)
        self._shared = shared
        self._bus = bus
        self._speech_queue = speech_queue
        self._stop_event = stop_event
        self._ocr = ocr

    def run(self) -> None:
        # mss нужно создавать именно в том потоке, где он используется.
        try:
            capture = ScreenCapture()
        except CaptureError as exc:
            self._bus.emit("error", str(exc))
            return

        settings = self._shared.snapshot()
        subtitle_filter = SubtitleFilter(threshold=settings.similarity_threshold)
        previous_image: Optional[np.ndarray] = None
        last_raw_text = ""
        consecutive_errors = 0

        self._bus.emit("status", "Слежение за субтитрами запущено.")
        logger.info("Поток захвата и OCR запущен.")

        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                settings = self._shared.snapshot()
                subtitle_filter.threshold = settings.similarity_threshold

                try:
                    frame = capture.grab(tuple(settings.roi))
                    image = preprocess_for_ocr(frame)

                    if image is None:
                        raw_text = ""
                    elif frames_are_similar(image, previous_image):
                        # Картинка не изменилась — повторный OCR не нужен.
                        raw_text = last_raw_text
                    else:
                        raw_text = self._ocr.recognize(image)
                    previous_image = image
                    last_raw_text = raw_text
                    consecutive_errors = 0
                except CaptureError as exc:
                    consecutive_errors += 1
                    self._report_repeated_error(consecutive_errors, str(exc))
                    self._sleep_until(cycle_start, 1.0)
                    continue
                except OcrError as exc:
                    consecutive_errors += 1
                    self._report_repeated_error(consecutive_errors, str(exc))
                    self._sleep_until(cycle_start, 1.0)
                    continue
                except Exception as exc:  # noqa: BLE001 — поток не должен падать из-за одного кадра
                    consecutive_errors += 1
                    logger.exception("Непредвиденная ошибка в цикле захвата")
                    self._report_repeated_error(consecutive_errors, f"Непредвиденная ошибка: {exc}")
                    self._sleep_until(cycle_start, 1.0)
                    continue

                phrase = subtitle_filter.process(raw_text)
                if phrase:
                    self._bus.emit("recognized", phrase)
                    self._enqueue_phrase(phrase)

                self._sleep_until(cycle_start, 1.0 / max(settings.capture_fps, 0.5))
        finally:
            capture.close()
            logger.info("Поток захвата и OCR остановлен.")

    # -- вспомогательные методы -------------------------------------------------

    def _enqueue_phrase(self, phrase: str) -> None:
        """Кладёт фразу в очередь; при переполнении выбрасывает самую старую."""
        while True:
            try:
                self._speech_queue.put_nowait(phrase)
                return
            except queue.Full:
                try:
                    dropped = self._speech_queue.get_nowait()
                    logger.info("Очередь озвучки переполнена, пропущена фраза: %s", dropped)
                except queue.Empty:
                    pass

    def _report_repeated_error(self, count: int, message: str) -> None:
        """Сообщает об ошибке в интерфейс, но не чаще чем раз в несколько попыток."""
        if count == 1 or count % 10 == 0:
            self._bus.emit("error", message)
        if count >= 30:
            self._bus.emit("error", "Слишком много ошибок подряд — слежение остановлено.")
            self._stop_event.set()

    def _sleep_until(self, cycle_start: float, period: float) -> None:
        remaining = period - (time.perf_counter() - cycle_start)
        if remaining > 0:
            self._stop_event.wait(remaining)


# ---------------------------------------------------------------------------
# Поток озвучки
# ---------------------------------------------------------------------------

class TtsPlaybackWorker(threading.Thread):
    """Берёт фразы из очереди, синтезирует речь и сразу воспроизводит."""

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        speech_queue: "queue.Queue[str]",
        stop_event: threading.Event,
        tts: SileroTts,
        player: AudioPlayer,
    ) -> None:
        super().__init__(name="TtsPlaybackWorker", daemon=True)
        self._shared = shared
        self._bus = bus
        self._speech_queue = speech_queue
        self._stop_event = stop_event
        self._tts = tts
        self._player = player

    def run(self) -> None:
        logger.info("Поток озвучки запущен.")
        try:
            while not self._stop_event.is_set():
                try:
                    phrase = self._speech_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Если за время синтеза накопились новые реплики, старые уже неактуальны.
                phrase = self._skip_to_latest(phrase)
                self._speak(phrase)
        finally:
            self._player.stop()
            logger.info("Поток озвучки остановлен.")

    def _skip_to_latest(self, phrase: str) -> str:
        """Пропускает устаревшие фразы, чтобы озвучка не отставала от экрана."""
        skipped = 0
        while self._speech_queue.qsize() > config.MAX_TTS_QUEUE - 1:
            try:
                phrase = self._speech_queue.get_nowait()
                skipped += 1
            except queue.Empty:
                break
        if skipped:
            logger.info("Пропущено устаревших реплик: %d", skipped)
        return phrase

    def _speak(self, phrase: str) -> None:
        settings = self._shared.snapshot()
        try:
            started = time.perf_counter()
            fragments = self._tts.synthesize(phrase, settings.speaker, settings.speed)
            if not fragments:
                return
            self._bus.emit("spoken", phrase, time.perf_counter() - started)
            for fragment in fragments:
                if self._stop_event.is_set():
                    return
                self._player.play(fragment, settings.volume)
        except TtsError as exc:
            logger.error("Ошибка озвучки: %s", exc)
            self._bus.emit("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка озвучки")
            self._bus.emit("error", f"Непредвиденная ошибка озвучки: {exc}")


# ---------------------------------------------------------------------------
# Управляющий класс
# ---------------------------------------------------------------------------

class NeuroVoxEngine:
    """
    Управляет жизненным циклом всех потоков: подготовка моделей, запуск, остановка.

    Тяжёлая подготовка (скачивание и загрузка моделей) идёт в отдельном
    потоке, поэтому окно остаётся отзывчивым.
    """

    def __init__(self, shared: SharedSettings, bus: EventBus) -> None:
        self._shared = shared
        self._bus = bus
        self._stop_event = threading.Event()
        self._speech_queue: "queue.Queue[str]" = queue.Queue(maxsize=config.MAX_TTS_QUEUE)
        self._threads = []
        self._starter: Optional[threading.Thread] = None
        self._ocr: Optional[BaseOcr] = None
        self._ocr_name: str = ""
        self._tts: Optional[SileroTts] = None
        self._player = AudioPlayer()
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # -- запуск --------------------------------------------------------------

    def start(self) -> None:
        """Запускает подготовку моделей и рабочие потоки (не блокирует GUI)."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()

        self._starter = threading.Thread(target=self._prepare_and_start, name="EngineStarter", daemon=True)
        self._starter.start()

    def _prepare_and_start(self) -> None:
        settings = self._shared.snapshot()
        try:
            self._bus.emit("status", "Проверка звукового устройства...")
            self._player.check_device()

            self._prepare_tts(settings)
            if self._stop_event.is_set():
                return
            self._prepare_ocr(settings)
            if self._stop_event.is_set():
                return
        except (TtsError, OcrError) as exc:
            logger.error("Не удалось запустить: %s", exc)
            self._bus.emit("error", str(exc))
            self._finish_stop()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка при запуске")
            self._bus.emit("error", f"Непредвиденная ошибка при запуске: {exc}")
            self._finish_stop()
            return

        self._speech_queue = queue.Queue(maxsize=config.MAX_TTS_QUEUE)
        capture_thread = CaptureOcrWorker(
            self._shared, self._bus, self._speech_queue, self._stop_event, self._ocr
        )
        tts_thread = TtsPlaybackWorker(
            self._shared, self._bus, self._speech_queue, self._stop_event, self._tts, self._player
        )
        self._threads = [capture_thread, tts_thread]
        for thread in self._threads:
            thread.start()
        self._bus.emit("ready", "Готово. Программа следит за субтитрами.")

    def _prepare_tts(self, settings: config.Settings) -> None:
        # Модель перезагружаем, только если пользователь выбрал другую.
        if self._tts is None or self._tts.model_name != settings.tts_model:
            self._tts = SileroTts(settings.tts_model)
        if not self._tts.is_loaded:
            self._bus.emit("status", "Загрузка модели озвучки...")
            self._tts.load(progress=lambda value, msg: self._bus.emit("progress", msg, value))

    def _prepare_ocr(self, settings: config.Settings) -> None:
        # Движок перезагружаем, только если пользователь выбрал другой.
        if self._ocr is None or self._ocr_name != settings.ocr_engine:
            self._bus.emit("status", "Загрузка модели распознавания текста...")
            self._ocr = create_with_fallback(
                settings.ocr_engine,
                settings.tesseract_path,
                on_message=lambda msg: self._bus.emit("status", msg),
            )
            self._ocr_name = self._ocr.name

    # -- остановка -----------------------------------------------------------

    def stop(self) -> None:
        """Останавливает потоки. Не блокирует GUI надолго."""
        self._stop_event.set()
        self._player.stop()
        threading.Thread(target=self._finish_stop, name="EngineStopper", daemon=True).start()

    def _finish_stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        self._threads = []
        with self._lock:
            self._running = False
        self._bus.emit("stopped", "Остановлено.")

    def preview_voice(self, text: str = "Привет! Так звучит выбранный голос.") -> None:
        """Проигрывает тестовую фразу выбранным голосом (в отдельном потоке)."""

        def task() -> None:
            settings = self._shared.snapshot()
            try:
                self._player.check_device()
                self._prepare_tts(settings)
                fragments = self._tts.synthesize(text, settings.speaker, settings.speed)
                for fragment in fragments:
                    self._player.play(fragment, settings.volume)
            except (TtsError, OcrError) as exc:
                self._bus.emit("error", str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ошибка проверки голоса")
                self._bus.emit("error", f"Ошибка проверки голоса: {exc}")

        threading.Thread(target=task, name="VoicePreview", daemon=True).start()
