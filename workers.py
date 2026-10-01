# -*- coding: utf-8 -*-
"""
Рабочие потоки NeuroVox.

Архитектура (главный поток занят только интерфейсом):

    [Поток снимков] --снимки--> [Буфер] --по очереди--> [Поток распознавания]
                                                              |
                                                            фразы
                                                              v
                                     [Очередь] --> [Поток озвучки и воспроизведения]

Снимки экрана делаются быстро (десятки раз в секунду) и не ждут медленного
распознавания: устоявшиеся снимки копятся в буфере, а распознавание разбирает их
строго по очереди. Поэтому ни одна реплика не теряется, даже если субтитр сменился,
пока читался предыдущий. Реплики из нескольких областей озвучиваются по одной, с паузой
между областями.

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
from typing import Callable, Dict, List, Optional

import numpy as np

import config
from capture import AreaTracker, CaptureError, ScreenCapture, preprocess_for_ocr
from frame_buffer import FrameBuffer
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
# Реплика в очереди озвучки
# ---------------------------------------------------------------------------

@dataclass
class SpeechItem:
    """Распознанная реплика, ожидающая озвучки."""

    area: int           # номер области, из которой взята реплика (0, 1, 2 ...)
    text: str
    created_at: float   # time.monotonic() в момент распознавания


def put_dropping_oldest(speech_queue: "queue.Queue[SpeechItem]", item: SpeechItem) -> None:
    """Кладёт реплику в очередь; при переполнении выбрасывает самую старую."""
    while True:
        try:
            speech_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                dropped = speech_queue.get_nowait()
                logger.info("Очередь озвучки переполнена, пропущена фраза: %s", dropped.text)
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Общая основа циклических потоков
# ---------------------------------------------------------------------------

class _LoopWorker(threading.Thread):
    """Основа рабочего потока: остановка по событию и учёт повторяющихся ошибок."""

    def __init__(self, name: str, bus: EventBus, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self._bus = bus
        self._stop_event = stop_event

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
# Поток снимков экрана
# ---------------------------------------------------------------------------

class CaptureWorker(_LoopWorker):
    """
    Быстро снимает все области экрана и складывает устоявшиеся снимки в буфер.

    Здесь нет распознавания текста, поэтому цикл укладывается в единицы миллисекунд
    на область и снимки идут с частотой ``capture_fps`` (по умолчанию 15 раз в секунду),
    что бы в это время ни делало распознавание.
    """

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        buffer: FrameBuffer,
        stop_event: threading.Event,
        capture_factory: Callable[[], ScreenCapture] = ScreenCapture,
    ) -> None:
        super().__init__("CaptureWorker", bus, stop_event)
        self._shared = shared
        self._buffer = buffer
        self._capture_factory = capture_factory

    def run(self) -> None:
        # mss нужно создавать именно в том потоке, где он используется.
        try:
            capture = self._capture_factory()
        except CaptureError as exc:
            self._bus.emit("error", str(exc))
            return

        trackers: List[AreaTracker] = []
        consecutive_errors = 0

        self._bus.emit("status", "Слежение за субтитрами запущено.")
        logger.info("Поток снимков запущен.")

        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                settings = self._shared.snapshot()
                rois = settings.rois
                if len(trackers) != len(rois):
                    # Число областей изменилось — начинаем слежение за каждой заново.
                    trackers = [AreaTracker() for _ in rois]
                    logger.info("Областей субтитров: %d", len(rois))

                failure: Optional[str] = None
                now = time.monotonic()
                for index, roi in enumerate(rois):
                    try:
                        image = preprocess_for_ocr(capture.grab(tuple(roi)))
                    except CaptureError as exc:
                        failure = str(exc)
                        continue
                    except Exception as exc:  # noqa: BLE001 — поток не должен падать из-за одного кадра
                        logger.exception("Непредвиденная ошибка в цикле снимков")
                        failure = f"Непредвиденная ошибка: {exc}"
                        continue

                    commit = trackers[index].update(image, now)
                    if commit is not None:
                        self._buffer.put(index, commit.image, settled=commit.settled, captured_at=now)

                if failure is not None:
                    consecutive_errors += 1
                    self._report_repeated_error(consecutive_errors, failure)
                    self._sleep_until(cycle_start, 1.0)
                    continue

                consecutive_errors = 0
                self._sleep_until(cycle_start, 1.0 / max(settings.capture_fps, config.MIN_CAPTURE_FPS))
        finally:
            capture.close()
            logger.info("Поток снимков остановлен.")


class _AreaHealthMonitor:
    """
    Отслеживает по каждой области, давно ли не было распознано ни одной фразы,
    и решает, когда пора предупредить пользователя, что рамка, возможно, смотрит
    не на игру (окно передвинули, сменили монитор/разрешение и т. п.).

    Идея простая: если конкретная область ЖИВАЯ (приходят непустые снимки —
    что-то в ней меняется, значит рамка вообще на что-то смотрит), но дольше
    ``warning_after`` секунд из неё не вышло ни одной распознанной фразы —
    это подозрительно, и стоит предупредить. Если снимков не приходит совсем
    (рамка указывает в совершенно статичное место — рабочий стол, чёрный
    экран) — предупреждение выдаётся по тому же таймеру: реальная субтитровая
    область почти наверняка за полторы минуты активной игры даст хотя бы одну
    реплику, а полное молчание настолько же подозрительно, насколько и «видим
    картинку, но текста в ней нет».

    Чтобы не спамить одним и тем же предупреждением: после срабатывания для
    области оно не повторяется, пока не появится новая распознанная фраза —
    именно она и означает, что всё в порядке и таймер можно снова обнулить.
    Так «тихая сцена» (в игре давно никто не говорит, но рамка настроена
    верно) даёт одно спокойное напоминание раз в ``warning_after`` секунд, а
    не поток сообщений на каждом цикле.
    """

    def __init__(self, warning_after: float = config.NO_SPEECH_WARNING_SECONDS) -> None:
        self._warning_after = warning_after
        self._last_speech_at: Dict[int, float] = {}
        self._warned: Dict[int, bool] = {}

    def reset(self, area: int, now: float) -> None:
        """Отмечает начало слежения за областью (или её появление заново)."""
        self._last_speech_at[area] = now
        self._warned[area] = False

    def on_phrase_recognized(self, area: int, now: float) -> None:
        """Фраза распознана — область точно наведена верно, таймер обнуляется."""
        self._last_speech_at[area] = now
        self._warned[area] = False

    def check(self, area: int, now: float) -> Optional[str]:
        """
        Возвращает текст предупреждения, если для области пора его выдать,
        иначе None. Одно и то же предупреждение не повторяется, пока не
        появится новая распознанная фраза (см. on_phrase_recognized).
        """
        last = self._last_speech_at.get(area)
        if last is None:
            self.reset(area, now)
            return None
        if self._warned.get(area):
            return None
        if now - last < self._warning_after:
            return None
        self._warned[area] = True
        seconds = int(self._warning_after)
        return (
            f"Область {area + 1}: субтитры не обнаруживаются {seconds} секунд — "
            f"проверьте, что рамка наведена на игру."
        )

    def forget_missing(self, active_areas: range) -> None:
        """Убирает состояние удалённых областей (число рамок могло уменьшиться)."""
        for area in list(self._last_speech_at):
            if area not in active_areas:
                del self._last_speech_at[area]
                self._warned.pop(area, None)


# ---------------------------------------------------------------------------
# Поток распознавания
# ---------------------------------------------------------------------------

class OcrWorker(_LoopWorker):
    """
    Берёт снимки из буфера по очереди, распознаёт текст и отправляет реплики на озвучку.

    Пока идёт распознавание одного снимка, поток снимков продолжает работать, а
    новые устоявшиеся снимки ждут своей очереди в буфере — ничего не пропадает.
    Для каждой области ведётся свой фильтр повторов.
    """

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        buffer: FrameBuffer,
        speech_queue: "queue.Queue[SpeechItem]",
        stop_event: threading.Event,
        ocr: BaseOcr,
        no_speech_warning_seconds: float = config.NO_SPEECH_WARNING_SECONDS,
    ) -> None:
        super().__init__("OcrWorker", bus, stop_event)
        self._shared = shared
        self._buffer = buffer
        # Вынесено в параметр (а не жёстко config.NO_SPEECH_WARNING_SECONDS внутри
        # run()) главным образом ради тестируемости: тест диагностики может
        # передать короткий порог вместо того, чтобы ждать минуты полторы.
        self._no_speech_warning_seconds = no_speech_warning_seconds
        self._speech_queue = speech_queue
        self._ocr = ocr

    def run(self) -> None:
        filters: Dict[int, SubtitleFilter] = {}
        area_count = 0
        consecutive_errors = 0
        last_purge = time.monotonic()
        health = _AreaHealthMonitor(self._no_speech_warning_seconds)
        logger.info("Поток распознавания запущен.")

        try:
            while not self._stop_event.is_set():
                frame = self._buffer.take_next(timeout=0.2)

                # Прочитанные снимки удаляем через FRAME_RETENTION_SECONDS, устаревшие — тоже.
                now = time.monotonic()
                if now - last_purge >= 1.0:
                    self._buffer.purge()
                    last_purge = now

                settings = self._shared.snapshot()
                if len(settings.rois) != area_count:
                    # Области добавили или убрали: нумерация могла сдвинуться — фильтры начинаем заново.
                    filters.clear()
                    area_count = len(settings.rois)
                    health.forget_missing(range(area_count))
                    for area in range(area_count):
                        health.reset(area, now)

                # Диагностика проверяется на каждом цикле, а не только когда пришёл
                # снимок: область, откуда вообще ничего не приходит (рамка смотрит
                # в совершенно статичное место), тоже должна быть замечена.
                for area in range(area_count):
                    warning = health.check(area, now)
                    if warning:
                        self._bus.emit("error", warning)

                if frame is None:
                    continue
                if frame.area >= area_count:
                    continue  # область уже удалена

                subtitle_filter = filters.get(frame.area)
                if subtitle_filter is None:
                    subtitle_filter = filters[frame.area] = SubtitleFilter(
                        threshold=settings.similarity_threshold
                    )
                subtitle_filter.threshold = settings.similarity_threshold

                if frame.is_marker:
                    # Область опустела: реплика закончилась, такая же фраза позже — уже новая.
                    subtitle_filter.forget()
                    continue

                try:
                    raw_text = self._ocr.recognize(frame.image)
                    consecutive_errors = 0
                except OcrError as exc:
                    consecutive_errors += 1
                    self._report_repeated_error(consecutive_errors, str(exc))
                    self._stop_event.wait(0.5)
                    continue
                except Exception as exc:  # noqa: BLE001
                    consecutive_errors += 1
                    logger.exception("Непредвиденная ошибка распознавания")
                    self._report_repeated_error(consecutive_errors, f"Непредвиденная ошибка: {exc}")
                    self._stop_event.wait(0.5)
                    continue

                phrase = subtitle_filter.process(raw_text, settled=frame.settled)
                if phrase:
                    health.on_phrase_recognized(frame.area, now)
                    label = f"[{frame.area + 1}] " if area_count > 1 else ""
                    self._bus.emit("recognized", f"{label}{phrase}")
                    put_dropping_oldest(
                        self._speech_queue, SpeechItem(frame.area, phrase, time.monotonic())
                    )
        finally:
            logger.info("Поток распознавания остановлен.")


# ---------------------------------------------------------------------------
# Поток озвучки
# ---------------------------------------------------------------------------

class TtsPlaybackWorker(threading.Thread):
    """
    Берёт реплики из очереди по одной, синтезирует речь и воспроизводит.

    Реплики читаются строго по очереди — две не звучат одновременно. Между репликами
    из разных областей выдерживается пауза ``area_pause``, между репликами одной
    области — короткая пауза. Пока играет одна реплика, следующая ждёт в очереди; её
    синтез идёт до паузы, чтобы пауза не растягивалась на время синтеза.
    """

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        speech_queue: "queue.Queue[SpeechItem]",
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
        self._last_area: Optional[int] = None
        self._last_finished: float = 0.0

    def run(self) -> None:
        logger.info("Поток озвучки запущен.")
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._speech_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Реплика, простоявшая в очереди слишком долго, в игре уже неактуальна.
                if time.monotonic() - item.created_at > config.MAX_PHRASE_AGE_SECONDS:
                    logger.info("Реплика устарела и пропущена: %s", item.text)
                    continue
                self._speak(item)
        finally:
            self._player.stop()
            logger.info("Поток озвучки остановлен.")

    def _wait_turn(self, area: int, area_pause: float) -> None:
        """Выдерживает паузу после предыдущей реплики (дольше — при смене области)."""
        if self._last_area is None:
            return
        gap = area_pause if area != self._last_area else config.SAME_AREA_GAP_SECONDS
        remaining = gap - (time.monotonic() - self._last_finished)
        if remaining > 0:
            self._stop_event.wait(remaining)

    def _speak(self, item: SpeechItem) -> None:
        settings = self._shared.snapshot()
        try:
            started = time.perf_counter()
            fragments = self._tts.synthesize(item.text, settings.speaker, settings.speed)
            if not fragments:
                return
            self._bus.emit("spoken", item.text, time.perf_counter() - started)
            self._wait_turn(item.area, settings.area_pause)
            for fragment in fragments:
                if self._stop_event.is_set():
                    return
                self._player.play(fragment, settings.volume)
            self._last_area = item.area
            self._last_finished = time.monotonic()
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
        self._speech_queue: "queue.Queue[SpeechItem]" = queue.Queue(maxsize=config.MAX_TTS_QUEUE)
        self._buffer = FrameBuffer()
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
        self._buffer = FrameBuffer()
        capture_thread = CaptureWorker(self._shared, self._bus, self._buffer, self._stop_event)
        ocr_thread = OcrWorker(
            self._shared, self._bus, self._buffer, self._speech_queue, self._stop_event, self._ocr
        )
        tts_thread = TtsPlaybackWorker(
            self._shared, self._bus, self._speech_queue, self._stop_event, self._tts, self._player
        )
        self._threads = [capture_thread, ocr_thread, tts_thread]
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
        self._buffer.clear()   # снимки прошлого сеанса при следующем запуске не нужны
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
