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
строго по очереди. Поэтому ни одна реплика не теряется при РАСПОЗНАВАНИИ, даже если
субтитр сменился, пока читался предыдущий, — каждая попадает в очередь на озвучку.
Реплики из нескольких областей озвучиваются по одной, с паузой между областями.

Озвучка (синтез речи и воспроизведение) на обычном процессоре без видеокарты занимает
заметное время — секунды, а не миллисекунды, — и может идти медленнее, чем в игре
меняются субтитры. Чтобы из-за этого не накапливалось отставание (когда программа
честно проговаривает ВСЮ очередь, но всё сильнее отстаёт от того, что уже на экране),
поток озвучки, прежде чем произнести очередную реплику, проверяет: не пришла ли для той
же области более новая, пока эта ждала своей очереди (см. skip_stale_duplicates). Если
да — устаревшие пропускаются, и озвучивается самая свежая: так голос всегда звучит
вровень с текущими субтитрами, а не читает историю диалога с нарастающей задержкой.

Потоки общаются через thread-safe очереди и события. Любые сообщения для
интерфейса отправляются в очередь событий, а GUI сам забирает их в своём
потоке — так окно никогда не зависает и не вызывается из чужих потоков.

Поток снимков перед каждым снимком экрана может (через OverlayCaptureGate)
попросить GUI на мгновение спрятать окна рамок — иначе рамка и её подпись
попадают в кадр, который видит OCR, и распознаются как часть субтитров.
"""

import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import config
from capture import AreaTracker, CaptureError, ScreenCapture, preprocess_for_ocr
from frame_buffer import FrameBuffer
from ocr_engine import BaseOcr, OcrError, create_with_fallback
from pronunciation import SpeakerNameList
from text_filter import SubtitleFilter, clean_text
from tts_engine import SileroTts, TtsError

logger = logging.getLogger("neurovox.workers")


# ---------------------------------------------------------------------------
# События для интерфейса
# ---------------------------------------------------------------------------

@dataclass
class UiEvent:
    """Сообщение из рабочего потока в интерфейс."""

    # "status" | "recognized" | "spoken" | "error" | "progress" | "ready" | "stopped" |
    # "speaking_start" | "speaking_end" (value — номер области) | "skipped"
    kind: str
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
# Синхронное скрытие рамок на время снимка (чтобы OCR не читал саму рамку)
# ---------------------------------------------------------------------------

class OverlayCaptureGate:
    """
    Мост между потоком снимков (CaptureWorker, фоновый поток) и окнами рамок
    (SelectionOverlay, живут только в главном потоке Tk) — безопасно просит
    главный поток на мгновение убрать все рамки с экрана перед снимком и
    вернуть их обратно сразу после, не трогая сами окна из чужого потока.

    Зачем это нужно. SelectionOverlay уже пытается исключить себя из захвата
    средствами Windows (см. overlay._exclude_from_capture), но этот способ
    опирается на то, что сам захват экрана поддерживает такое исключение, —
    а mss снимает экран через GDI/BitBlt, и на части систем (видеокарта,
    драйвер, сборка Windows) это исключение для таких снимков не действует.
    Тогда в кадр, который видит OCR, попадает контур рамки и её подпись
    («Область субтитров N — перетащите, тяните за угол»), и распознавание
    читает эту надпись вместо (или вместе с) субтитров игры.

    Единственный способ, который работает всегда и везде, — физически убрать
    окно рамки с экрана (withdraw) на время самого снимка и сразу показать
    обратно. Это и делает этот класс, синхронно:

        gate.request_hide()   # из CaptureWorker, ждёт, пока GUI спрячет рамки
        image = capture.grab(roi)
        gate.request_show()   # из CaptureWorker, не ждёт — показ не обязан
                               # блокировать следующий снимок

    GUI со своей стороны регистрируется один раз при старте слежения
    (см. MainWindow._install_capture_gate) и снимает регистрацию при
    остановке — после этого gate ничего не ждёт и методы мгновенно
    возвращаются (чтобы ошибка в GUI или его закрытие не подвесили поток
    снимков навсегда).
    """

    # Сколько ждать ответа от GUI на один запрос. GUI опрашивает этот мост
    # каждые OVERLAY_GATE_POLL_MS миллисекунд (см. gui.py), поэтому реальное
    # время отклика обычно куда меньше этого потолка; таймаут — только защита
    # от зависшего или перегруженного интерфейса, чтобы поток снимков не
    # застрял навсегда и в худшем случае просто снял кадр с видимой рамкой.
    _RESPONSE_TIMEOUT_S = 0.25

    def __init__(self) -> None:
        self._hide_fn: Optional[Callable[[], None]] = None
        self._show_fn: Optional[Callable[[], None]] = None
        self._request = threading.Event()
        self._done = threading.Event()
        self._pending_show = False
        self._lock = threading.Lock()

    def install(self, hide_fn: Callable[[], None], show_fn: Callable[[], None]) -> None:
        """Регистрирует GUI-колбэки (вызывается из главного потока при старте)."""
        with self._lock:
            self._hide_fn = hide_fn
            self._show_fn = show_fn
            self._pending_show = False
            self._done.clear()
            self._request.clear()

    def uninstall(self) -> None:
        """Снимает регистрацию (вызывается из главного потока при остановке)."""
        with self._lock:
            self._hide_fn = None
            self._show_fn = None
            self._request.clear()
            self._done.set()   # любой, кто сейчас ждёт — не должен зависнуть навсегда

    def request_hide(self) -> bool:
        """
        Просит GUI спрятать рамки и ждёт подтверждения (вызывается из
        CaptureWorker). Возвращает True, если GUI подтвердил скрытие вовремя —
        тогда после снимка обязательно нужно вызвать request_show().
        Если GUI не зарегистрирован (слежение ещё не запущено через окно,
        например в тестах) или не успел ответить за _RESPONSE_TIMEOUT_S —
        возвращает False и ничего не блокирует: снимок будет сделан с рамкой
        как есть, лучше так, чем подвесить поток снимков насовсем.
        """
        with self._lock:
            if self._hide_fn is None:
                return False
            self._pending_show = False
            self._done.clear()
            self._request.set()
        confirmed = self._done.wait(self._RESPONSE_TIMEOUT_S)
        return confirmed and self._hide_fn is not None

    def request_show(self) -> None:
        """
        Просит GUI вернуть рамки на экран (вызывается из CaptureWorker сразу
        после снимка, если request_hide() вернул True). Не ждёт ответа — пока
        рамка возвращается на экран, следующий снимок делать уже можно, показ
        не обязан его блокировать.
        """
        with self._lock:
            if self._show_fn is None:
                return
            self._pending_show = True
            self._request.set()

    # -- вызывается из главного потока (GUI) ---------------------------------

    def poll(self) -> None:
        """
        Выполняет накопившийся запрос, если он есть (вызывается из главного
        потока Tk коротким таймером, см. gui.py). Сам решает, спрятать рамки
        или показать — приоритет у более позднего запроса: если за один тик
        накопились и hide, и show, выполняется только последний (show), а
        не оба подряд — это по-прежнему корректно, потому что CaptureWorker
        уже получил бы confirmed=False и снял бы кадр с видимой рамкой.
        """
        with self._lock:
            if not self._request.is_set():
                return
            self._request.clear()
            pending_show = self._pending_show
            hide_fn, show_fn = self._hide_fn, self._show_fn
        if pending_show:
            if show_fn is not None:
                show_fn()
        else:
            if hide_fn is not None:
                hide_fn()
            self._done.set()


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


def skip_stale_duplicates(
    speech_queue: "queue.Queue[SpeechItem]", item: SpeechItem
) -> "Tuple[SpeechItem, int]":
    """
    Если, пока ``item`` ждала своей очереди, для ТОЙ ЖЕ области уже пришла более
    новая реплика — пропускает устаревшую и сразу переходит к самой свежей.

    Зачем это нужно. Синтез и воспроизведение одной фразы на реальном железе (без
    видеокарты) занимает секунды, а субтитры в диалоге могут меняться чаще. Без этой
    функции очередь просто растёт, и спустя минуту диалога озвучка начинает заметно
    отставать от того, что уже давно на экране — читает всё подряд, но с нарастающей
    задержкой. Раз докладчик физически не может говорить быстрее, чем на самом деле
    уходит время на синтез и произнесение фразы, единственный способ не отставать —
    не пытаться озвучить КАЖДУЮ реплику из backlog'а, а при появлении более новой для
    той же области сразу переходить к ней: так озвучка всегда звучит вровень с
    текущими субтитрами, а не с теми, что были полминуты назад.

    Реплики ДРУГИХ областей, оказавшиеся в очереди впереди неё, не теряются — они
    возвращаются в очередь в прежнем порядке и дожидаются своей очереди как обычно;
    пропускаются только устаревшие реплики ТОЙ ЖЕ области, что и ``item``.

    Возвращает (реплику, которую нужно озвучить, число пропущенных устаревших реплик
    той же области — 0, если ``item`` и так самая свежая).
    """
    deferred: List[SpeechItem] = []
    skipped = 0
    latest = item
    while True:
        try:
            candidate = speech_queue.get_nowait()
        except queue.Empty:
            break
        if candidate.area == latest.area:
            skipped += 1
            latest = candidate
        else:
            deferred.append(candidate)
    # Возвращаем на место реплики других областей — в том же порядке, в котором
    # их забрали, так что относительный порядок между ними не меняется. Очередь
    # переполниться не может: мы вернём не больше элементов, чем только что забрали.
    for pending in deferred:
        put_dropping_oldest(speech_queue, pending)
    return latest, skipped


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
        overlay_gate: Optional[OverlayCaptureGate] = None,
    ) -> None:
        super().__init__("CaptureWorker", bus, stop_event)
        self._shared = shared
        self._buffer = buffer
        self._capture_factory = capture_factory
        # Необязателен: NeuroVoxEngine без окна (preview_recognition, тесты)
        # просто не передаёт его, и снимки делаются как раньше, без скрытия
        # рамки, — на уровне окна этот способ и не нужен (рамки нет вовсе).
        self._overlay_gate = overlay_gate

    def _grab_area(self, capture: ScreenCapture, roi) -> np.ndarray:
        """
        Снимает одну область, на время самого снимка пряча рамки через
        overlay_gate (если он установлен и GUI сейчас его обслуживает) — см.
        OverlayCaptureGate выше. Рамка возвращается на экран сразу после
        снимка, независимо от того, успешен он был или бросил исключение.
        """
        gate = self._overlay_gate
        if gate is None:
            return capture.grab(roi)
        hidden = gate.request_hide()
        try:
            return capture.grab(roi)
        finally:
            if hidden:
                gate.request_show()

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
                        image = preprocess_for_ocr(self._grab_area(capture, tuple(roi)))
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
        speaker_names: Optional[SpeakerNameList] = None,
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
        # Список имён для режима "list" отсечения имени говорящего (см.
        # SubtitleFilter.speaker_name_mode ниже) — None, если вызывающий код
        # его не передал (тесты, которые не используют эту настройку вовсе).
        self._speaker_names = speaker_names

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
                        threshold=settings.similarity_threshold,
                        speaker_name_mode=settings.speaker_name_mode,
                        speaker_names=self._speaker_names,
                    )
                subtitle_filter.threshold = settings.similarity_threshold
                subtitle_filter.speaker_name_mode = settings.speaker_name_mode

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
    области — короткая пауза.

    Синтез следующей реплики готовится ЗАРАНЕЕ, в отдельном потоке, пока звучит
    текущая (см. _prefetch): синтез речи на CPU без видеокарты занимает заметное
    время — не миллисекунды, а иногда больше секунды на фразу, — и раньше это время
    целиком добавлялось к задержке каждой реплики (сначала синтез, потом пауза,
    потом воспроизведение, одно за другим). Теперь, пока играет реплика N, уже идёт
    синтез реплики N+1 — и к моменту, когда реплика N доиграна, озвучка реплики N+1
    чаще всего уже готова и остаётся только выдержать паузу и начать её сразу,
    вместо того чтобы ждать ещё и синтез в придачу. skip_stale_duplicates
    по-прежнему применяется к реплике непосредственно перед тем, как она реально
    синтезируется, поэтому упреждающий синтез не озвучивает то, что должно быть
    пропущено как устаревшее.
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
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_result: "Optional[List[Optional[_PreparedSpeech]]]" = None

    def run(self) -> None:
        logger.info("Поток озвучки запущен.")
        pending: Optional[_PreparedSpeech] = None
        try:
            while not self._stop_event.is_set():
                if pending is not None:
                    # Это результат упреждающего синтеза (_start_prefetch), который
                    # готовился, пока звучала ПРЕДЫДУЩАЯ реплика — за то время, что
                    # он ждал своей очереди здесь, для его области могла прийти ещё
                    # более новая реплика, и её нужно поймать именно сейчас, перед
                    # тем как её озвучить (см. docstring класса). Для реплики,
                    # которая только что получена через _fetch_and_prepare ниже,
                    # эта проверка уже была сделана ВНУТРИ неё — повторять её здесь
                    # не только лишне, но и неверно: такая реплика ещё ни разу не
                    # звучала, и более новая реплика той же области, появившаяся
                    # в очереди прямо во время её первого синтеза, пришла едва ли
                    # на мгновение позже — это не «отставание», а обычный порядок
                    # появления субтитров, и первую реплику в этом случае нужно
                    # всё равно озвучить, а не пропустить как устаревшую.
                    prepared = pending
                    pending = None
                    item, skipped = skip_stale_duplicates(self._speech_queue, prepared.item)
                    if skipped:
                        self._announce_skip(item, skipped)
                        prepared = self._resynthesize_if_item_changed(prepared, item)
                    if prepared is None or self._is_too_old(item):
                        if prepared is None:
                            logger.info("Реплика устарела и пропущена: %s", item.text)
                        continue
                else:
                    prepared = self._fetch_and_prepare(block=True)
                    if prepared is None:
                        continue   # таймаут ожидания (_stop_event мог быть выставлен) — проверяем цикл заново

                self._speak(prepared)

                # Синтез следующей реплики начинается сразу после того, как текущая
                # начала играть (внутри _speak, в фоновом потоке) — к этому моменту
                # он уже должен был завершиться или вот-вот завершится.
                pending = self._collect_prefetch()
        finally:
            self._player.stop()
            logger.info("Поток озвучки остановлен.")

    # -- подготовка (синтез) реплик, с упреждением ---------------------------

    def _fetch_and_prepare(self, block: bool) -> "Optional[_PreparedSpeech]":
        """Достаёт одну реплику из очереди и сразу синтезирует её (без упреждения)."""
        try:
            item = self._speech_queue.get(timeout=0.2) if block else self._speech_queue.get_nowait()
        except queue.Empty:
            return None
        item, skipped = skip_stale_duplicates(self._speech_queue, item)
        if skipped:
            self._announce_skip(item, skipped)
        if self._is_too_old(item):
            logger.info("Реплика устарела и пропущена: %s", item.text)
            return None
        return self._synthesize(item)

    def _is_too_old(self, item: SpeechItem) -> bool:
        return time.monotonic() - item.created_at > config.MAX_PHRASE_AGE_SECONDS

    def _announce_skip(self, item: SpeechItem, skipped: int) -> None:
        area_count = len(self._shared.snapshot().rois)
        label = f"Область {item.area + 1}: " if area_count > 1 else ""
        message = f"{label}пропущено устаревших реплик: {skipped} — озвучиваем актуальную: {item.text}"
        logger.info(message)
        self._bus.emit("skipped", message)

    def _synthesize(self, item: SpeechItem) -> "Optional[_PreparedSpeech]":
        settings = self._shared.snapshot()
        try:
            started = time.perf_counter()
            fragments = self._tts.synthesize(item.text, settings.speaker, settings.speed)
        except TtsError as exc:
            logger.error("Ошибка озвучки: %s", exc)
            self._bus.emit("error", str(exc))
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Непредвиденная ошибка озвучки")
            self._bus.emit("error", f"Непредвиденная ошибка озвучки: {exc}")
            return None
        if not fragments:
            return None
        return _PreparedSpeech(item=item, fragments=fragments, synth_seconds=time.perf_counter() - started)

    def _resynthesize_if_item_changed(
        self, prepared: "_PreparedSpeech", item: SpeechItem
    ) -> "Optional[_PreparedSpeech]":
        """
        Если skip_stale_duplicates заменил реплику на более свежую, то, что было
        синтезировано заранее (для старой реплики), больше не годится — нужно
        синтезировать актуальный текст. Если же реплика не изменилась, уже
        готовый результат используется как есть (без повторного синтеза).
        """
        if item is prepared.item:
            return prepared
        return self._synthesize(item)

    def _start_prefetch(self) -> None:
        """
        Запускает в фоновом потоке подготовку следующей реплики (взятие из
        очереди + синтез), пока текущая реплика звучит в этом потоке. Если
        очередь сейчас пуста, фоновый поток недолго подождёт (timeout у
        _speech_queue.get) и тихо завершится с результатом None — это не
        ошибка, а обычный случай, когда субтитры идут неторопливо и новая
        реплика ещё не появилась.
        """
        result: "List[Optional[_PreparedSpeech]]" = [None]

        def worker() -> None:
            result[0] = self._fetch_and_prepare(block=True)

        thread = threading.Thread(target=worker, name="TtsPrefetch", daemon=True)
        thread.start()
        self._prefetch_thread = thread
        self._prefetch_result = result

    def _collect_prefetch(self) -> "Optional[_PreparedSpeech]":
        """
        Дожидается завершения фоновой подготовки, запущенной _start_prefetch,
        и возвращает её результат (None, если очередь была пуста всё это время
        или остановка программы прервала ожидание).
        """
        thread = self._prefetch_thread
        if thread is None:
            return None
        thread.join()
        result = self._prefetch_result[0]
        self._prefetch_thread = None
        self._prefetch_result = None
        return result

    # -- воспроизведение ------------------------------------------------------

    def _wait_turn(self, area: int, area_pause: float) -> None:
        """Выдерживает паузу после предыдущей реплики (дольше — при смене области)."""
        if self._last_area is None:
            return
        gap = area_pause if area != self._last_area else config.SAME_AREA_GAP_SECONDS
        remaining = gap - (time.monotonic() - self._last_finished)
        if remaining > 0:
            self._stop_event.wait(remaining)

    def _speak(self, prepared: "_PreparedSpeech") -> None:
        item, fragments = prepared.item, prepared.fragments
        settings = self._shared.snapshot()
        self._bus.emit("spoken", item.text, prepared.synth_seconds)
        self._wait_turn(item.area, settings.area_pause)
        # Синтез следующей реплики запускается прямо сейчас, в фоновом потоке,
        # — до начала паузы и воспроизведения этой, чтобы он шёл параллельно с
        # ними, а не добавлялся к задержке после. _collect_prefetch() в run()
        # дожидается его результата уже после того, как эта реплика доиграна.
        self._start_prefetch()
        # "speaking_start"/"speaking_end" обрамляют именно звучание, а не синтез и
        # не паузу перед ним — интерфейс использует их, чтобы на это время прятать
        # рамку области (см. SelectionOverlay.set_reading). finally гарантирует
        # парный "speaking_end" и при обычном завершении, и при ранней остановке
        # (_stop_event), и если воспроизведение само бросит исключение.
        self._bus.emit("speaking_start", value=float(item.area))
        try:
            for fragment in fragments:
                if self._stop_event.is_set():
                    return
                self._player.play(fragment, settings.volume)
        finally:
            self._bus.emit("speaking_end", value=float(item.area))
        self._last_area = item.area
        self._last_finished = time.monotonic()


@dataclass
class _PreparedSpeech:
    """Реплика, для которой синтез речи уже выполнен и ждёт своей очереди на воспроизведение."""

    item: SpeechItem
    fragments: List[np.ndarray]
    synth_seconds: float


# ---------------------------------------------------------------------------
# Управляющий класс
# ---------------------------------------------------------------------------

class NeuroVoxEngine:
    """
    Управляет жизненным циклом всех потоков: подготовка моделей, запуск, остановка.

    Тяжёлая подготовка (скачивание и загрузка моделей) идёт в отдельном
    потоке, поэтому окно остаётся отзывчивым.
    """

    def __init__(
        self,
        shared: SharedSettings,
        bus: EventBus,
        capture_factory: Callable[[], ScreenCapture] = ScreenCapture,
        overlay_gate: Optional[OverlayCaptureGate] = None,
    ) -> None:
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
        # Та же фабрика, что получает CaptureWorker — по умолчанию настоящий
        # экран, в тестах подменяется поддельным источником кадров (см.
        # preview_recognition, единственное место, где NeuroVoxEngine сам
        # создаёт ScreenCapture — CaptureWorker строит собственный).
        self._capture_factory = capture_factory
        # Мост к окнам рамок (см. OverlayCaptureGate выше) — передаётся из
        # MainWindow, чтобы CaptureWorker мог на время снимка прятать рамки,
        # не трогая их окна напрямую из чужого потока. Без окна (тесты,
        # preview_recognition) остаётся None — снимки делаются как раньше.
        self._overlay_gate = overlay_gate
        # Список имён говорящих для режима "list" отсечения имени (см.
        # text_filter.strip_speaker_name и config.speaker_name_mode) — живёт
        # здесь по тому же принципу, что StressDictionary внутри SileroTts:
        # один экземпляр на весь запуск программы, файл на диске перечитывается
        # сам при изменении (см. pronunciation.SpeakerNameList).
        self._speaker_names = SpeakerNameList()

    @property
    def is_running(self) -> bool:
        return self._running

    def available_speakers(self) -> List[str]:
        """
        Голоса, реально доступные в данный момент для озвучки.

        Пока модель ещё не загружена (программа не запускалась, или была
        запущена с другой моделью), возвращает статичный список-заглушку —
        тот же, что вернула бы SileroTts.available_speakers() без модели.
        Безопасен для вызова из потока интерфейса: ничего не загружает и не
        блокирует, только читает уже загруженную (или ещё не загруженную)
        модель — актуальный список появляется после события "ready".
        """
        if self._tts is None:
            return list(config.SILERO_SPEAKERS)
        return self._tts.available_speakers()

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
        capture_thread = CaptureWorker(
            self._shared, self._bus, self._buffer, self._stop_event,
            overlay_gate=self._overlay_gate,
        )
        ocr_thread = OcrWorker(
            self._shared, self._bus, self._buffer, self._speech_queue, self._stop_event, self._ocr,
            speaker_names=self._speaker_names,
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

    def preview_recognition(self) -> None:
        """
        Делает один снимок каждой настроенной области, распознаёт текст через
        текущий выбранный OCR-движок и выводит в журнал и сырой, и очищенный
        (text_filter.clean_text) результат — без обращения к TTS и плееру.

        Нужна, чтобы проверить, что рамка действительно наведена на субтитры
        и OCR их читает, не дожидаясь полноценного запуска слежения и не
        слушая озвучку: достаточно посмотреть в журнал программы.

        Движок OCR грузится лениво при первом использовании (как и TTS в
        preview_voice) — если ещё не загружен, пользователь увидит статус
        загрузки и результат появится чуть позже, интерфейс при этом не
        замирает (вся работа — в отдельном потоке).
        """

        def task() -> None:
            settings = self._shared.snapshot()
            if not settings.rois:
                self._bus.emit("error", "Нет ни одной настроенной области — нечего проверять.")
                return
            try:
                self._prepare_ocr(settings)
            except OcrError as exc:
                self._bus.emit("error", str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ошибка загрузки распознавания при проверке")
                self._bus.emit("error", f"Ошибка загрузки распознавания: {exc}")
                return

            try:
                capture = self._capture_factory()
            except CaptureError as exc:
                self._bus.emit("error", str(exc))
                return

            try:
                multiple = len(settings.rois) > 1
                for index, roi in enumerate(settings.rois):
                    label = f"Область {index + 1}: " if multiple else ""
                    try:
                        frame = capture.grab(roi)
                    except CaptureError as exc:
                        self._bus.emit("error", f"{label}не удалось сделать снимок: {exc}")
                        continue

                    image = preprocess_for_ocr(frame)
                    if image is None:
                        self._bus.emit(
                            "status",
                            f"{label}текст не обнаружен (похожей на текст области в кадре нет).",
                        )
                        continue

                    try:
                        raw_text = self._ocr.recognize(image)
                    except OcrError as exc:
                        self._bus.emit("error", f"{label}ошибка распознавания: {exc}")
                        continue

                    cleaned = clean_text(raw_text, settings.speaker_name_mode, self._speaker_names)
                    if not cleaned:
                        self._bus.emit(
                            "status",
                            f"{label}похожая на текст область есть, но распознать ничего не "
                            f"удалось (сырой ответ движка: {raw_text!r}).",
                        )
                        continue
                    self._bus.emit("status", f"{label}сырой текст: {raw_text!r}")
                    self._bus.emit("status", f"{label}очищенный текст: {cleaned!r}")
            finally:
                capture.close()

        threading.Thread(target=task, name="RecognitionPreview", daemon=True).start()
