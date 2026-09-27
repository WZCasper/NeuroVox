# -*- coding: utf-8 -*-
"""
Подделки для проверки всего конвейера «снимок -> буфер -> распознавание -> озвучка».

Реальные потоки работают на реальном времени, но вместо экрана, нейросети распознавания
и звуковой карты используются имитации:

  * FakeScreen — «показывает» субтитры по сценарию (что видно в каждой области в
    каждый момент времени) и рисует их как настоящие кадры;
  * FakeOcr    — узнаёт нарисованный текст и тратит заданное время на распознавание
    (у настоящего EasyOCR на процессоре это сотни миллисекунд);
  * FakeTts / FakePlayer — записывают, какая реплика и когда звучала.

Так можно проверить порядок, паузы и то, что быстрые субтитры не теряются, — без
экрана, моделей и звука.
"""

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from capture import preprocess_for_ocr  # noqa: E402
from ocr_engine import BaseOcr  # noqa: E402

FRAME_H, FRAME_W = 100, 640

# Сценарий: (номер области, время t в секундах от старта) -> текст на экране или None (пусто).
Timeline = Callable[[int, float], Optional[str]]


def render(text: Optional[str]) -> np.ndarray:
    """Рисует кадр области: светлый текст на тёмном фоне (или пустой фон)."""
    frame = np.full((FRAME_H, FRAME_W, 3), (30, 34, 50), dtype=np.uint8)
    if text:
        cv2.putText(frame, text, (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return frame


def segments(items: Sequence[Tuple[float, float, str]]) -> Callable[[float], Optional[str]]:
    """Сценарий одной области из отрезков (начало, конец, текст)."""

    def at(t: float) -> Optional[str]:
        for start, end, text in items:
            if start <= t < end:
                return text
        return None

    return at


class FakeScreen:
    """Вместо ScreenCapture: кадр зависит от области (по её координате top) и от времени."""

    def __init__(self, rois: Sequence[Sequence[int]], timeline: Timeline, started: float) -> None:
        self._area_by_top: Dict[int, int] = {int(roi[1]): index for index, roi in enumerate(rois)}
        self._timeline = timeline
        self._started = started

    def grab(self, roi) -> np.ndarray:
        area = self._area_by_top[int(roi[1])]
        return render(self._timeline(area, time.monotonic() - self._started))

    def close(self) -> None:
        pass


class FakeOcr(BaseOcr):
    """Узнаёт заранее известные тексты по картинке; распознавание «занимает» latency секунд."""

    name = "fake"

    def __init__(self, texts: Sequence[str], latency: float) -> None:
        self.latency = latency
        self.calls = 0
        self._references: List[Tuple[str, np.ndarray]] = []
        for text in texts:
            image = preprocess_for_ocr(render(text))
            assert image is not None, f"текст «{text}» не превратился в картинку"
            self._references.append((text, image))

    def load(self) -> None:
        pass

    def recognize(self, image: np.ndarray) -> str:
        self.calls += 1
        time.sleep(self.latency)
        best_text, best_diff = "", 1e9
        for text, reference in self._references:
            if reference.shape != image.shape:
                continue
            diff = float(cv2.absdiff(reference, image).mean())
            if diff < best_diff:
                best_text, best_diff = text, diff
        return best_text if best_diff < 8.0 else ""


class FakeTts:
    """Вместо SileroTts: запоминает фразы; звук — массив с номером фразы в первом отсчёте."""

    model_name = "fake"
    is_loaded = True

    def __init__(self) -> None:
        self.spoken: List[str] = []

    def synthesize(self, text: str, speaker: str, speed: float = 1.0):
        self.spoken.append(text)
        return [np.full(64, float(len(self.spoken)), dtype=np.float32)]


class FakePlayer:
    """Вместо AudioPlayer: «проигрывает» звук заданное время и записывает интервалы."""

    def __init__(self, tts: FakeTts, play_seconds: float = 0.15) -> None:
        self._tts = tts
        self._play_seconds = play_seconds
        self.events: List[Tuple[str, float, float]] = []   # (фраза, начало, конец)

    def play(self, samples: np.ndarray, volume: float = 1.0) -> None:
        text = self._tts.spoken[int(samples[0]) - 1]
        begin = time.monotonic()
        time.sleep(self._play_seconds)
        self.events.append((text, begin, time.monotonic()))

    def stop(self) -> None:
        pass

    @property
    def phrases(self) -> List[str]:
        return [event[0] for event in self.events]


def run_pipeline(
    timeline: Timeline,
    rois: Sequence[Sequence[int]],
    texts: Sequence[str],
    duration: float,
    ocr_latency: float = 0.5,
    area_pause: float = 0.8,
    capture_fps: float = 15.0,
    play_seconds: float = 0.15,
    expected: Optional[int] = None,
):
    """
    Запускает настоящие потоки на подделках и возвращает (плеер, OCR, буфер).

    Работает до истечения ``duration`` секунд или (если задан ``expected``) до тех пор,
    пока не прозвучит столько реплик и не пройдёт ещё немного времени — чтобы поймать
    лишние повторы.
    """
    import config
    from frame_buffer import FrameBuffer
    from workers import CaptureWorker, EventBus, OcrWorker, SharedSettings, TtsPlaybackWorker

    settings = config.Settings()
    settings.rois = [list(roi) for roi in rois]
    settings.capture_fps = capture_fps
    settings.area_pause = area_pause
    shared = SharedSettings(settings)
    bus = EventBus()
    buffer = FrameBuffer()
    speech_queue: "queue.Queue" = queue.Queue(maxsize=config.MAX_TTS_QUEUE)
    stop = threading.Event()

    ocr = FakeOcr(texts, ocr_latency)
    tts = FakeTts()
    player = FakePlayer(tts, play_seconds)

    started = time.monotonic()
    threads = [
        CaptureWorker(shared, bus, buffer, stop, capture_factory=lambda: FakeScreen(rois, timeline, started)),
        OcrWorker(shared, bus, buffer, speech_queue, stop, ocr),
        TtsPlaybackWorker(shared, bus, speech_queue, stop, tts, player),
    ]
    for thread in threads:
        thread.start()

    deadline = started + duration
    reached_at: Optional[float] = None
    while time.monotonic() < deadline:
        time.sleep(0.05)
        if expected is not None and len(player.events) >= expected:
            reached_at = reached_at or time.monotonic()
            if time.monotonic() - reached_at > 1.5:   # ждём ещё: вдруг появятся лишние повторы
                break

    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)
    return player, ocr, buffer
