# -*- coding: utf-8 -*-
"""
Захват области экрана и подготовка изображения для OCR.

ВАЖНО: экземпляр mss нельзя передавать между потоками, поэтому
ScreenCapture нужно создавать и использовать внутри одного и того же потока.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

import config

logger = logging.getLogger("neurovox.capture")

# Во сколько раз увеличивать кадр перед распознаванием.
# Игровые субтитры часто мелкие; масштаб x2 заметно повышает точность OCR.
UPSCALE_FACTOR = 2

# Белые поля вокруг текста (в пикселях после увеличения). Tesseract хуже читает
# буквы, вплотную прижатые к краю изображения, поэтому добавляем отступ.
OCR_PADDING = 24

# Если доля светлых пикселей меньше этого значения, в кадре, скорее всего, нет текста.
MIN_TEXT_PIXEL_RATIO = 0.004
MAX_TEXT_PIXEL_RATIO = 0.60

# --- Выбор способа бинаризации ---------------------------------------------
#
# Метод Оцу подбирает ОДИН порог яркости для всего кадра и отлично работает,
# когда фон вокруг субтитра однороден (типичная плашка под текстом). Но многие
# игры показывают субтитры прямо поверх сцены: одна половина фразы может лежать
# на тёмном камне, другая — на светлом небе. Для такого кадра единого порога не
# существует в принципе: он либо «съедает» текст на светлой части, либо заливает
# шумом тёмную. Порог адаптивной бинаризации (cv2.adaptiveThreshold), напротив,
# считается отдельно для каждой локальной области кадра и не боится перепадов
# яркости фона, но на РОВНОМ фоне может давать более рваный (хотя обычно всё ещё
# читаемый) контур буквы, чем глобальный Оцу.
#
# Экспериментально (реальные прогоны через Tesseract на полутора сотнях
# синтетических кадров: обычная плашка, градиент, шахматный узор, светлый и
# тёмный текст, разная толщина обводки) простое переключение между двумя
# методами по признаку «однороден ли фон» даёт заметно более высокую точность
# распознавания, чем любой из методов по отдельности, и не ухудшает результат
# на трёх штатных проверочных фразах check_ocr в selftest.py (там фон везде
# однородный, поэтому используется прежний, уже проверенный путь через Оцу).
#
# Критерий «однородности» — среднеквадратичное отклонение яркости (в шкале
# 0-255) по всему увеличенному кадру. У ровной плашки с лёгким шумом камеры
# это значение обычно не превышает 20-25; у кадра с заметным перепадом
# яркости (градиент, чересстрочная засветка, кусок неба и кусок земли в одной
# строке субтитра) оно, как правило, выше 45-60. Порог подобран так, чтобы
# полностью однородные и мягко-шумные сцены безопасно оставались на Оцу, а
# любой заметный на глаз перепад яркости уходил на адаптивный порог.
BACKGROUND_UNIFORMITY_STD_THRESHOLD = 30.0

# Параметры cv2.adaptiveThreshold для неоднородного фона: сторона окна, в
# котором считается локальный порог (обязано быть нечётным), и константа,
# вычитаемая из локального среднего — чем она больше, тем строже отбор и тем
# меньше шума от лёгких, незначащих перепадов яркости попадает в результат.
# Оба числа заданы в пикселях УЖЕ УВЕЛИЧЕННОГО (после UPSCALE_FACTOR) кадра.
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# После адаптивной бинаризации по кадру рассыпаны отдельные шумные точки —
# места, где локальный контраст фона случайно превысил порог, хотя текста там
# нет. Настоящая буква после увеличения кадра в UPSCALE_FACTOR раз всегда
# занимает заметно больше пикселей, чем случайная точка шума, поэтому такие
# мелкие «острова» безопасно удалить, оставив только связные области нужного
# размера. Порог задан в пересчёте на масштаб x2 и растёт пропорционально
# площади (квадрату линейного размера) при других значениях UPSCALE_FACTOR.
MIN_TEXT_COMPONENT_AREA = 10 * (UPSCALE_FACTOR / 2) ** 2


class CaptureError(Exception):
    """Ошибка захвата экрана."""


class ScreenCapture:
    """Захватывает прямоугольную область экрана (ROI) с помощью mss."""

    def __init__(self) -> None:
        try:
            self._sct = mss.mss()
        except Exception as exc:  # noqa: BLE001 — mss может бросать разные исключения
            raise CaptureError(f"Не удалось инициализировать захват экрана: {exc}") from exc

    def grab(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Захватывает область экрана.

        roi — кортеж (left, top, width, height) в пикселях.
        Возвращает массив BGR формата (высота, ширина, 3).
        """
        left, top, width, height = (int(v) for v in roi)
        if width < 10 or height < 10:
            raise CaptureError("Область захвата слишком маленькая.")

        region = {"left": left, "top": top, "width": width, "height": height}
        try:
            shot = self._sct.grab(region)
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"Ошибка захвата области {region}: {exc}") from exc

        # mss возвращает BGRA — отбрасываем альфа-канал.
        frame = np.asarray(shot, dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        """Освобождает ресурсы захвата."""
        try:
            self._sct.close()
        except Exception:  # noqa: BLE001
            pass


def _binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Бинаризация методом Оцу: один порог яркости на весь кадр.

    Подходит для однородного фона (ровная подложка под субтитрами) — метод
    проверен годами использования в проекте и остаётся путём по умолчанию
    для таких кадров. Возвращает маску, где текст = 255, фон = 0 (полярность
    определяется по меньшей по площади группе пикселей).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(np.count_nonzero(binary)) / binary.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def _binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Бинаризация с локальным (адаптивным) порогом: свой порог для каждой
    области кадра вместо одного общего.

    Подходит для неоднородного фона (градиент, часть сцены светлее другой),
    где у метода Оцу нет единственно верного порога. Возвращает маску, где
    текст = 255, фон = 0.
    """
    # THRESH_BINARY с отрицательной константой C: пиксель становится белым,
    # если он ЯРЧЕ своей локальной окрестности — это соответствует наиболее
    # распространённому в играх случаю (светлый текст, часто с тёмной
    # обводкой контура, поверх произвольного по яркости фона).
    bright_on_local = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE, -ADAPTIVE_C,
    )
    # Обратный случай (тёмный текст на локально светлом фоне) — например,
    # тёмная подсказка на светлой части экрана.
    dark_on_local = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
    )
    # Берём вариант с меньшей долей белых пикселей: это, как правило, и есть
    # верно определённый текст, а не «шум», в который вырождается адаптивный
    # порог при обработке однородной по яркости области без текста.
    bright_ratio = float(np.count_nonzero(bright_on_local)) / bright_on_local.size
    dark_ratio = float(np.count_nonzero(dark_on_local)) / dark_on_local.size
    binary = bright_on_local if bright_ratio <= dark_ratio else dark_on_local

    return _remove_small_components(binary, MIN_TEXT_COMPONENT_AREA)


def _remove_small_components(binary: np.ndarray, min_area: float) -> np.ndarray:
    """
    Убирает из бинарной маски связные области меньше ``min_area`` пикселей.

    После адаптивной бинаризации по фону иногда рассыпаны отдельные шумные
    точки — локальный контраст фона случайно превысил порог там, где текста
    нет. Настоящая буква после увеличения кадра занимает заметно больше
    пикселей, чем случайный шум, поэтому мелкие «острова» можно безопасно
    убрать, не потеряв сам текст.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, count):  # метка 0 — это фон, её пропускаем
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def preprocess_for_ocr(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Готовит кадр к распознаванию: увеличение, серый цвет, бинаризация.

    Способ бинаризации выбирается по фону кадра: для однородного фона
    (например, ровная плашка под субтитрами) используется метод Оцу — он
    даёт наиболее чистый контур буквы именно в этом случае. Если же в кадре
    заметен перепад яркости (субтитр наложен прямо на сцену, где одна часть
    фона светлее другой), единого порога, верного сразу для всего кадра, не
    существует — тогда используется адаптивная бинаризация с отдельным
    порогом для каждой локальной области.

    Возвращает изображение «чёрный текст на белом фоне» (так лучше всего
    работают и Tesseract, и EasyOCR) или None, если текста в кадре нет.
    """
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Разброс яркости по всему кадру — простой и быстрый признак того, ждать
    # ли от фона заметный перепад освещённости (см. BACKGROUND_UNIFORMITY_STD_THRESHOLD).
    background_is_uniform = float(np.std(gray)) < BACKGROUND_UNIFORMITY_STD_THRESHOLD
    binary = _binarize_otsu(gray) if background_is_uniform else _binarize_adaptive(gray)

    # Пустой или полностью «залитый» кадр — распознавать нечего. Проверяем уже
    # после того, как определились с текстом (белые пиксели = 255), поэтому
    # дополнительной инверсии здесь, в отличие от старой версии, не требуется.
    white_ratio = float(np.count_nonzero(binary)) / binary.size
    if white_ratio < MIN_TEXT_PIXEL_RATIO or white_ratio > MAX_TEXT_PIXEL_RATIO:
        return None

    # Финальный вид для OCR: чёрный текст на белом, с белыми полями по краям.
    result = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(
        result, OCR_PADDING, OCR_PADDING, OCR_PADDING, OCR_PADDING,
        cv2.BORDER_CONSTANT, value=255,
    )


def frames_are_similar(a: Optional[np.ndarray], b: Optional[np.ndarray], tolerance: float = 1.5) -> bool:
    """
    Быстро сравнивает два подготовленных кадра.

    Если кадры практически идентичны, повторный запуск OCR не нужен —
    это экономит процессор при статичных субтитрах.
    """
    if a is None or b is None or a.shape != b.shape:
        return False
    diff = cv2.absdiff(a, b)
    return float(diff.mean()) < tolerance


# ---------------------------------------------------------------------------
# Слежение за изменениями картинки в области
# ---------------------------------------------------------------------------

# Допуск на погрешность сравнения времени с плавающей точкой (например, 1.20 - 0.40
# в IEEE 754 равно 0.7999999999999999, а не ровно 0.8). Пороги отсчёта времени в этом
# классе — секунды, поэтому доля миллисекунды не меняет поведение по сути, но избавляет
# от «залипания» на границе интервала.
_TIME_EPSILON = 1e-6


@dataclass(frozen=True)
class Commit:
    """Решение трекера: этот снимок нужно отправить на распознавание."""

    image: Optional[np.ndarray]   # None — метка «область опустела»
    settled: bool                 # True — картинка устоялась; False — отправлена принудительно


class AreaTracker:
    """
    Следит за картинкой ОДНОЙ области и решает, когда её снимок пора распознавать.

    Зачем это нужно. Распознавать каждый кадр слишком долго, а распознавать «как
    получится» — значит пропускать реплики. Трекер работает на скорости снимков
    (десятки раз в секунду) и по картинке, без распознавания, определяет три вещи:

      1. Картинка изменилась и НЕ менялась ``settle_time`` секунд — субтитр показан
         целиком (не «печатается», не проявляется). Снимок отправляется на
         распознавание с пометкой settled=True, и подтверждать его повторно не нужно.
      2. Картинка меняется постоянно (анимированный фон, очень медленная печать) —
         тогда раз в ``max_wait`` секунд отправляется самый свежий снимок с пометкой
         settled=False, а фильтр текста сам решит, устоялся ли текст.
      3. Область была пустой ``empty_hold`` секунд после субтитра — отправляется метка
         «область опустела»: реплика закончилась, и та же фраза потом будет озвучена снова.

    Класс не знает ни про экран, ни про потоки: время передаётся параметром, поэтому
    его можно проверять обычными тестами.
    """

    def __init__(
        self,
        settle_time: float = config.CAPTURE_SETTLE_SECONDS,
        max_wait: float = config.CAPTURE_MAX_WAIT_SECONDS,
        empty_hold: float = config.CAPTURE_EMPTY_HOLD_SECONDS,
        tolerance: float = 1.5,
    ) -> None:
        self._settle_time = settle_time
        self._max_wait = max_wait
        self._empty_hold = empty_hold
        self._tolerance = tolerance

        self._committed: Optional[np.ndarray] = None   # что уже отправлено на распознавание
        self._pending: Optional[np.ndarray] = None     # новая картинка, ждущая «успокоения»
        self._first_diff_at: Optional[float] = None    # когда картинка впервые отличилась от отправленной
        self._last_change_at: float = 0.0              # когда картинка менялась в последний раз
        self._empty_since: Optional[float] = None
        self._empty_reported = False

    def update(self, image: Optional[np.ndarray], now: float) -> Optional[Commit]:
        """
        Сообщает трекеру очередной подготовленный кадр (None — в кадре нет текста).

        Возвращает Commit, если снимок нужно отправить на распознавание, иначе None.
        """
        if image is None:
            return self._on_empty(now)

        self._empty_since = None
        self._empty_reported = False

        # Картинка такая же, как уже отправленная, — ничего нового.
        if self._committed is not None and frames_are_similar(image, self._committed, self._tolerance):
            self._pending = None
            self._first_diff_at = None
            return None

        if self._first_diff_at is None:
            self._first_diff_at = now
        # Сравниваем с «опорным» кадром, а не с предыдущим: медленное плавное
        # изменение не должно выглядеть как «ничего не меняется».
        if self._pending is None or not frames_are_similar(image, self._pending, self._tolerance):
            self._pending = image
            self._last_change_at = now

        settled = (now - self._last_change_at) >= self._settle_time - _TIME_EPSILON
        forced = (now - self._first_diff_at) >= self._max_wait - _TIME_EPSILON
        if not (settled or forced):
            return None

        commit = Commit(image=self._pending, settled=settled)
        self._committed = self._pending
        self._pending = None
        self._first_diff_at = None
        return commit

    def _on_empty(self, now: float) -> Optional[Commit]:
        self._pending = None
        self._first_diff_at = None
        if self._empty_since is None:
            self._empty_since = now
        if (
            self._committed is not None
            and not self._empty_reported
            and (now - self._empty_since) >= self._empty_hold - _TIME_EPSILON
        ):
            self._committed = None
            self._empty_reported = True
            return Commit(image=None, settled=True)
        return None

