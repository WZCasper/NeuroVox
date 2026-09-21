# -*- coding: utf-8 -*-
"""
Захват области экрана и подготовка изображения для OCR.

ВАЖНО: экземпляр mss нельзя передавать между потоками, поэтому
ScreenCapture нужно создавать и использовать внутри одного и того же потока.
"""

import logging
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

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


def preprocess_for_ocr(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Готовит кадр к распознаванию: увеличение, серый цвет, бинаризация.

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

    # Метод Оцу подбирает порог автоматически под яркость сцены.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Субтитры — обычно светлый текст на тёмном фоне. После Оцу светлые пиксели = 255.
    # Определяем, что является текстом: меньшая по площади часть изображения.
    white_ratio = float(np.count_nonzero(binary)) / binary.size
    if white_ratio > 0.5:
        # Текст тёмный на светлом фоне — инвертируем, чтобы «текст» стал белым.
        binary = cv2.bitwise_not(binary)
        white_ratio = 1.0 - white_ratio

    # Пустой или полностью «залитый» кадр — распознавать нечего.
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
