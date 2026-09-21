# -*- coding: utf-8 -*-
"""
Движки оптического распознавания текста (OCR).

Поддерживаются два движка с единым интерфейсом:
  * EasyOCR   — работает «из коробки» через pip, только на CPU (без CUDA);
  * Tesseract — быстрее и легче, но требует установленный tesseract.exe.

Тяжёлые библиотеки импортируются лениво (при первом использовании),
чтобы окно программы открывалось мгновенно.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

import config

logger = logging.getLogger("neurovox.ocr")

# Отбрасываем результаты EasyOCR, в которых он не уверен.
EASYOCR_MIN_CONFIDENCE = 0.30


class OcrError(Exception):
    """Ошибка инициализации или работы OCR-движка."""


class BaseOcr(ABC):
    """Общий интерфейс OCR-движка."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Подготавливает движок к работе (загрузка моделей и т.п.)."""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> str:
        """Распознаёт текст на изображении и возвращает одну строку."""


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

class EasyOcrEngine(BaseOcr):
    """Распознавание через EasyOCR. Принудительно на CPU (gpu=False)."""

    name = "easyocr"

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        self._languages = languages or list(config.OCR_LANGUAGES)
        self._reader = None

    def load(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr  # noqa: WPS433 — ленивый импорт намеренный
        except ImportError as exc:
            raise OcrError(
                "Библиотека EasyOCR не установлена. Выполните: pip install easyocr"
            ) from exc

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = config.MODELS_DIR / "easyocr"
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Загрузка модели EasyOCR (первый запуск скачает файлы, это займёт время)...")
        try:
            self._reader = easyocr.Reader(
                self._languages,
                gpu=False,  # У AMD Radeon нет CUDA — работаем на процессоре.
                model_storage_directory=str(model_dir),
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise OcrError(
                "Не удалось загрузить модель EasyOCR. Проверьте подключение к интернету "
                f"при первом запуске. Подробности: {exc}"
            ) from exc
        logger.info("EasyOCR готов к работе.")

    def recognize(self, image: np.ndarray) -> str:
        if self._reader is None:
            raise OcrError("Движок EasyOCR не инициализирован.")
        try:
            # paragraph=False, чтобы получить уверенность по каждому фрагменту.
            results = self._reader.readtext(image, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"Ошибка распознавания EasyOCR: {exc}") from exc

        # Сортируем фрагменты слева направо и сверху вниз, чтобы порядок слов был верным.
        fragments = []
        for box, text, conf in results:
            if conf < EASYOCR_MIN_CONFIDENCE:
                continue
            top = min(point[1] for point in box)
            left = min(point[0] for point in box)
            fragments.append((round(top / 20), left, text))
        fragments.sort(key=lambda item: (item[0], item[1]))
        return " ".join(item[2] for item in fragments).strip()


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

class TesseractEngine(BaseOcr):
    """Распознавание через Tesseract (требует установленный tesseract.exe)."""

    name = "tesseract"

    def __init__(self, tesseract_path: str = "") -> None:
        self._custom_path = tesseract_path
        self._pytesseract = None

    def load(self) -> None:
        if self._pytesseract is not None:
            return
        try:
            import pytesseract  # noqa: WPS433
        except ImportError as exc:
            raise OcrError(
                "Библиотека pytesseract не установлена. Выполните: pip install pytesseract"
            ) from exc

        exe = config.find_tesseract(self._custom_path)
        if not exe:
            raise OcrError(
                "Программа Tesseract не найдена. Установите её "
                "(https://github.com/UB-Mannheim/tesseract/wiki), отметив русский язык, "
                "либо переключитесь на движок EasyOCR."
            )
        pytesseract.pytesseract.tesseract_cmd = exe

        try:
            languages = pytesseract.get_languages(config="")
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"Не удалось запустить Tesseract ({exe}): {exc}") from exc
        if "rus" not in languages:
            raise OcrError(
                "В Tesseract не установлен русский язык (rus.traineddata). "
                "Переустановите Tesseract и отметьте «Russian» в списке языков."
            )

        self._pytesseract = pytesseract
        logger.info("Tesseract готов к работе: %s", exe)

    def recognize(self, image: np.ndarray) -> str:
        if self._pytesseract is None:
            raise OcrError("Движок Tesseract не инициализирован.")
        try:
            # psm 6 — единый блок текста (подходит и для одно-, и для двухстрочных субтитров).
            text = self._pytesseract.image_to_string(
                image, lang="rus+eng", config="--oem 1 --psm 6"
            )
        except Exception as exc:  # noqa: BLE001
            raise OcrError(f"Ошибка распознавания Tesseract: {exc}") from exc
        return " ".join(text.split())


# ---------------------------------------------------------------------------
# Фабрика
# ---------------------------------------------------------------------------

def create_engine(engine_name: str, tesseract_path: str = "") -> BaseOcr:
    """Создаёт OCR-движок по имени. Модели загружаются отдельным вызовом load()."""
    if engine_name == "tesseract":
        return TesseractEngine(tesseract_path)
    if engine_name == "easyocr":
        return EasyOcrEngine()
    raise OcrError(f"Неизвестный OCR-движок: {engine_name}")


def create_with_fallback(
    engine_name: str,
    tesseract_path: str = "",
    on_message: Optional[Callable[[str], None]] = None,
) -> BaseOcr:
    """
    Создаёт и загружает выбранный движок. Если он недоступен — пробует запасной.

    on_message — функция для вывода сообщений пользователю (например, в лог GUI).
    """
    notify = on_message or (lambda _msg: None)
    order = [engine_name] + [n for n in config.OCR_ENGINES if n != engine_name]

    last_error: Optional[OcrError] = None
    for name in order:
        engine = create_engine(name, tesseract_path)
        try:
            engine.load()
        except OcrError as exc:
            last_error = exc
            logger.warning("Движок %s недоступен: %s", name, exc)
            notify(f"Движок «{name}» недоступен: {exc}")
            continue
        if name != engine_name:
            notify(f"Переключено на запасной OCR-движок: {name}")
        return engine

    raise OcrError(f"Ни один OCR-движок не удалось запустить. {last_error}")
