# -*- coding: utf-8 -*-
"""
Конвейер фильтрации распознанного текста.

Этапы:
  1. Очистка от артефактов интерфейса и OCR-мусора (регулярные выражения).
  2. Проверка «устоявшегося» текста: строку озвучиваем, только когда она не
     меняется несколько кадров подряд (иначе озвучим недопечатанную фразу).
  3. Нечёткое сравнение с последней озвученной фразой: если похожесть выше
     порога (по умолчанию 85%), фраза считается повтором и игнорируется.
"""

import logging
import re
from typing import Optional

from rapidfuzz import fuzz

import config

logger = logging.getLogger("neurovox.filter")

# Символы, которые почти всегда являются артефактами интерфейса или ошибками OCR.
_JUNK_CHARS = re.compile(r"[\]\[|\\/<>{}~^_=*#@`©®™°•·▪■□▲►◄«»§¦]")
# Любые «серые» повторы знаков препинания: «...», «!!!», «---».
_REPEATED_PUNCT = re.compile(r"([.,!?;:\-–—])\1{2,}")
# Разрывы в словах вида «п р и в е т» не чиним (слишком рискованно), но схлопываем пробелы.
_SPACES = re.compile(r"\s+")
# Одиночные символы, оставшиеся по краям после удаления мусора.
_EDGE_JUNK = re.compile(r"^[\s.,;:!?\-–—'\"()]+|[\s,;:\-–—'\"()]+$")
# Строка, состоящая только из цифр и знаков (таймеры, счёт, HP и т.п.) — не речь.
_NO_LETTERS = re.compile(r"^[^A-Za-zА-Яа-яЁё]+$")
# Типичная ошибка OCR: латинские «двойники» внутри русских слов.
_LATIN_TO_CYR = str.maketrans(
    {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
     "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
     "O": "О", "P": "Р", "T": "Т", "X": "Х"}
)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def _fix_mixed_alphabet(text: str) -> str:
    """
    Исправляет слова, где русские буквы перемешаны с латинскими двойниками.

    Пример: «Пpивет» (латинская p) -> «Привет». Слова, целиком написанные
    латиницей, не трогаем — это может быть настоящий английский текст.
    """

    def repair(match: "re.Match[str]") -> str:
        word = match.group(0)
        if _CYRILLIC.search(word) and _LATIN.search(word):
            return word.translate(_LATIN_TO_CYR)
        return word

    return _WORD.sub(repair, text)


def clean_text(raw: str) -> str:
    """Очищает распознанную строку. Возвращает пустую строку, если речи нет."""
    if not raw:
        return ""

    text = raw.replace("\n", " ").replace("\r", " ")
    text = _JUNK_CHARS.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _fix_mixed_alphabet(text)
    text = _SPACES.sub(" ", text).strip()
    text = _EDGE_JUNK.sub("", text).strip()

    if len(text) < config.MIN_TEXT_LENGTH:
        return ""
    if _NO_LETTERS.match(text):
        return ""
    return text


def normalize_for_compare(text: str) -> str:
    """Приводит текст к виду, удобному для сравнения (регистр, ё, знаки препинания)."""
    lowered = text.lower().replace("ё", "е")
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return _SPACES.sub(" ", lowered).strip()


def similarity(a: str, b: str, strict: bool = True) -> float:
    """
    Похожесть двух строк в процентах (0–100).

    strict=True  — только посимвольное сходство. Нужен для проверки
                   «устоялся ли текст»: обрывок «Куда ты» НЕ должен считаться
                   той же фразой, что и «Куда ты пропал?».
    strict=False — дополнительно учитывает пересечение слов (устойчив к
                   перестановкам). Нужен, чтобы не озвучивать повтор фразы,
                   в которой OCR слегка исказил или переставил слова.
    """
    na, nb = normalize_for_compare(a), normalize_for_compare(b)
    if not na or not nb:
        return 0.0
    score = fuzz.ratio(na, nb)
    if not strict:
        # token_sort_ratio сравнивает слова независимо от порядка; в отличие от
        # token_set_ratio он не считает подмножество слов полным совпадением.
        score = max(score, fuzz.token_sort_ratio(na, nb))
    return float(score)


class SubtitleFilter:
    """
    Решает, нужно ли озвучивать очередную распознанную строку.

    Использование:
        f = SubtitleFilter(threshold=85)
        phrase = f.process(ocr_text)   # вернёт текст для озвучки или None
    """

    def __init__(
        self,
        threshold: int = config.SIMILARITY_THRESHOLD,
        stable_frames: int = config.STABLE_FRAMES_REQUIRED,
    ) -> None:
        self.threshold = threshold
        self.stable_frames = max(1, stable_frames)
        self._last_spoken: str = ""
        self._candidate: str = ""
        self._candidate_count: int = 0
        self._empty_frames: int = 0

    def _is_same_growing_phrase(self, new: str, old: str) -> bool:
        """
        Проверяет, что новая строка — это та же фраза (возможно, с шумом OCR).

        Строка, которая просто «выросла» из предыдущей (дописан хвост) —
        тоже не считается устоявшейся: она ещё печатается.
        """
        return similarity(new, old, strict=True) >= self.threshold

    def reset(self) -> None:
        """Полный сброс состояния (например, при перезапуске)."""
        self._last_spoken = ""
        self._candidate = ""
        self._candidate_count = 0
        self._empty_frames = 0

    def process(self, raw_text: str) -> Optional[str]:
        """
        Обрабатывает распознанную строку.

        Возвращает очищенный текст, если его нужно озвучить, иначе None.
        """
        text = clean_text(raw_text)

        # Пустой кадр: субтитры исчезли. Сбрасываем кандидата, но помним последнюю
        # озвученную фразу — если та же фраза вернётся, это уже новое появление.
        if not text:
            self._candidate = ""
            self._candidate_count = 0
            self._empty_frames += 1
            # Субтитры пропали на несколько кадров подряд — реплика закончилась.
            # Забываем её, чтобы такая же фраза при повторном появлении
            # (например, повторная реплика персонажа) была озвучена снова.
            if self._empty_frames >= config.PAUSE_FRAMES_TO_FORGET:
                self._last_spoken = ""
            return None
        self._empty_frames = 0

        # Ждём, пока текст «устоится» (перестанет меняться между кадрами).
        if self._candidate and self._is_same_growing_phrase(text, self._candidate):
            self._candidate_count += 1
            # Берём более длинную версию: она обычно полнее (дописанный хвост).
            if len(text) > len(self._candidate):
                self._candidate = text
        else:
            self._candidate = text
            self._candidate_count = 1

        if self._candidate_count < self.stable_frames:
            return None

        candidate = self._candidate

        # Повтор уже озвученной фразы — игнорируем.
        if self._last_spoken and similarity(candidate, self._last_spoken, strict=False) >= self.threshold:
            return None

        self._last_spoken = candidate
        return candidate
