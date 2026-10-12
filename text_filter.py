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
from pronunciation import SpeakerNameList

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
# Одно «слово» из одних только букв, длиннее этого — почти наверняка не
# настоящее слово, а несколько слов субтитра, которые OCR слепил без
# пробела («почемубьтебененрисестьменадолго»). Самые длинные настоящие
# русские слова («человеконенавистничество», «высокопревосходительство»)
# укладываются в 25-26 букв — порог взят заметно выше, с запасом, чтобы не
# резать редкие, но настоящие длинные слова (топонимы, термины), и ловит
# именно слипшиеся фразы, которые на практике намного длиннее любого
# отдельного слова.
_MAX_SINGLE_WORD_LETTERS = 28
# Типичная ошибка OCR: латинские «двойники» внутри русских слов.
_LATIN_TO_CYR = str.maketrans(
    {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
     "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
     "O": "О", "P": "Р", "T": "Т", "X": "Х"}
)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")

# Имя говорящего в начале реплики: «Хозия: Несите его сюда!» или «Датч — Ты нам
# нужен сильным.» — 1-2 слова с заглавной буквы (имя, и, возможно, фамилия или
# прозвище), сразу (без пробела перед ним) двоеточие или тире, затем пробел и
# сам текст реплики. Длина слова (2-15 букв) отсекает как однобуквенные
# артефакты OCR, так и маловероятно длинные «имена» (которые на деле почти
# наверняка не имя, а начало обычного предложения).
_SPEAKER_PREFIX = re.compile(
    r"^([A-ZА-ЯЁ][a-zа-яё]{1,14}(?:\s[A-ZА-ЯЁ][a-zа-яё]{1,14})?)\s*[:—-]\s+(?=\S)"
)


def strip_speaker_name(text: str, mode: str, names: Optional[SpeakerNameList] = None) -> str:
    """
    Убирает имя говорящего из начала реплики, если оно там есть.

    mode:
      "off"  — текст не трогается (по умолчанию).
      "auto" — убирается ЛЮБОЙ текст, похожий на имя по форме (см.
               _SPEAKER_PREFIX): 1-2 слова с заглавной буквы перед «:» или
               «—»/«-» в начале строки. Это не всегда верно отличает имя от
               обычного вводного слова фразы («Запомни: ...») — пользователь
               сам выбирает этот режим, зная о таком риске (см. config.py,
               где объясняется разница между "auto" и "list").
      "list" — убирается, только если текст перед разделителем совпадает (без
               учёта регистра) с одним из имён в ``names``. Требует ``names``;
               без него (или с пустым списком) ничего не убирается.

    Если текст не начинается с подходящего по форме префикса вообще, возвращает
    его без изменений, независимо от режима.
    """
    if mode == "off" or not text:
        return text
    match = _SPEAKER_PREFIX.match(text)
    if not match:
        return text
    candidate = match.group(1)
    if mode == "list":
        if names is None or not names.contains(candidate):
            return text
    elif mode != "auto":
        return text
    return text[match.end():]


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


def _looks_like_ocr_garbage(text: str) -> bool:
    """
    Отличает бессвязный мусор распознавания от настоящей фразы субтитра.

    OCR на нечётком или занятом сценой кадре иногда не просто ошибается в
    отдельных буквах, а склеивает несколько слов субтитра в одно без
    пробела («почемубьтебененрисестьменадолго») или вперемешку с
    единичными буквами и цифрами («ree . И бласть ин: рай BR 4»). И то, и
    другое Silero озвучит как бессмысленный набор звуков — лучше
    промолчать на таком кадре и дождаться, пока субтитр устоится и
    распознается чище, чем произнести эту кашу вслух.
    """
    words = _WORD.findall(text)
    if not words:
        return False
    return any(len(word) > _MAX_SINGLE_WORD_LETTERS for word in words)


def clean_text(
    raw: str,
    speaker_name_mode: str = "off",
    speaker_names: Optional[SpeakerNameList] = None,
) -> str:
    """
    Очищает распознанную строку. Возвращает пустую строку, если речи нет.

    speaker_name_mode и speaker_names — см. strip_speaker_name ниже; по
    умолчанию отсечение имени выключено ("off"), поэтому вызовы без этих
    аргументов ведут себя как раньше.
    """
    if not raw:
        return ""

    text = raw.replace("\n", " ").replace("\r", " ")
    text = _JUNK_CHARS.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _fix_mixed_alphabet(text)
    text = _SPACES.sub(" ", text).strip()
    text = _EDGE_JUNK.sub("", text).strip()
    # Имя говорящего убирается ДО остальных проверок длины/мусора ниже, чтобы
    # короткая реплика после имени («Датч: Нет.») не отбраковывалась как
    # слишком короткая из-за того, что само имя считалось частью длины.
    text = strip_speaker_name(text, speaker_name_mode, speaker_names)
    text = text.strip()

    if len(text) < config.MIN_TEXT_LENGTH:
        return ""
    if _NO_LETTERS.match(text):
        return ""
    if _looks_like_ocr_garbage(text):
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
        speaker_name_mode: str = "off",
        speaker_names: Optional[SpeakerNameList] = None,
    ) -> None:
        self.threshold = threshold
        self.stable_frames = max(1, stable_frames)
        # См. strip_speaker_name/clean_text — режим и список можно менять «на
        # лету» (как и threshold выше): OcrWorker.run перечитывает настройки
        # каждый цикл и присваивает их атрибутам существующего фильтра, а не
        # создаёт SubtitleFilter заново.
        self.speaker_name_mode = speaker_name_mode
        self.speaker_names = speaker_names
        self._last_spoken: str = ""
        self._candidate: str = ""
        self._candidate_count: int = 0

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

    def forget(self) -> None:
        """
        Реплика закончилась (область субтитров опустела).

        Забываем последнюю озвученную фразу: если такая же появится снова —
        это новая реплика, и её нужно озвучить.
        """
        self.reset()

    def process(self, raw_text: str, settled: bool = False) -> Optional[str]:
        """
        Обрабатывает распознанную строку с устоявшегося (непустого) снимка.

        О том, что область субтитров опустела — реплика закончилась, —
        вызывающий код сообщает отдельным вызовом forget(), а не пустой
        строкой здесь: в текущем конвейере (workers.OcrWorker) распознавание
        вообще не запускается для пустых снимков, туда приходит только текст,
        для которого OCR что-то нашёл. process() с пустым raw_text по-прежнему
        безопасен (сразу вернёт None, не уронит остальную логику), но
        считать опустевшую область он больше не пытается — за это отвечает
        forget().

        settled=True означает, что снимок уже признан устоявшимся по картинке
        (она не менялась заметное время), поэтому ждать подтверждения текста
        следующими кадрами не нужно. При settled=False действует прежнее правило:
        текст должен повториться ``stable_frames`` раз подряд.

        Возвращает очищенный текст, если его нужно озвучить, иначе None.
        """
        text = clean_text(raw_text, self.speaker_name_mode, self.speaker_names)
        if not text:
            self._candidate = ""
            self._candidate_count = 0
            return None

        # Ждём, пока текст «устоится» (перестанет меняться между кадрами).
        if settled:
            # Устоявшуюся по картинке реплику подтверждать повторно не нужно.
            self._candidate = text
            self._candidate_count = self.stable_frames
        elif self._candidate and self._is_same_growing_phrase(text, self._candidate):
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
