# -*- coding: utf-8 -*-
"""
Подготовка текста к озвучке: словарь ударений и нормализация заглавных фраз.

Автоматические ударения у Silero (особенно в модели v5) очень хороши, но имена героев,
названия мест и игровые термины не знает ни одна модель. Для таких слов есть
пользовательский словарь — обычный текстовый файл, который можно править в «Блокноте».

Формат словаря (по одному слову на строку):

    геральт = гер+альт
    скиппи  = ск+иппи

Знак «+» ставится ПЕРЕД ударной гласной — так Silero обозначает ударение.
Регистр букв при поиске не учитывается, а «е» и «ё» — разные буквы («все» и «всё»
это разные слова). Строки, начинающиеся с «#», — комментарии. Файл перечитывается
автоматически, как только изменился, поэтому программу перезапускать не нужно.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger("neurovox.pronunciation")

DICT_TEMPLATE = """# Словарь ударений NeuroVox
#
# Сюда можно добавить слова, которые озвучка произносит неверно: имена героев,
# названия мест, игровые термины.
#
# Как пользоваться:
#   1. Впишите слово, знак «=» и то же слово с ударением, по одному слову на строку.
#   2. Ударение обозначается знаком «+» ПЕРЕД ударной гласной: з+амок или зам+ок.
#   3. Сохраните файл (Ctrl+S). Перезапускать программу не нужно: изменения
#      применяются сами при следующей озвученной фразе.
#
# Регистр букв значения не имеет, а «е» и «ё» — разные буквы: «все» и «всё» это разные
# слова. Если нужны оба написания, добавьте две строки.
# Строки, которые начинаются с «#», — комментарии, программа их пропускает.
#
# Примеры (чтобы включить, уберите «#» в начале строки):
# геральт = гер+альт
# скиппи = ск+иппи

"""

# Символ, который считается частью слова. Дефис и «+» входят в него, чтобы слово из
# словаря не находилось внутри «кто-то» или внутри уже расставленного «з+амок».
_WORD_CHAR = r"[\w+\-]"

_CYR_LETTER = re.compile(r"[А-Яа-яЁё]")
_CYR_WORD = re.compile(r"[А-Яа-яЁё]+")
_SENTENCE_START = re.compile(r"(^|[.!?…]\s+)([а-яё])")


# ---------------------------------------------------------------------------
# Заглавные фразы
# ---------------------------------------------------------------------------

def normalize_caps(text: str) -> str:
    """
    Приводит фразу, набранную ЗАГЛАВНЫМИ буквами, к обычному написанию.

    Во многих играх субтитры печатаются капсом. Модели озвучки обучены на обычном
    тексте: слова заглавными буквами они произносят хуже (ударения ставятся
    неуверенно, часть слов может читаться по буквам). Поэтому, если во фразе нет
    ни одной строчной русской буквы, слова переводятся в строчные, а первая буква
    предложения делается заглавной.

    Не трогаем короткие аббревиатуры («ФБР», «США», «ГГ»): фраза приводится к обычному
    виду, только если в ней есть слово из четырёх и более букв. Латиницу не меняем.
    """
    letters = _CYR_LETTER.findall(text)
    if not letters or any(ch.islower() for ch in letters):
        return text
    words = _CYR_WORD.findall(text)
    if max(len(word) for word in words) < 4:
        return text

    lowered = _CYR_WORD.sub(lambda match: match.group(0).lower(), text)
    return _SENTENCE_START.sub(lambda match: match.group(1) + match.group(2).upper(), lowered)


# ---------------------------------------------------------------------------
# Словарь ударений
# ---------------------------------------------------------------------------

def _normalize_key(word: str) -> str:
    """Ключ для поиска: строчные буквы, одиночные пробелы, без знака «+»."""
    key = word.lower().replace("+", "")
    return re.sub(r"\s+", " ", key).strip()


def parse_dictionary(text: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Разбирает текст словаря.

    Возвращает (записи, замечания): записи — {ключ для поиска: слово с ударением};
    замечания — понятные человеку сообщения о строках, которые пришлось пропустить.
    """
    entries: Dict[str, str] = {}
    problems: List[str] = []
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            problems.append(f"строка {number}: нет знака «=» — «{line}»")
            continue
        left, right = (part.strip() for part in line.split("=", 1))
        key = _normalize_key(left)
        if not key or not right:
            problems.append(f"строка {number}: пустое слово слева или справа от «=» — «{line}»")
            continue
        entries[key] = right
    return entries, problems


def _key_to_pattern(key: str) -> str:
    """Регулярное выражение для ключа; пробел в ключе находит любые пробельные символы."""
    return "".join(r"\s+" if char == " " else re.escape(char) for char in key)


def _match_case(source: str, replacement: str) -> str:
    """
    Подгоняет регистр замены под регистр найденного в тексте слова.

    Три случая: слово было целиком заглавным (кроме однобуквенных — не путаем с
    обычной заглавной первой буквой) — заменa тоже становится заглавной; слово
    начиналось с заглавной буквы — заглавной становится только первая буква
    замены; иначе — замена остаётся такой, как в словаре.
    """
    letters = [ch for ch in source if ch.isalpha()]
    if len(letters) > 1 and all(ch.isupper() for ch in letters):
        return replacement.upper()
    if not source[:1].isupper():
        return replacement
    for index, char in enumerate(replacement):
        if char.isalpha():
            return replacement[:index] + char.upper() + replacement[index + 1:]
    return replacement


class StressDictionary:
    """Пользовательский словарь ударений с автоматической перезагрузкой файла."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._explicit_path = path
        self._lock = threading.Lock()
        self._signature: Optional[Tuple[int, int]] = None   # (время изменения, размер) файла
        self._entries: Dict[str, str] = {}
        self._pattern: Optional["re.Pattern[str]"] = None

    @property
    def path(self) -> Path:
        # Путь берём из config при каждом обращении — так его можно подменить в тестах.
        return self._explicit_path or config.STRESS_DICT_FILE

    def ensure_file(self) -> Path:
        """Создаёт файл словаря с инструкцией, если его ещё нет. Возвращает путь к файлу."""
        path = self.path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # utf-8-sig: со «скрытой меткой» кодировки старый «Блокнот» тоже не путает русские буквы.
            path.write_text(DICT_TEMPLATE, encoding="utf-8-sig")
        return path

    def __len__(self) -> int:
        self._reload_if_changed()
        return len(self._entries)

    def apply(self, text: str) -> str:
        """Заменяет слова из словаря на их варианты с ударением."""
        self._reload_if_changed()
        pattern = self._pattern
        if pattern is None or not text:
            return text
        entries = self._entries

        def replace(match: "re.Match[str]") -> str:
            replacement = entries.get(_normalize_key(match.group(0)))
            if replacement is None:
                return match.group(0)
            return _match_case(match.group(0), replacement)

        return pattern.sub(replace, text)

    # -- внутреннее ---------------------------------------------------------

    def _reload_if_changed(self) -> None:
        try:
            info = os.stat(self.path)
            signature: Optional[Tuple[int, int]] = (info.st_mtime_ns, info.st_size)
        except OSError:
            signature = None          # файла нет — словарь пуст

        with self._lock:
            if signature == self._signature:
                return
            self._signature = signature
            if signature is None:
                self._entries, self._pattern = {}, None
                return
            try:
                text = self.path.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Не удалось прочитать словарь ударений (%s): %s", self.path, exc)
                self._entries, self._pattern = {}, None
                return

            entries, problems = parse_dictionary(text)
            for problem in problems:
                logger.warning("Словарь ударений, %s", problem)
            self._entries = entries
            if entries:
                # Длинные ключи — первыми, чтобы «серый волк» находился раньше, чем «волк».
                keys = sorted(entries, key=len, reverse=True)
                alternatives = "|".join(_key_to_pattern(key) for key in keys)
                self._pattern = re.compile(
                    rf"(?<!{_WORD_CHAR})(?:{alternatives})(?!{_WORD_CHAR})", re.IGNORECASE
                )
            else:
                self._pattern = None
            logger.info("Словарь ударений загружен: %d слов.", len(entries))
