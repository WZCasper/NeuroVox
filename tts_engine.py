# -*- coding: utf-8 -*-
"""
Синтез речи (TTS) на базе Silero. Работает полностью офлайн.

Модель скачивается один раз при первом запуске в папку models/, после чего
интернет больше не нужен. Синтез выполняется на процессоре (CPU) — у видеокарт
AMD нет CUDA, поэтому GPU не используется.
"""

import inspect
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import config
from pronunciation import StressDictionary, normalize_caps

logger = logging.getLogger("neurovox.tts")

ProgressCallback = Callable[[float, str], None]

# Silero ограничивает длину текста за один вызов; длинные реплики режем на части.
MAX_CHUNK_CHARS = 250

# Разделители предложений для нарезки длинных реплик.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Допустимые символы для Silero (кириллица, латиница, цифры, базовая пунктуация).
_ALLOWED = re.compile(r"[^А-Яа-яЁёA-Za-z0-9\s.,!?:;\-–—…'\"()+]")


# Необязательные параметры apply_tts (из официального примера Silero для моделей v5):
#   put_accent       — автоматически ставить ударения;
#   put_yo           — восстанавливать букву «ё»;
#   put_stress_homo  — выбирать ударение у омографов по смыслу (замОк / зАмок);
#   put_yo_homo      — то же для «ё» у омографов (все / всё).
# У разных версий модели набор параметров отличается (в v4_ru омографов нет), поэтому
# передаём только те, которые модель действительно принимает.
_OPTIONAL_TTS_KWARGS = ("put_accent", "put_yo", "put_stress_homo", "put_yo_homo")

# Python сообщает имя лишнего параметра в тексте ошибки: unexpected keyword argument 'имя'.
_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '(\w+)'")


def detect_optional_kwargs(model) -> tuple:
    """Определяет, какие из необязательных параметров умеет принимать model.apply_tts."""
    try:
        params = inspect.signature(model.apply_tts).parameters
    except (AttributeError, TypeError, ValueError):
        return _OPTIONAL_TTS_KWARGS
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return _OPTIONAL_TTS_KWARGS
    return tuple(name for name in _OPTIONAL_TTS_KWARGS if name in params)


class TtsError(Exception):
    """Ошибка загрузки модели или синтеза речи."""


def model_path(model_name: str) -> Path:
    """Путь к файлу модели на диске."""
    return config.MODELS_DIR / f"{model_name}.pt"


def is_model_downloaded(model_name: str) -> bool:
    """Проверяет, что модель скачана и не повреждена (по размеру файла)."""
    path = model_path(model_name)
    return path.is_file() and path.stat().st_size >= config.MIN_MODEL_SIZE_BYTES


def download_model(model_name: str, progress: Optional[ProgressCallback] = None) -> Path:
    """
    Скачивает модель Silero, если её ещё нет на диске.

    Запись идёт во временный файл, который переименовывается только после
    полной загрузки — обрыв связи не оставит «битую» модель.
    """
    notify = progress or (lambda _p, _m: None)

    if model_name not in config.SILERO_MODELS:
        raise TtsError(f"Неизвестная модель озвучки: {model_name}")

    target = model_path(model_name)
    if is_model_downloaded(model_name):
        return target

    if target.exists():
        logger.warning("Файл модели повреждён или неполный, скачиваем заново: %s", target)
        try:
            target.unlink()
        except OSError as exc:
            raise TtsError(f"Не удалось удалить повреждённый файл модели: {exc}") from exc

    url = config.SILERO_MODELS[model_name]
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TtsError(f"Не удалось создать папку для моделей ({config.MODELS_DIR}): {exc}") from exc

    logger.info("Скачивание модели озвучки %s ...", model_name)
    notify(0.0, f"Скачивание модели озвучки «{model_name}»...")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".part", dir=str(config.MODELS_DIR))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": f"{config.APP_NAME}/{config.APP_VERSION}"})
        with urllib.request.urlopen(request, timeout=30) as response, open(tmp_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_reported = -1
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent != last_reported:
                        last_reported = percent
                        notify(
                            downloaded / total,
                            f"Скачивание модели: {percent}% "
                            f"({downloaded // (1024 * 1024)} из {total // (1024 * 1024)} МБ)",
                        )

        size = tmp_path.stat().st_size
        if size < config.MIN_MODEL_SIZE_BYTES:
            raise TtsError(
                f"Скачанный файл слишком мал ({size} байт) — загрузка прервана или сервер вернул ошибку."
            )
        os.replace(tmp_path, target)
    except urllib.error.URLError as exc:
        raise TtsError(
            "Не удалось скачать модель озвучки. Проверьте подключение к интернету "
            f"(нужно только при первом запуске). Подробности: {exc.reason}"
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise TtsError(f"Ошибка при скачивании модели: {exc}") from exc
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    logger.info("Модель озвучки скачана: %s", target)
    notify(1.0, "Модель озвучки готова.")
    return target


def prepare_text(text: str) -> str:
    """Убирает символы, которые Silero не умеет произносить, и лишние пробелы."""
    cleaned = _ALLOWED.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_into_chunks(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Режет длинный текст на части по предложениям (для быстрого начала озвучки)."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        # Предложение длиннее лимита режем по словам.
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def change_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """
    Меняет скорость речи линейной интерполяцией.

    speed > 1 — быстрее, speed < 1 — медленнее. Тон при этом смещается
    незначительно в пределах допустимого диапазона 0.5–2.0, что для
    озвучки реплик приемлемо и не требует тяжёлых зависимостей.
    """
    if abs(speed - 1.0) < 0.02 or audio.size < 2:
        return audio
    new_length = max(2, int(round(audio.size / speed)))
    old_positions = np.linspace(0.0, 1.0, num=audio.size)
    new_positions = np.linspace(0.0, 1.0, num=new_length)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def _cache_key(text: str, speaker: str, speed: float) -> Tuple[str, str, float]:
    """
    Ключ кэша синтеза.

    ``speed`` округляется до сотых: слайдер скорости в интерфейсе меняется
    дискретными шагами, а число с плавающей точкой из настроек (после чтения
    из JSON, копирования и т. п.) может отличаться на неразличимую на слух
    долю — без округления это давало бы «непопадание» в кэш там, где по
    смыслу должно быть попадание.
    """
    return (text, speaker, round(speed, 2))


class _SynthesisCache:
    """
    LRU-кэш готового аудио синтеза речи.

    Ограничение — не число запомненных фраз, а суммарная ДЛИТЕЛЬНОСТЬ
    аудио в кэше (в секундах, см. config.TTS_CACHE_MAX_SECONDS): реплики
    сильно различаются по длине, и лимит «столько-то штук» плохо предсказывает
    реальное потребление памяти. Вытесняются наименее давно запрошенные записи.

    Класс не знает про текст, голос и модель — только про пары «ключ, список
    аудиофрагментов» и объём, который они занимают. Реализован отдельно от
    SileroTts, чтобы логику вытеснения можно было проверить в тестах без
    какой-либо модели.
    """

    def __init__(self, sample_rate: int, max_seconds: float = config.TTS_CACHE_MAX_SECONDS) -> None:
        self._sample_rate = max(1, sample_rate)
        self._max_samples = max(0, int(max_seconds * self._sample_rate))
        self._entries: "OrderedDict[tuple, List[np.ndarray]]" = OrderedDict()
        self._total_samples = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: tuple) -> Optional[List[np.ndarray]]:
        """Возвращает копию закэшированных фрагментов или None, если их нет."""
        fragments = self._entries.get(key)
        if fragments is None:
            return None
        # Запрошенную запись отмечаем как «свежую» (LRU).
        self._entries.move_to_end(key)
        # Копии, а не ссылки на закэшированные массивы: воспроизведение не должно
        # иметь возможности случайно изменить данные, отданные при следующем запросе.
        return [fragment.copy() for fragment in fragments]

    def put(self, key: tuple, fragments: List[np.ndarray]) -> None:
        """Запоминает фрагменты, вытесняя наименее давно запрошенные при нехватке места."""
        if not fragments or self._max_samples == 0:
            return
        size = sum(fragment.size for fragment in fragments)
        # Одна фраза длиннее всего лимита кэша целиком — например, очень длинный
        # текст. Хранить её всё равно бессмысленно: она одна вытеснит всё
        # остальное, а повторно запрошена (весь текст целиком, слово в слово)
        # будет, скорее всего, нескоро.
        if size > self._max_samples:
            return
        if key in self._entries:
            self._total_samples -= sum(f.size for f in self._entries[key])
            del self._entries[key]
        while self._entries and self._total_samples + size > self._max_samples:
            _, evicted = self._entries.popitem(last=False)  # last=False — самая старая запись
            self._total_samples -= sum(f.size for f in evicted)
        self._entries[key] = [fragment.copy() for fragment in fragments]
        self._total_samples += size

    def clear(self) -> None:
        """Полностью очищает кэш (например, при смене модели озвучки)."""
        self._entries.clear()
        self._total_samples = 0


class SileroTts:
    """Обёртка над моделью Silero: загрузка и превращение текста в звук."""

    def __init__(self, model_name: str = config.DEFAULT_TTS_MODEL) -> None:
        self.model_name = model_name
        self.sample_rate = config.TTS_SAMPLE_RATE
        self._model = None
        self._torch = None
        self._optional_kwargs: tuple = _OPTIONAL_TTS_KWARGS
        self._dictionary = StressDictionary()
        self._cache = _SynthesisCache(self.sample_rate)
        # Имя модели, для которой сейчас актуален кэш. Отдельно от model_name:
        # смена self.model_name без повторной загрузки не должна сама по себе
        # опустошать кэш — тот остаётся верным для уже загруженной модели, пока
        # не загружена другая.
        self._cache_model_name: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress: Optional[ProgressCallback] = None) -> None:
        """Скачивает (при необходимости) и загружает модель в память."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: WPS433 — ленивый импорт намеренный
            from torch import package as torch_package
        except ImportError as exc:
            raise TtsError(
                "Библиотека PyTorch не установлена. Выполните: pip install torch"
            ) from exc
        self._torch = torch

        path = download_model(self.model_name, progress)

        # Ограничиваем потоки, чтобы озвучка не «съедала» процессор игры целиком.
        cpu_count = os.cpu_count() or 4
        torch.set_num_threads(max(2, min(4, cpu_count // 2)))

        try:
            importer = torch_package.PackageImporter(str(path))
            model = importer.load_pickle("tts_models", "model")
            model.to(torch.device("cpu"))
        except Exception as exc:  # noqa: BLE001
            # Файл, скорее всего, повреждён — удаляем, чтобы при следующем запуске скачать заново.
            try:
                path.unlink()
            except OSError:
                pass
            raise TtsError(
                "Не удалось загрузить модель озвучки (файл повреждён и будет скачан заново "
                f"при следующем запуске). Подробности: {exc}"
            ) from exc

        self._model = model
        self._optional_kwargs = detect_optional_kwargs(model)
        # Кэш хранит аудио конкретной модели: та же фраза, синтезированная
        # другой моделью, звучит по-другому, и отдавать старую запись из кэша
        # было бы неверно. Сбрасываем, только если модель действительно сменилась
        # (а не при повторной загрузке той же самой — тогда кэш остаётся полезен).
        if self._cache_model_name != self.model_name:
            self._cache.clear()
            self._cache_model_name = self.model_name
        logger.info("Модель озвучки %s загружена (CPU).", self.model_name)

    def available_speakers(self) -> List[str]:
        """Список голосов, реально присутствующих в загруженной модели."""
        if self._model is None:
            return list(config.SILERO_SPEAKERS)
        try:
            return [s for s in self._model.speakers if s in config.SILERO_SPEAKERS] or list(
                config.SILERO_SPEAKERS
            )
        except AttributeError:
            return list(config.SILERO_SPEAKERS)

    def _apply_tts(self, chunk: str, speaker: str):
        """
        Вызывает model.apply_tts с необязательными параметрами.

        Если модель не принимает какой-то параметр, отказываемся именно от него, а не
        от всех сразу: например, v4_ru не знает про омографы, но ударения ставит.
        """
        base = {"text": chunk, "speaker": speaker, "sample_rate": self.sample_rate}
        extra = {name: True for name in self._optional_kwargs}
        while True:
            try:
                return self._model.apply_tts(**base, **extra)
            except TypeError as exc:
                if not extra:
                    raise
                match = _UNEXPECTED_KWARG.search(str(exc))
                rejected = match.group(1) if match else None
                if rejected in extra:
                    del extra[rejected]
                    logger.info("Модель не принимает параметр %s — синтез выполняется без него.", rejected)
                else:
                    logger.warning(
                        "Модель не приняла параметры %s — синтез выполняется без них.", ", ".join(extra)
                    )
                    extra = {}
                self._optional_kwargs = tuple(extra)

    def synthesize(self, text: str, speaker: str, speed: float = 1.0) -> List[np.ndarray]:
        """
        Превращает текст в звук. Возвращает список звуковых фрагментов (float32),
        чтобы воспроизведение можно было начать, не дожидаясь конца синтеза.
        """
        if self._model is None or self._torch is None:
            raise TtsError("Модель озвучки не загружена.")

        prepared = prepare_text(text)
        if not prepared:
            return []
        # Фразы капсом приводим к обычному виду, затем применяем словарь ударений пользователя.
        prepared = self._dictionary.apply(normalize_caps(prepared))

        if speaker not in self.available_speakers():
            speaker = config.DEFAULT_SPEAKER

        # Результат синтеза детерминирован при одинаковых (итоговый текст,
        # голос, скорость) — повторная фраза (боевой выкрик, частая подсказка)
        # возвращается из кэша без обращения к модели.
        key = _cache_key(prepared, speaker, speed)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        fragments: List[np.ndarray] = []
        for chunk in split_into_chunks(prepared):
            try:
                with self._torch.inference_mode():
                    audio = self._apply_tts(chunk, speaker)
            except Exception as exc:  # noqa: BLE001
                raise TtsError(f"Ошибка синтеза речи: {exc}") from exc

            samples = audio.detach().cpu().numpy().astype(np.float32)
            fragments.append(change_speed(samples, speed))

        self._cache.put(key, fragments)
        return fragments
