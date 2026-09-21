# -*- coding: utf-8 -*-
"""
Синтез речи (TTS) на базе Silero. Работает полностью офлайн.

Модель скачивается один раз при первом запуске в папку models/, после чего
интернет больше не нужен. Синтез выполняется на процессоре (CPU) — у видеокарт
AMD нет CUDA, поэтому GPU не используется.
"""

import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import config

logger = logging.getLogger("neurovox.tts")

ProgressCallback = Callable[[float, str], None]

# Silero ограничивает длину текста за один вызов; длинные реплики режем на части.
MAX_CHUNK_CHARS = 250

# Разделители предложений для нарезки длинных реплик.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Допустимые символы для Silero (кириллица, латиница, цифры, базовая пунктуация).
_ALLOWED = re.compile(r"[^А-Яа-яЁёA-Za-z0-9\s.,!?:;\-–—…'\"()+]")


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


class SileroTts:
    """Обёртка над моделью Silero: загрузка и превращение текста в звук."""

    def __init__(self, model_name: str = config.DEFAULT_TTS_MODEL) -> None:
        self.model_name = model_name
        self.sample_rate = config.TTS_SAMPLE_RATE
        self._model = None
        self._torch = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, progress: Optional[ProgressCallback] = None) -> None:
        """Скачивает (при необходимости) и загружает модель в память."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: WPS433 — ленивый импорт намеренный
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
            importer = torch.package.PackageImporter(str(path))
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

        if speaker not in self.available_speakers():
            speaker = config.DEFAULT_SPEAKER

        fragments: List[np.ndarray] = []
        for chunk in split_into_chunks(prepared):
            try:
                with self._torch.inference_mode():
                    audio = self._model.apply_tts(
                        text=chunk,
                        speaker=speaker,
                        sample_rate=self.sample_rate,
                        put_accent=True,
                        put_yo=True,
                    )
            except Exception as exc:  # noqa: BLE001
                raise TtsError(f"Ошибка синтеза речи: {exc}") from exc

            samples = audio.detach().cpu().numpy().astype(np.float32)
            fragments.append(change_speed(samples, speed))
        return fragments
