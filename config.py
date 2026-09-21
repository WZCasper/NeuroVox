# -*- coding: utf-8 -*-
"""
Конфигурация NeuroVox.

Содержит пути к папкам, константы и настройки по умолчанию.
Пользовательские настройки сохраняются в файл settings.json и
восстанавливаются при следующем запуске.
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("neurovox.config")

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

APP_NAME = "NeuroVox"
APP_VERSION = "1.0.0"


def _base_dir() -> Path:
    """Возвращает папку приложения (учитывает запуск из .exe через PyInstaller)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _data_dir() -> Path:
    """
    Папка для пользовательских данных: модели, журналы, настройки.

    * Переменная окружения NEUROVOX_HOME имеет приоритет (нужна для тестов).
    * В собранной программе (.exe) данные лежат в %LOCALAPPDATA%\\NeuroVox: туда
      можно писать без прав администратора, где бы ни лежала сама программа.
    * При запуске из исходников — рядом с кодом.
    """
    override = os.environ.get("NEUROVOX_HOME")
    if override:
        return Path(override).expanduser()
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(root) / APP_NAME
        return Path.home() / f".{APP_NAME.lower()}"
    return _base_dir()


BASE_DIR: Path = _base_dir()
DATA_DIR: Path = _data_dir()
MODELS_DIR: Path = DATA_DIR / "models"
LOGS_DIR: Path = DATA_DIR / "logs"
SETTINGS_FILE: Path = DATA_DIR / "settings.json"

# ---------------------------------------------------------------------------
# Модели Silero TTS
# ---------------------------------------------------------------------------

# Прямые ссылки на официальные модели Silero (см. models.yml в snakers4/silero-models).
SILERO_MODELS = {
    "v4_ru": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
    "v5_ru": "https://models.silero.ai/models/tts/ru/v5_ru.pt",
}

# Модель по умолчанию: v4_ru — самая обкатанная.
DEFAULT_TTS_MODEL = "v4_ru"

# Голоса модели v4_ru и v5_ru (из официальной документации Silero).
# Ключ — имя в модели, значение — отображаемое имя в интерфейсе.
SILERO_SPEAKERS = {
    "baya": "Байя (женский)",
    "kseniya": "Ксения (женский)",
    "xenia": "Ксения-2 (женский)",
    "aidar": "Айдар (мужской)",
    "eugene": "Евгений (мужской)",
}
DEFAULT_SPEAKER = "baya"

# Частота дискретизации синтеза. 48000 — лучшее качество, 24000 — быстрее.
TTS_SAMPLE_RATE = 48000

# Минимально допустимый размер файла модели (защита от битой/обрезанной загрузки).
MIN_MODEL_SIZE_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------------------
# Параметры OCR и фильтрации
# ---------------------------------------------------------------------------

OCR_ENGINES = ("easyocr", "tesseract")
DEFAULT_OCR_ENGINE = "easyocr"
OCR_LANGUAGES = ["ru", "en"]

# Порог похожести (в процентах) для отсеивания повторов.
SIMILARITY_THRESHOLD = 85

# Минимальная длина осмысленной реплики (в символах) после очистки.
MIN_TEXT_LENGTH = 3

# Сколько раз в секунду захватывать экран.
DEFAULT_CAPTURE_FPS = 3.5

# Сколько кадров подряд текст должен оставаться неизменным, прежде чем его
# считать «устоявшимся». Защищает от озвучивания недопечатанных строк.
STABLE_FRAMES_REQUIRED = 2

# Сколько пустых кадров подряд означает «реплика закончилась» (после этого
# та же фраза, появившаяся снова, будет озвучена повторно).
PAUSE_FRAMES_TO_FORGET = 3

# Максимальный размер очереди озвучки (старое отбрасывается, чтобы не копились задержки).
MAX_TTS_QUEUE = 3


@dataclass
class Settings:
    """Пользовательские настройки, сохраняемые между запусками."""

    ocr_engine: str = DEFAULT_OCR_ENGINE
    tts_model: str = DEFAULT_TTS_MODEL
    speaker: str = DEFAULT_SPEAKER
    speed: float = 1.0
    volume: float = 1.0
    capture_fps: float = DEFAULT_CAPTURE_FPS
    similarity_threshold: int = SIMILARITY_THRESHOLD
    tesseract_path: str = ""
    # Область захвата: left, top, width, height (в пикселях экрана).
    roi: list = field(default_factory=lambda: [400, 800, 1100, 120])
    overlay_visible: bool = True

    # -- сохранение / загрузка -------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Сохраняет настройки в JSON. Ошибки записи не должны ронять приложение."""
        path = path or SETTINGS_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(asdict(self), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Не удалось сохранить настройки: %s", exc)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        """Загружает настройки. При любой ошибке возвращает значения по умолчанию."""
        path = path or SETTINGS_FILE
        settings = cls()
        if not path.exists():
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Файл настроек повреждён, используются значения по умолчанию: %s", exc)
            return settings

        if not isinstance(data, dict):
            return settings

        # Применяем только известные поля, чтобы старые файлы не ломали программу.
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        settings._validate()
        return settings

    def _validate(self) -> None:
        """Приводит значения к допустимым диапазонам."""
        if self.ocr_engine not in OCR_ENGINES:
            self.ocr_engine = DEFAULT_OCR_ENGINE
        if self.tts_model not in SILERO_MODELS:
            self.tts_model = DEFAULT_TTS_MODEL
        if self.speaker not in SILERO_SPEAKERS:
            self.speaker = DEFAULT_SPEAKER
        self.speed = _clamp(_to_float(self.speed, 1.0), 0.5, 2.0)
        self.volume = _clamp(_to_float(self.volume, 1.0), 0.0, 1.0)
        self.capture_fps = _clamp(_to_float(self.capture_fps, DEFAULT_CAPTURE_FPS), 1.0, 8.0)
        self.similarity_threshold = int(
            _clamp(_to_float(self.similarity_threshold, SIMILARITY_THRESHOLD), 50, 100)
        )
        if (
            not isinstance(self.roi, (list, tuple))
            or len(self.roi) != 4
            or not all(isinstance(v, (int, float)) for v in self.roi)
        ):
            self.roi = [400, 800, 1100, 120]
        else:
            left, top, width, height = (int(v) for v in self.roi)
            self.roi = [left, top, max(width, 40), max(height, 20)]


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает вывод логов в консоль и в файл (всё на русском)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    # Не дублируем обработчики при повторном вызове.
    if root.handlers:
        return

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        file_handler = logging.FileHandler(LOGS_DIR / "neurovox.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("Не удалось открыть файл журнала: %s", exc)


def find_tesseract(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Ищет исполняемый файл Tesseract.

    Порядок поиска: путь из настроек -> переменная окружения -> PATH -> стандартные папки Windows.
    Возвращает путь или None, если Tesseract не найден.
    """
    import shutil

    candidates = []
    if custom_path:
        candidates.append(custom_path)
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path:
        candidates.append(env_path)
    in_path = shutil.which("tesseract")
    if in_path:
        candidates.append(in_path)
    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None
