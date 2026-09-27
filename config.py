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
APP_VERSION = "1.1.0"


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
# Словарь ударений и произношений: пользователь дописывает туда слова, которые озвучка
# произносит неверно (имена героев, игровые термины). Читается при каждой фразе.
STRESS_DICT_FILE: Path = DATA_DIR / "stress_dict.txt"

# ---------------------------------------------------------------------------
# Модели Silero TTS
# ---------------------------------------------------------------------------

# Прямые ссылки на официальные модели Silero (см. models.yml в snakers4/silero-models).
#
# v5_5_ru — актуальная русская модель: автоматические ударения, разбор омографов
# (замОк / зАмок) и вопросительная интонация. В v4_ru омографов нет, а редкие слова
# и имена она ставит хуже — поэтому по умолчанию используется v5_5_ru, а v4_ru
# оставлена как запасной вариант.
SILERO_MODELS = {
    "v5_5_ru": "https://models.silero.ai/models/tts/ru/v5_5_ru.pt",
    "v4_ru": "https://models.silero.ai/models/tts/ru/v4_ru.pt",
}

DEFAULT_TTS_MODEL = "v5_5_ru"

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

# Сколько раз в секунду делать снимки экрана. Снимок стоит единицы миллисекунд, а
# распознавание — сотни, поэтому они разнесены по разным потокам: снимки идут быстро
# и копятся в буфере, а распознавание разбирает буфер по очереди.
DEFAULT_CAPTURE_FPS = 15.0
MIN_CAPTURE_FPS = 2.0
MAX_CAPTURE_FPS = 30.0

# Сколько секунд картинка в области должна оставаться неизменной, чтобы считать, что
# субтитр показан целиком (не «печатается» и не появляется плавно).
CAPTURE_SETTLE_SECONDS = 0.30

# Если картинка не успокаивается (анимированный фон, очень медленная печать), снимок
# всё равно отправляется на распознавание не реже, чем раз в столько секунд.
CAPTURE_MAX_WAIT_SECONDS = 1.5

# Сколько секунд область должна быть пустой, чтобы считать реплику законченной.
CAPTURE_EMPTY_HOLD_SECONDS = 0.8

# Буфер снимков: прочитанные снимки удаляются через FRAME_RETENTION_SECONDS, непрочитанные
# (если распознавание не справляется) — через FRAME_MAX_UNREAD_AGE_SECONDS: реплика
# полуминутной давности в игре уже неактуальна.
FRAME_RETENTION_SECONDS = 10.0
FRAME_MAX_UNREAD_AGE_SECONDS = 30.0
FRAME_BUFFER_MAX_FRAMES = 48

# Несколько областей субтитров.
MAX_AREAS = 6
DEFAULT_ROI = (400, 800, 1100, 120)

# Сколько кадров подряд текст должен оставаться неизменным, прежде чем его
# считать «устоявшимся». Защищает от озвучивания недопечатанных строк.
STABLE_FRAMES_REQUIRED = 2

# Сколько пустых кадров подряд означает «реплика закончилась» (после этого
# та же фраза, появившаяся снова, будет озвучена повторно).
PAUSE_FRAMES_TO_FORGET = 3

# Очередь озвучки. Реплики читаются строго по очереди и не теряются; отбрасываются только
# те, что ждали дольше MAX_PHRASE_AGE_SECONDS (или не поместились в очередь).
MAX_TTS_QUEUE = 12
MAX_PHRASE_AGE_SECONDS = 25.0

# Пауза между репликами из РАЗНЫХ областей (настраивается в окне) и короткая пауза
# между репликами одной и той же области.
DEFAULT_AREA_PAUSE = 1.0
MAX_AREA_PAUSE = 5.0
SAME_AREA_GAP_SECONDS = 0.15

# Версия формата файла настроек (нужна для переноса старых настроек).
SETTINGS_VERSION = 2


@dataclass
class Settings:
    """Пользовательские настройки, сохраняемые между запусками."""

    settings_version: int = SETTINGS_VERSION
    ocr_engine: str = DEFAULT_OCR_ENGINE
    tts_model: str = DEFAULT_TTS_MODEL
    speaker: str = DEFAULT_SPEAKER
    speed: float = 1.0
    volume: float = 1.0
    capture_fps: float = DEFAULT_CAPTURE_FPS
    similarity_threshold: int = SIMILARITY_THRESHOLD
    tesseract_path: str = ""
    # Области захвата: список [left, top, width, height] (в пикселях экрана).
    # Порядок в списке = порядок озвучки, когда реплики появляются одновременно.
    rois: list = field(default_factory=lambda: [list(DEFAULT_ROI)])
    # Пауза (секунды) между репликами из разных областей.
    area_pause: float = DEFAULT_AREA_PAUSE
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

        # Файлы версии 1 хранили одну область в поле «roi» — переносим её в список.
        if "rois" not in data and "roi" in data:
            data["rois"] = [data["roi"]]
        data.pop("roi", None)

        old_version = _to_int(data.get("settings_version"), 1)

        # Применяем только известные поля, чтобы старые файлы не ломали программу.
        for key, value in data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        settings._migrate(old_version)
        settings._validate()
        return settings

    def _migrate(self, old_version: int) -> None:
        """Переносит настройки из файла старой версии."""
        if old_version < 2:
            # Прежние 3,5 кадра/с были слишком медленными — берём новое значение по умолчанию.
            self.capture_fps = DEFAULT_CAPTURE_FPS
            # v4_ru была моделью по умолчанию, а v5_5_ru ставит ударения заметно точнее.
            # Выбрать v4_ru снова можно в окне программы.
            if self.tts_model == "v4_ru":
                self.tts_model = DEFAULT_TTS_MODEL
        self.settings_version = SETTINGS_VERSION

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
        self.capture_fps = _clamp(
            _to_float(self.capture_fps, DEFAULT_CAPTURE_FPS), MIN_CAPTURE_FPS, MAX_CAPTURE_FPS
        )
        self.area_pause = _clamp(_to_float(self.area_pause, DEFAULT_AREA_PAUSE), 0.0, MAX_AREA_PAUSE)
        self.similarity_threshold = int(
            _clamp(_to_float(self.similarity_threshold, SIMILARITY_THRESHOLD), 50, 100)
        )
        self.rois = normalize_rois(self.rois)


def normalize_rois(rois) -> list:
    """
    Приводит список областей к безопасному виду.

    Некорректные записи отбрасываются, размеры ограничиваются снизу, лишние области
    (сверх MAX_AREAS) обрезаются. Если ни одной верной области нет — возвращается область
    по умолчанию, чтобы программе всегда было что захватывать.
    """
    cleaned = []
    if isinstance(rois, (list, tuple)):
        for roi in rois:
            if (
                isinstance(roi, (list, tuple))
                and len(roi) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in roi)
            ):
                left, top, width, height = (int(v) for v in roi)
                cleaned.append([left, top, max(width, 40), max(height, 20)])
            if len(cleaned) >= MAX_AREAS:
                break
    return cleaned or [list(DEFAULT_ROI)]


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
