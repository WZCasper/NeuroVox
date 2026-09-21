# -*- mode: python ; coding: utf-8 -*-
"""
Спецификация PyInstaller для NeuroVox.

Результат: папка dist/NeuroVox/ с файлом NeuroVox.exe (без окна консоли).
Сборка:    pyinstaller --noconfirm --clean NeuroVox.spec

Почему «папка», а не один .exe: PyTorch весит сотни мегабайт, и единый .exe
распаковывался бы во временную папку при КАЖДОМ запуске (десятки секунд ожидания).
Папка запускается мгновенно; для удобства её упаковывают в установщик или zip.
"""

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH)  # noqa: F821 — SPECPATH задаёт сам PyInstaller
ICON = ROOT / "assets" / "neurovox.ico"


def _available(name):
    """Есть ли пакет в окружении (позволяет собирать и без тяжёлых библиотек при отладке)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


datas, binaries, hiddenimports = [], [], []

# Пакеты с данными или динамически подгружаемыми модулями: берём их целиком.
#   customtkinter — темы и шрифты интерфейса
#   rapidfuzz     — C-модули выбираются в момент запуска
#   easyocr       — списки символов и словари распознавания
#   skimage       — «ленивая» загрузка подмодулей (нужна EasyOCR)
#   shapely, pyclipper, yaml, bidi — зависимости EasyOCR с бинарными модулями
#   sounddevice / _sounddevice_data — библиотека PortAudio для вывода звука
for package in (
    "customtkinter",
    "rapidfuzz",
    "easyocr",
    "skimage",
    "shapely",
    "pyclipper",
    "yaml",
    "bidi",
    "sounddevice",
    "_sounddevice_data",
):
    if _available(package):
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    else:
        print(f"[spec] пакет {package} не найден, пропускаем")

# Метаданные пакетов: часть библиотек читает свою версию через importlib.metadata.
for distribution in (
    "torch", "torchvision", "numpy", "scipy", "scikit-image", "imageio", "tifffile",
    "lazy_loader", "packaging", "pillow", "opencv-python-headless", "tqdm", "easyocr",
    "PyYAML", "shapely", "pyclipper", "python-bidi", "rapidfuzz", "mss", "sounddevice",
    "customtkinter", "pytesseract",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:  # noqa: BLE001 — пакета нет в окружении, это не ошибка
        pass

# Модели Silero — это архивы torch.package: внутри лежит собственный Python-код модели,
# который при загрузке импортирует стандартные модули. PyInstaller не видит внутрь .pt,
# поэтому перечисляем стандартные модули явно, иначе модель могла бы не загрузиться.
STDLIB_FOR_MODELS = [
    "abc", "array", "ast", "base64", "binascii", "bisect", "cmath", "codecs", "collections",
    "collections.abc", "contextlib", "copy", "copyreg", "csv", "dataclasses", "datetime",
    "decimal", "difflib", "dis", "enum", "fractions", "functools", "glob", "hashlib", "heapq",
    "importlib", "importlib.abc", "importlib.machinery", "importlib.util", "inspect", "io",
    "itertools", "json", "locale", "logging", "math", "numbers", "operator", "os", "pathlib",
    "pickle", "pickletools", "platform", "queue", "random", "re", "shutil", "statistics",
    "string", "struct", "tempfile", "textwrap", "threading", "time", "tokenize", "traceback",
    "types", "typing", "unicodedata", "warnings", "weakref", "zipfile", "zlib",
]

hiddenimports += STDLIB_FOR_MODELS
if _available("torch"):
    hiddenimports.append("torch.package")
hiddenimports += [
    "config", "capture", "ocr_engine", "text_filter", "tts_engine", "workers",
    "overlay", "gui", "selftest",
]

# Заведомо ненужное: уменьшает размер и ускоряет сборку.
EXCLUDES = [
    "matplotlib", "IPython", "jupyter", "notebook", "pytest",
    "tensorboard", "torch.utils.tensorboard",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeuroVox",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,        # UPX часто вызывает ложные срабатывания антивирусов и ломает DLL torch
    console=False,    # окно консоли не нужно: ошибки показываются в окне и пишутся в журнал
    icon=str(ICON) if ICON.exists() else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NeuroVox",
)
