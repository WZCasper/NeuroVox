# -*- coding: utf-8 -*-
"""
NeuroVox — офлайн-озвучка игровых субтитров (Windows).

Запуск:            python main.py    (или NeuroVox.exe)
Самопроверка:      NeuroVox.exe --selftest [--strict] [--models] [--out отчёт.txt]
"""

import argparse
import logging
import multiprocessing
import os
import sys
import threading
import traceback
from typing import List, Optional

import config


def _ensure_std_streams() -> None:
    """
    Подставляет «пустые» sys.stdout/sys.stderr, если консоли нет.

    В собранной программе без консоли (.exe) эти потоки равны None, и любая
    библиотека, которая пишет в них (индикаторы загрузки, print), упала бы с ошибкой.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _enable_dpi_awareness() -> None:
    """
    Включает поддержку высокого DPI в Windows.

    Без этого координаты рамки и координаты захвата экрана расходятся на
    мониторах с масштабированием 125–150%, и OCR смотрит не туда.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001 — не критично, просто продолжаем
        pass


def _show_fatal_error(details: str) -> None:
    """Показывает окно с фатальной ошибкой (консоли в .exe нет, и ошибку иначе не увидеть)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "NeuroVox — критическая ошибка",
            "Программа неожиданно завершила работу.\n\n"
            f"{details}\n\nПодробности сохранены в журнале:\n{config.LOGS_DIR}",
        )
        root.destroy()
    except Exception:  # noqa: BLE001
        pass


def _install_exception_hooks() -> None:
    """Логирует любые необработанные исключения (в основном и в рабочих потоках)."""
    log = logging.getLogger("neurovox.main")

    def handle(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("Необработанное исключение:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))

    def handle_thread(args):
        handle(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = handle
    threading.excepthook = handle_thread


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=config.APP_NAME,
        description="Озвучка игровых субтитров в реальном времени (офлайн).",
    )
    parser.add_argument("--selftest", action="store_true", help="проверить, что все компоненты работают, и выйти")
    parser.add_argument("--strict", action="store_true", help="самопроверка: отсутствие любой библиотеки считать ошибкой")
    parser.add_argument("--models", action="store_true", help="самопроверка: скачать и проверить модели озвучки и OCR")
    parser.add_argument("--out", default=None, help="самопроверка: сохранить отчёт в этот файл")
    # Неизвестные аргументы игнорируем, чтобы запуск из ярлыков и «Открыть с помощью» не ломался.
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_std_streams()

    # Кодировка вывода: русские сообщения не должны ломаться в консоли Windows.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = _parse_args(argv)

    config.setup_logging()
    _install_exception_hooks()
    log = logging.getLogger("neurovox.main")
    log.info("Запуск %s версии %s", config.APP_NAME, config.APP_VERSION)

    if sys.version_info < (3, 9):
        message = "Требуется Python версии 3.9 или новее."
        log.error(message)
        _show_fatal_error(message)
        return 1

    _enable_dpi_awareness()

    if args.selftest:
        import selftest

        return selftest.run(strict=args.strict, models=args.models, out_path=args.out)

    try:
        from gui import MainWindow

        app = MainWindow()
        app.mainloop()
    except ImportError as exc:
        message = (
            f"Не установлена нужная библиотека: {exc.name or exc}.\n"
            "Выполните команду:  pip install -r requirements.txt"
        )
        log.exception(message)
        _show_fatal_error(message)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("Критическая ошибка")
        _show_fatal_error(str(exc))
        return 1

    log.info("Программа закрыта.")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()  # нужно для собранной программы (PyInstaller)
    sys.exit(main())
