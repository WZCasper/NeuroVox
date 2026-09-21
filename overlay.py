# -*- coding: utf-8 -*-
"""
Прозрачный безрамочный оверлей для выбора области захвата (ROI).

Пользователь перетаскивает окно поверх субтитров игры и растягивает его
за правый нижний угол. Координаты окна на экране и есть область захвата.

Особенность реализации: окно делается полупрозрачным (-alpha), а не
«сквозным» (-transparentcolor). Сквозное окно не ловит клики мыши, и его
нельзя было бы перетаскивать.
"""

import logging
import tkinter as tk
from typing import Callable, Optional, Tuple

import customtkinter as ctk

logger = logging.getLogger("neurovox.overlay")

MIN_WIDTH = 80
MIN_HEIGHT = 30
GRIP_SIZE = 22           # размер «уголка» для изменения размера
BORDER_COLOR = "#00d4ff"
FILL_COLOR = "#0b1d2a"
IDLE_ALPHA = 0.30        # прозрачность, пока пользователь ничего не делает
ACTIVE_ALPHA = 0.55      # прозрачность во время перетаскивания


class SelectionOverlay(ctk.CTkToplevel):
    """Рамка выбора области захвата, которую можно двигать и растягивать."""

    def __init__(
        self,
        master,
        roi: Tuple[int, int, int, int],
        on_change: Optional[Callable[[Tuple[int, int, int, int]], None]] = None,
    ) -> None:
        super().__init__(master)
        self._on_change = on_change
        self._drag_offset: Tuple[int, int] = (0, 0)
        self._resize_origin: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._mode: str = ""

        left, top, width, height = roi
        self.overrideredirect(True)                 # без рамки и заголовка
        self.attributes("-topmost", True)           # поверх всех окон (в т.ч. игры в оконном режиме)
        self._set_alpha(IDLE_ALPHA)
        self.configure(fg_color=FILL_COLOR)
        self.geometry(f"{max(width, MIN_WIDTH)}x{max(height, MIN_HEIGHT)}+{left}+{top}")
        self.title("NeuroVox — область захвата")

        # Внешняя рамка (canvas рисует контур, который хорошо виден на любом фоне).
        self._canvas = tk.Canvas(self, bg=FILL_COLOR, highlightthickness=0, bd=0, cursor="fleur")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._redraw)

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Motion>", self._on_hover)

        # Стрелки на клавиатуре — точная подстройка (Shift = быстрее, Ctrl = размер).
        for key in ("<Left>", "<Right>", "<Up>", "<Down>"):
            self.bind(key, self._on_arrow)
        self.bind("<Escape>", lambda _e: self.hide())

    # -- публичные методы -----------------------------------------------------

    def get_roi(self) -> Tuple[int, int, int, int]:
        """Текущая область захвата на экране: (left, top, width, height)."""
        self.update_idletasks()
        return (self.winfo_x(), self.winfo_y(), self.winfo_width(), self.winfo_height())

    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        left, top, width, height = roi
        self.geometry(f"{max(width, MIN_WIDTH)}x{max(height, MIN_HEIGHT)}+{left}+{top}")
        self._notify()

    def show(self) -> None:
        self.deiconify()
        self.attributes("-topmost", True)
        self.lift()

    def hide(self) -> None:
        self.withdraw()

    def is_visible(self) -> bool:
        return bool(self.winfo_viewable())

    # -- отрисовка ------------------------------------------------------------

    def _set_alpha(self, value: float) -> None:
        try:
            self.attributes("-alpha", value)
        except tk.TclError:
            # Некоторые окружения не поддерживают прозрачность — работаем без неё.
            logger.debug("Прозрачность окна не поддерживается")

    def _redraw(self, _event=None) -> None:
        canvas = self._canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 4 or height < 4:
            return

        canvas.create_rectangle(1, 1, width - 2, height - 2, outline=BORDER_COLOR, width=3)
        canvas.create_text(
            width // 2, height // 2,
            text="Область субтитров\nперетащите  •  тяните за угол",
            fill="#e8faff", font=("Segoe UI", 10), justify="center",
        )
        # Уголок изменения размера в правом нижнем углу.
        canvas.create_polygon(
            width - GRIP_SIZE, height - 2,
            width - 2, height - 2,
            width - 2, height - GRIP_SIZE,
            fill=BORDER_COLOR, outline="",
        )

    # -- мышь -----------------------------------------------------------------

    def _in_grip(self, x: int, y: int) -> bool:
        width, height = self._canvas.winfo_width(), self._canvas.winfo_height()
        return x >= width - GRIP_SIZE and y >= height - GRIP_SIZE

    def _on_hover(self, event) -> None:
        self._canvas.configure(cursor="size_nw_se" if self._in_grip(event.x, event.y) else "fleur")

    def _on_press(self, event) -> None:
        self._set_alpha(ACTIVE_ALPHA)
        if self._in_grip(event.x, event.y):
            self._mode = "resize"
            self._resize_origin = (event.x_root, event.y_root, self.winfo_width(), self.winfo_height())
        else:
            self._mode = "move"
            self._drag_offset = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _on_motion(self, event) -> None:
        if self._mode == "move":
            x = event.x_root - self._drag_offset[0]
            y = event.y_root - self._drag_offset[1]
            self.geometry(f"+{x}+{y}")
        elif self._mode == "resize":
            start_x, start_y, start_w, start_h = self._resize_origin
            new_w = max(MIN_WIDTH, start_w + (event.x_root - start_x))
            new_h = max(MIN_HEIGHT, start_h + (event.y_root - start_y))
            self.geometry(f"{new_w}x{new_h}")

    def _on_release(self, _event) -> None:
        self._mode = ""
        self._set_alpha(IDLE_ALPHA)
        self._notify()

    def _on_arrow(self, event) -> None:
        step = 10 if (event.state & 0x0001) else 1      # Shift — крупный шаг
        resize = bool(event.state & 0x0004)             # Ctrl — менять размер
        dx = {"Left": -step, "Right": step}.get(event.keysym, 0)
        dy = {"Up": -step, "Down": step}.get(event.keysym, 0)
        left, top, width, height = self.get_roi()
        if resize:
            width = max(MIN_WIDTH, width + dx)
            height = max(MIN_HEIGHT, height + dy)
        else:
            left, top = left + dx, top + dy
        self.set_roi((left, top, width, height))

    def _notify(self) -> None:
        if self._on_change:
            try:
                self._on_change(self.get_roi())
            except Exception:  # noqa: BLE001 — сбой обработчика не должен ломать оверлей
                logger.exception("Ошибка в обработчике изменения области")
