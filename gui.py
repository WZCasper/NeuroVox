# -*- coding: utf-8 -*-
"""
Главное окно NeuroVox (CustomTkinter, тёмная тема, интерфейс на русском).

Окно работает только в главном потоке. Рабочие потоки не трогают виджеты
напрямую: они кладут сообщения в EventBus, а окно забирает их по таймеру.
"""

import logging
import time
import tkinter as tk
from tkinter import messagebox
from typing import Dict

import customtkinter as ctk

import config
from overlay import SelectionOverlay
from workers import EventBus, NeuroVoxEngine, SharedSettings

logger = logging.getLogger("neurovox.gui")

POLL_INTERVAL_MS = 100
MAX_LOG_LINES = 400

OCR_LABELS = {
    "easyocr": "EasyOCR (универсальный)",
    "tesseract": "Tesseract (быстрый)",
}
MODEL_LABELS = {
    "v4_ru": "Silero v4 (проверенная)",
    "v5_ru": "Silero v5 (новая)",
}

# Цвета статуса.
COLOR_IDLE = "#8a94a6"
COLOR_LOADING = "#f5a623"
COLOR_RUNNING = "#3ddc84"
COLOR_ERROR = "#ff5c5c"


class MainWindow(ctk.CTk):
    """Главное окно управления программой."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"{config.APP_NAME} — голос для субтитров")
        self.geometry("620x760")
        self.minsize(560, 680)

        self.settings = config.Settings.load()
        self.shared = SharedSettings(self.settings)
        self.bus = EventBus()
        self.engine = NeuroVoxEngine(self.shared, self.bus)
        self.overlay: SelectionOverlay = None  # создаётся после построения интерфейса

        self._voice_by_label: Dict[str, str] = {label: key for key, label in config.SILERO_SPEAKERS.items()}
        self._engine_by_label: Dict[str, str] = {label: key for key, label in OCR_LABELS.items()}
        self._model_by_label: Dict[str, str] = {label: key for key, label in MODEL_LABELS.items()}
        self._starting = False

        self._build_ui()
        self._create_overlay()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_INTERVAL_MS, self._poll_events)
        self._log("Добро пожаловать в NeuroVox! Наведите рамку на субтитры и нажмите «Запустить».")

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        # -- шапка ----------------------------------------------------------
        header = ctk.CTkFrame(self, corner_radius=12)
        header.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="NeuroVox", font=ctk.CTkFont(size=26, weight="bold")).grid(
            row=0, column=0, padx=16, pady=(12, 0), sticky="w"
        )
        ctk.CTkLabel(
            header,
            text="Озвучка игровых субтитров в реальном времени — полностью офлайн",
            text_color=COLOR_IDLE,
        ).grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        self.status_dot = ctk.CTkLabel(header, text="●", font=ctk.CTkFont(size=22), text_color=COLOR_IDLE)
        self.status_dot.grid(row=0, column=1, rowspan=2, padx=(0, 6))
        self.status_label = ctk.CTkLabel(header, text="Остановлено", width=140, anchor="w")
        self.status_label.grid(row=0, column=2, rowspan=2, padx=(0, 16))

        # -- кнопки управления ---------------------------------------------
        controls = ctk.CTkFrame(self, corner_radius=12)
        controls.grid(row=1, column=0, padx=16, pady=8, sticky="ew")
        controls.grid_columnconfigure((0, 1), weight=1)

        self.start_button = ctk.CTkButton(
            controls, text="▶  Запустить", height=44, font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1f9d55", hover_color="#187a43", command=self._toggle,
        )
        self.start_button.grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="ew")

        self.overlay_button = ctk.CTkButton(
            controls, text="Скрыть рамку", command=self._toggle_overlay, fg_color="#3a4358", hover_color="#2d3547",
        )
        self.overlay_button.grid(row=1, column=0, padx=(16, 6), pady=(0, 16), sticky="ew")

        self.test_button = ctk.CTkButton(
            controls, text="🔊 Проверить голос", command=self._preview_voice,
            fg_color="#3a4358", hover_color="#2d3547",
        )
        self.test_button.grid(row=1, column=1, padx=(6, 16), pady=(0, 16), sticky="ew")

        self.progress = ctk.CTkProgressBar(controls)
        self.progress.set(0)
        # Полоса прогресса показывается только во время загрузки моделей.

        # -- настройки ------------------------------------------------------
        options = ctk.CTkFrame(self, corner_radius=12)
        options.grid(row=2, column=0, padx=16, pady=8, sticky="ew")
        options.grid_columnconfigure(1, weight=1)

        def add_row(row: int, caption: str) -> None:
            ctk.CTkLabel(options, text=caption, anchor="w").grid(row=row, column=0, padx=(16, 8), pady=8, sticky="w")

        add_row(0, "Голос")
        self.voice_menu = ctk.CTkOptionMenu(
            options, values=list(self._voice_by_label), command=self._on_voice_change,
        )
        self.voice_menu.set(config.SILERO_SPEAKERS.get(self.settings.speaker, config.SILERO_SPEAKERS[config.DEFAULT_SPEAKER]))
        self.voice_menu.grid(row=0, column=1, padx=(0, 16), pady=8, sticky="ew")

        add_row(1, "Скорость речи")
        speed_frame = ctk.CTkFrame(options, fg_color="transparent")
        speed_frame.grid(row=1, column=1, padx=(0, 16), pady=8, sticky="ew")
        speed_frame.grid_columnconfigure(0, weight=1)
        self.speed_slider = ctk.CTkSlider(
            speed_frame, from_=0.5, to=2.0, number_of_steps=30, command=self._on_speed_change,
        )
        self.speed_slider.set(self.settings.speed)
        self.speed_slider.grid(row=0, column=0, sticky="ew")
        self.speed_value = ctk.CTkLabel(speed_frame, text=f"{self.settings.speed:.2f}×", width=54)
        self.speed_value.grid(row=0, column=1, padx=(8, 0))

        add_row(2, "Громкость")
        volume_frame = ctk.CTkFrame(options, fg_color="transparent")
        volume_frame.grid(row=2, column=1, padx=(0, 16), pady=8, sticky="ew")
        volume_frame.grid_columnconfigure(0, weight=1)
        self.volume_slider = ctk.CTkSlider(
            volume_frame, from_=0.0, to=1.0, number_of_steps=20, command=self._on_volume_change,
        )
        self.volume_slider.set(self.settings.volume)
        self.volume_slider.grid(row=0, column=0, sticky="ew")
        self.volume_value = ctk.CTkLabel(volume_frame, text=f"{int(self.settings.volume * 100)}%", width=54)
        self.volume_value.grid(row=0, column=1, padx=(8, 0))

        add_row(3, "Модуль распознавания")
        self.ocr_menu = ctk.CTkOptionMenu(
            options, values=list(self._engine_by_label), command=self._on_engine_change,
        )
        self.ocr_menu.set(OCR_LABELS.get(self.settings.ocr_engine, OCR_LABELS[config.DEFAULT_OCR_ENGINE]))
        self.ocr_menu.grid(row=3, column=1, padx=(0, 16), pady=8, sticky="ew")

        add_row(4, "Модель голоса")
        self.model_menu = ctk.CTkOptionMenu(
            options, values=list(self._model_by_label), command=self._on_model_change,
        )
        self.model_menu.set(MODEL_LABELS.get(self.settings.tts_model, MODEL_LABELS[config.DEFAULT_TTS_MODEL]))
        self.model_menu.grid(row=4, column=1, padx=(0, 16), pady=8, sticky="ew")

        add_row(5, "Язык распознавания")
        ctk.CTkLabel(options, text="Русский (ru)", anchor="w", text_color=COLOR_IDLE).grid(
            row=5, column=1, padx=(0, 16), pady=(8, 12), sticky="w"
        )

        # -- журнал ---------------------------------------------------------
        log_frame = ctk.CTkFrame(self, corner_radius=12)
        log_frame.grid(row=3, column=0, padx=16, pady=(8, 16), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, padx=12, pady=(10, 0), sticky="ew")
        log_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(log_header, text="Журнал: распознанный текст и состояние", anchor="w").grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(
            log_header, text="Очистить", width=80, height=26, fg_color="#3a4358", hover_color="#2d3547",
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(log_frame, wrap="word", font=ctk.CTkFont(size=13))
        self.log_box.grid(row=1, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.log_box.configure(state="disabled")

    def _create_overlay(self) -> None:
        roi = tuple(self.settings.roi)
        self.overlay = SelectionOverlay(self, roi, on_change=self._on_roi_change)
        if not self.settings.overlay_visible:
            self.overlay.hide()
            self.overlay_button.configure(text="Показать рамку")

    # ------------------------------------------------------------------
    # Обработчики действий
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        if self._starting or self.engine.is_running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        # Сохраняем актуальную область на случай, если пользователь двигал рамку.
        self._on_roi_change(self.overlay.get_roi())
        self._starting = True
        self.settings.save()
        self._set_status("Подготовка...", COLOR_LOADING)
        self.start_button.configure(text="■  Остановить", fg_color="#c0392b", hover_color="#992d22")
        self.engine.start()

    def _stop(self) -> None:
        self._starting = False
        self.engine.stop()
        self._set_status("Остановка...", COLOR_LOADING)
        self.start_button.configure(state="disabled")

    def _toggle_overlay(self) -> None:
        if self.overlay.is_visible():
            self.overlay.hide()
            self.overlay_button.configure(text="Показать рамку")
            self.shared.update(overlay_visible=False)
            self.settings.overlay_visible = False
        else:
            self.overlay.show()
            self.overlay_button.configure(text="Скрыть рамку")
            self.shared.update(overlay_visible=True)
            self.settings.overlay_visible = True

    def _preview_voice(self) -> None:
        self._log("Проверка голоса...")
        self.engine.preview_voice()

    def _on_roi_change(self, roi) -> None:
        roi_list = [int(v) for v in roi]
        self.shared.update(roi=roi_list)
        self.settings.roi = roi_list

    def _on_voice_change(self, label: str) -> None:
        speaker = self._voice_by_label.get(label, config.DEFAULT_SPEAKER)
        self.shared.update(speaker=speaker)
        self.settings.speaker = speaker

    def _on_speed_change(self, value: float) -> None:
        value = round(float(value), 2)
        self.speed_value.configure(text=f"{value:.2f}×")
        self.shared.update(speed=value)
        self.settings.speed = value

    def _on_volume_change(self, value: float) -> None:
        value = round(float(value), 2)
        self.volume_value.configure(text=f"{int(value * 100)}%")
        self.shared.update(volume=value)
        self.settings.volume = value

    def _on_engine_change(self, label: str) -> None:
        engine = self._engine_by_label.get(label, config.DEFAULT_OCR_ENGINE)
        self.shared.update(ocr_engine=engine)
        self.settings.ocr_engine = engine
        if self.engine.is_running:
            self._log("Движок распознавания сменится при следующем запуске.")

    def _on_model_change(self, label: str) -> None:
        model = self._model_by_label.get(label, config.DEFAULT_TTS_MODEL)
        self.shared.update(tts_model=model)
        self.settings.tts_model = model
        if self.engine.is_running:
            self._log("Модель голоса сменится при следующем запуске.")

    # ------------------------------------------------------------------
    # События из рабочих потоков
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            for event in self.bus.poll():
                self._handle_event(event)
        except Exception:  # noqa: BLE001 — сбой обработки не должен остановить опрос
            logger.exception("Ошибка обработки события интерфейса")
        finally:
            self.after(POLL_INTERVAL_MS, self._poll_events)

    def _handle_event(self, event) -> None:
        if event.kind == "status":
            self._set_status(event.text, COLOR_LOADING)
            self._log(event.text)
        elif event.kind == "progress":
            self._show_progress(event.value, event.text)
        elif event.kind == "ready":
            self._starting = False
            self._hide_progress()
            self._set_status("Работает", COLOR_RUNNING)
            self._log(event.text)
        elif event.kind == "recognized":
            self._log(f"📖 {event.text}")
        elif event.kind == "spoken":
            self._log(f"🔊 озвучено за {event.value:.2f} с")
        elif event.kind == "error":
            self._set_status("Ошибка", COLOR_ERROR)
            self._log(f"⚠ {event.text}")
        elif event.kind == "stopped":
            self._starting = False
            self._hide_progress()
            self._set_status("Остановлено", COLOR_IDLE)
            self.start_button.configure(
                text="▶  Запустить", fg_color="#1f9d55", hover_color="#187a43", state="normal",
            )
            self._log(event.text)

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text[:22])
        self.status_dot.configure(text_color=color)

    def _show_progress(self, value: float, text: str) -> None:
        self.progress.grid(row=2, column=0, columnspan=2, padx=16, pady=(0, 12), sticky="ew")
        self.progress.set(max(0.0, min(1.0, value)))
        self._set_status(text[:22], COLOR_LOADING)

    def _hide_progress(self) -> None:
        self.progress.grid_remove()

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{stamp}] {message}\n")
        # Ограничиваем размер журнала, чтобы он не разрастался при долгой игре.
        lines = int(self.log_box.index("end-1c").split(".")[0])
        if lines > MAX_LOG_LINES:
            self.log_box.delete("1.0", f"{lines - MAX_LOG_LINES}.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _on_close(self) -> None:
        if self.engine.is_running:
            if not messagebox.askyesno("Выход", "Озвучка ещё работает. Закрыть программу?"):
                return
        try:
            self._on_roi_change(self.overlay.get_roi())
            self.settings.save()
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось сохранить настройки при выходе")
        self.engine.stop()
        try:
            self.overlay.destroy()
        except tk.TclError:
            pass
        self.destroy()
