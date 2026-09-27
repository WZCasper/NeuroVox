# -*- coding: utf-8 -*-
"""
Самопроверка NeuroVox.

Запуск:
    NeuroVox.exe --selftest --strict --out отчёт.txt            — быстрая проверка
    NeuroVox.exe --selftest --strict --models --out отчёт.txt   — плюс скачивание
                                                                  и проверка моделей

Зачем это нужно: у собранной программы (.exe) нет консоли, поэтому убедиться,
что внутрь попали все библиотеки и что озвучка с распознаванием реально работают,
можно только так. Эту проверку автоматически выполняет GitHub Actions перед
публикацией релиза: если что-то сломано, релиз не создаётся.

Код возврата: 0 — всё в порядке, 1 — есть ошибки.
"""

import gc
import importlib
import logging
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Optional

import config

logger = logging.getLogger("neurovox.selftest")

# Без этих библиотек программа не запустится вообще.
_ALWAYS_REQUIRED = ("customtkinter", "mss", "cv2", "numpy", "PIL", "rapidfuzz")
# Остальные обязательны в строгом режиме: в собранной программе они должны быть все.
_STRICT_REQUIRED = ("sounddevice", "pytesseract", "torch", "torchvision", "easyocr", "scipy", "skimage")

_TTS_PHRASE = "Проверка озвучки. Привет, это тест голоса!"

# (фраза, цвет фона, обводка текста) — три типичных стиля игровых субтитров.
_OCR_CASES = (
    ("Куда ты пропал? Мы искали тебя всю ночь.", (28, 34, 51), False),
    ("Не оборачивайся. Они уже рядом, слышишь?", (74, 93, 58), True),
    ("Ладно, я пойду первым, а ты прикрывай меня.", (91, 58, 58), True),
)
_OCR_MIN_SCORE = 75.0

_FONT_CANDIDATES = (
    "arialbd.ttf",
    "segoeuib.ttf",
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


class Report:
    """Собирает результаты проверок и сразу дописывает их в файл отчёта."""

    def __init__(self, out_path: Optional[str]) -> None:
        self._path = Path(out_path) if out_path else None
        self.failures: List[str] = []
        self.warnings: List[str] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")

    def line(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        full = f"[{stamp}] {text}"
        try:
            print(full, flush=True)
        except Exception:  # noqa: BLE001 — консоли может не быть
            pass
        if self._path is not None:
            # Дописываем сразу: если проверка зависнет, частичный отчёт всё равно останется.
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(full + "\n")
        logger.debug(text)

    def section(self, title: str) -> None:
        self.line(f"=== {title} ===")

    def info(self, text: str) -> None:
        self.line(f"     {text}")

    def ok(self, text: str) -> None:
        self.line(f"OK   {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.line(f"ВНИМ {text}")

    def fail(self, text: str) -> None:
        self.failures.append(text)
        self.line(f"ОШИБ {text}")


def _step(report: Report, title: str, func: Callable[[Report], None]) -> None:
    """Выполняет одну проверку; любое исключение превращается в запись об ошибке."""
    report.section(title)
    started = time.perf_counter()
    try:
        func(report)
    except Exception as exc:  # noqa: BLE001 — проверка не должна прерывать остальные
        report.fail(f"{title}: {type(exc).__name__}: {exc}")
        for text in traceback.format_exc().splitlines():
            report.info(text)
    report.info(f"({time.perf_counter() - started:.1f} с)")


# ---------------------------------------------------------------------------
# Быстрые проверки
# ---------------------------------------------------------------------------

def check_environment(rep: Report) -> None:
    rep.info(f"Python {platform.python_version()} | {platform.platform()}")
    frozen = bool(getattr(sys, "frozen", False))
    rep.info(f"Собранная программа (frozen): {frozen}")
    if frozen:
        rep.info(f"Папка программы: {getattr(sys, '_MEIPASS', '?')}")
    rep.info(f"Папка данных: {config.DATA_DIR}")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    probe = config.DATA_DIR / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    rep.ok("папка данных доступна для записи")


def check_imports(rep: Report, strict: bool) -> None:
    for name in _ALWAYS_REQUIRED + _STRICT_REQUIRED:
        required = strict or name in _ALWAYS_REQUIRED
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — бывает не только ImportError (например, нет PortAudio)
            message = f"{name}: не импортируется ({type(exc).__name__}: {exc})"
            (rep.fail if required else rep.warn)(message)
            continue
        rep.ok(f"{name} {getattr(module, '__version__', '')}".strip())

    try:
        import torch
        from torch import package as torch_package
    except Exception:  # noqa: BLE001 — причина уже записана выше
        return
    rep.info(
        f"torch: CUDA доступна = {torch.cuda.is_available()} "
        f"(для видеокарты AMD должно быть False), потоков CPU = {torch.get_num_threads()}"
    )
    total = float((torch.ones(3) * 2).sum().item())
    if total != 6.0 or not hasattr(torch_package, "PackageImporter"):
        rep.fail("torch: вычисления на CPU или torch.package работают неверно")
    else:
        rep.ok("torch: вычисления на CPU работают, torch.package доступен")


def check_filter(rep: Report) -> None:
    from text_filter import SubtitleFilter, similarity

    flt = SubtitleFilter(threshold=85, stable_frames=2)
    frames = ["Куда ты пропал?"] * 3 + [""] * 3 + ["Не оборачивайся, они рядом"] * 3
    spoken = [phrase for phrase in (flt.process(frame) for frame in frames) if phrase]
    expected = ["Куда ты пропал?", "Не оборачивайся, они рядом"]
    if spoken != expected:
        rep.fail(f"фильтр повторов вернул {spoken!r}, ожидалось {expected!r}")
    else:
        rep.ok("фильтр повторов работает (rapidfuzz)")
    if similarity("Привет, Ёжик", "привет ежик") < 95:
        rep.fail("сравнение строк игнорирует регистр/«ё» неверно")


def check_capture(rep: Report) -> None:
    from capture import ScreenCapture

    try:
        capture = ScreenCapture()
        try:
            frame = capture.grab((0, 0, 64, 64))
        finally:
            capture.close()
    except Exception as exc:  # noqa: BLE001
        rep.warn(f"захват экрана не удался: {exc} (на сервере сборки без монитора это допустимо)")
        return
    rep.ok(f"захват экрана работает, кадр {tuple(frame.shape)}")


def check_gui(rep: Report) -> None:
    from gui import MainWindow

    app = MainWindow()
    try:
        for _ in range(3):
            app.update()
            time.sleep(0.05)
        rep.ok(f"главное окно создано: «{app.title()}»")
        rep.ok(f"области при запуске: {len(app.overlays)} (по умолчанию — одна)")

        app._add_area()
        app.update()
        if len(app.overlays) != 2:
            rep.fail(f"кнопка «+» должна была добавить область, областей стало {len(app.overlays)}")
        else:
            rep.ok(f"кнопка «+» добавляет область: областей стало {len(app.overlays)}")

        app.overlays[0].set_roi((50, 50, 300, 80))
        app.update()
        rois = [overlay.get_roi() for overlay in app.overlays]
        if rois[0] != (50, 50, 300, 80):
            rep.fail(f"перемещение рамки не сохранилось: {rois[0]}")
        else:
            rep.ok(f"прозрачные рамки работают, области: {rois}")

        app._remove_area(1)
        app.update()
        if len(app.overlays) != 1:
            rep.fail(f"удаление области не сработало, осталось {len(app.overlays)}")
        else:
            rep.ok("удаление лишней области работает")
    finally:
        app.destroy()


def check_pronunciation(rep: Report) -> None:
    from pronunciation import StressDictionary, normalize_caps

    if normalize_caps("КУДА ТЫ ПРОПАЛ?") != "Куда ты пропал?":
        rep.fail("нормализация капса субтитров работает неверно")
    else:
        rep.ok("нормализация субтитров, набранных капсом, работает")

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stress_dict.txt"
        created = StressDictionary(path).ensure_file()
        if not created.exists() or "Словарь ударений" not in created.read_text(encoding="utf-8-sig"):
            rep.fail("файл словаря ударений не создаётся с инструкцией")
        else:
            rep.ok("словарь ударений создаётся с инструкцией при первом обращении")

        path.write_text("геральт = гер+альт\n", encoding="utf-8")
        sd = StressDictionary(path)
        result = sd.apply("Геральт пришёл.")
        if result != "Гер+альт пришёл.":
            rep.fail(f"словарь ударений применяется неверно: {result!r}")
        else:
            rep.ok("пользовательский словарь ударений применяется корректно")


def check_frame_pipeline(rep: Report) -> None:
    """
    Проверяет связку «трекер области -> буфер -> распознавание» без реального экрана:
    печатающийся, затем повторяющийся субтитр должен дать ровно две реплики (после
    паузы фраза повторяется и должна прозвучать снова), а не одну и не три.
    """
    import numpy as np

    from capture import AreaTracker, preprocess_for_ocr
    from frame_buffer import FrameBuffer
    from text_filter import SubtitleFilter

    def frame(text: str) -> np.ndarray:
        import cv2

        image = np.full((100, 700, 3), (30, 34, 50), dtype=np.uint8)
        cv2.putText(image, text, (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        return preprocess_for_ocr(image)

    tracker = AreaTracker(settle_time=0.05, max_wait=0.3, empty_hold=0.05)
    buffer = FrameBuffer(retention=1.0, max_unread_age=5.0)
    subtitle_filter = SubtitleFilter(threshold=85)
    spoken = []

    # Печать по буквам (шаг короче settle_time — фраза ещё не отправлена), затем текст
    # держится неизменным дольше settle_time (отправлен), потом исчезает дольше empty_hold
    # (отправлена метка «опустела»), и наконец та же фраза появляется снова.
    script = [
        (0.00, "П"), (0.02, "ПР"), (0.04, "ПРИ"), (0.06, "ПРИВЕТ"),
        (0.14, "ПРИВЕТ"),      # держится дольше settle_time (0.06 -> 0.14 = 0.08 с)
        (0.20, None),
        (0.30, None),          # пусто дольше empty_hold (0.20 -> 0.30 = 0.10 с)
        (0.35, "ПРИВЕТ"),
        (0.43, "ПРИВЕТ"),      # снова держится дольше settle_time
    ]
    for now, text in script:
        commit = tracker.update(frame(text) if text else None, now=now)
        if commit is not None:
            buffer.put(area=0, image=commit.image, settled=commit.settled, captured_at=now)

    while True:
        found = buffer.take_next(timeout=0.01)
        if found is None:
            break
        if found.image is None:
            subtitle_filter.forget()
            continue
        phrase = subtitle_filter.process("ПРИВЕТ", settled=found.settled)
        if phrase:
            spoken.append(phrase)

    if spoken != ["ПРИВЕТ", "ПРИВЕТ"]:
        rep.fail(f"конвейер снимков дал {spoken!r}, ожидалось: фраза повторяется дважды")
    else:
        rep.ok("быстрая смена и повтор субтитра обрабатываются верно (буфер снимков + трекер)")


# ---------------------------------------------------------------------------
# Проверки с моделями (нужен интернет при первом запуске)
# ---------------------------------------------------------------------------

def check_tts(rep: Report) -> None:
    import numpy as np

    from tts_engine import SileroTts

    marks = {"last": -1}

    def progress(value: float, message: str) -> None:
        step = int(value * 100) // 25
        if step != marks["last"]:
            marks["last"] = step
            rep.info(message)

    for model_name in config.SILERO_MODELS:
        try:
            marks["last"] = -1
            tts = SileroTts(model_name)
            started = time.perf_counter()
            tts.load(progress=progress)
            rep.ok(f"модель {model_name}: загружена за {time.perf_counter() - started:.1f} с")

            speakers = tts.available_speakers()
            rep.info(f"голоса модели {model_name}: {', '.join(speakers)}")
            for speaker in speakers:
                speeds = (1.0, 1.4) if speaker == config.DEFAULT_SPEAKER else (1.0,)
                for speed in speeds:
                    begun = time.perf_counter()
                    fragments = tts.synthesize(_TTS_PHRASE, speaker, speed)
                    spent = max(time.perf_counter() - begun, 1e-6)
                    audio = np.concatenate(fragments) if fragments else np.zeros(0, dtype=np.float32)
                    seconds = audio.size / float(tts.sample_rate)
                    peak = float(np.abs(audio).max()) if audio.size else 0.0
                    label = f"{model_name}/{speaker} ×{speed}"
                    if seconds < 0.8 or peak < 0.01 or not bool(np.isfinite(audio).all()):
                        rep.fail(f"{label}: пустой или испорченный звук ({seconds:.2f} с, пик {peak:.3f})")
                    else:
                        rep.ok(
                            f"{label}: {seconds:.2f} с речи за {spent:.2f} с "
                            f"({seconds / spent:.1f}× быстрее реального времени), пик {peak:.2f}"
                        )
        except Exception as exc:  # noqa: BLE001 — одна модель не должна скрывать проверку другой
            rep.fail(f"модель {model_name}: {type(exc).__name__}: {exc}")
            for text in traceback.format_exc().splitlines():
                rep.info(text)
        finally:
            tts = None
            gc.collect()


def _find_font(size: int):
    from PIL import ImageFont

    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return None


def _render_subtitle(text: str, background, outline: bool, font):
    """Рисует «игровой субтитр» и возвращает кадр в формате BGR (как отдаёт захват экрана)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1000, 100), background)
    draw = ImageDraw.Draw(image)
    draw.text(
        (20, 28), text, font=font, fill=(255, 255, 255),
        stroke_width=2 if outline else 0, stroke_fill=(0, 0, 0),
    )
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def check_ocr(rep: Report) -> None:
    from capture import preprocess_for_ocr
    from ocr_engine import EasyOcrEngine
    from text_filter import similarity

    font = _find_font(34)
    if font is None:
        rep.warn("не найден шрифт с кириллицей — проверка распознавания пропущена")
        return

    engine = EasyOcrEngine()
    started = time.perf_counter()
    engine.load()
    rep.ok(f"EasyOCR: модель загружена за {time.perf_counter() - started:.1f} с")

    timings = []
    for text, background, outline in _OCR_CASES:
        image = preprocess_for_ocr(_render_subtitle(text, background, outline, font))
        if image is None:
            rep.fail(f"кадр «{text}» признан пустым после бинаризации")
            continue
        begun = time.perf_counter()
        got = engine.recognize(image)
        spent = time.perf_counter() - begun
        timings.append(spent)
        score = similarity(text, got, strict=True)
        message = f"{score:.0f}% за {spent * 1000:.0f} мс | ожидалось: {text} | распознано: {got}"
        if score >= _OCR_MIN_SCORE:
            rep.ok(message)
        else:
            rep.fail(message)
    if timings:
        rep.info(f"среднее время распознавания кадра: {sum(timings) / len(timings) * 1000:.0f} мс")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def run(strict: bool = False, models: bool = False, out_path: Optional[str] = None) -> int:
    """Выполняет самопроверку. Возвращает 0 при успехе и 1, если найдены ошибки."""
    rep = Report(out_path)
    rep.line(
        f"{config.APP_NAME} {config.APP_VERSION}: самопроверка "
        f"(строгий режим: {'да' if strict else 'нет'}, модели: {'да' if models else 'нет'})"
    )

    _step(rep, "Окружение", check_environment)
    _step(rep, "Библиотеки", lambda r: check_imports(r, strict))
    _step(rep, "Фильтр повторов", check_filter)
    _step(rep, "Захват экрана", check_capture)
    _step(rep, "Интерфейс", check_gui)
    _step(rep, "Произношение и словарь ударений", check_pronunciation)
    _step(rep, "Конвейер снимков (несколько областей, быстрая смена)", check_frame_pipeline)
    if models:
        _step(rep, "Озвучка Silero", check_tts)
        _step(rep, "Распознавание EasyOCR", check_ocr)

    rep.section("Итог")
    rep.info(f"предупреждений: {len(rep.warnings)}, ошибок: {len(rep.failures)}")
    if rep.failures:
        for message in rep.failures:
            rep.info(f"ОШИБКА: {message}")
        rep.line("САМОПРОВЕРКА НЕ ПРОЙДЕНА")
        return 1
    rep.line("САМОПРОВЕРКА ПРОЙДЕНА")
    return 0
