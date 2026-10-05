# -*- coding: utf-8 -*-
"""
Тесты SelectionOverlay.set_reading() — прозрачность рамки на время озвучки её
области — и платформенной защиты _exclude_from_capture().

Требуют рабочего графического дисплея (создают настоящее окно CustomTkinter) —
если дисплея нет, весь модуль пропускается, а не падает (см. test_gui.py: то же
самое решение и по той же причине — headless Linux без Xvfb не редкость).
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import tkinter

    _probe = tkinter.Tk()
    _probe.destroy()
    _DISPLAY_AVAILABLE = True
except Exception:  # noqa: BLE001 — любая причина недоступности дисплея
    _DISPLAY_AVAILABLE = False

if _DISPLAY_AVAILABLE:
    import customtkinter as ctk

    from overlay import ACTIVE_ALPHA, IDLE_ALPHA, READING_ALPHA, SelectionOverlay


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class SetReadingTests(unittest.TestCase):
    def setUp(self):
        self.root = ctk.CTk()
        self.root.withdraw()
        self.overlay = SelectionOverlay(self.root, (10, 10, 300, 100), index=0)
        self.overlay.show()
        self.root.update()
        # Некоторые X11-окружения без compositor'а (например, голый Xvfb, в
        # котором гоняются эти тесты) применяют "-alpha", заданную ДО первого
        # показа окна, только после того, как окно уже было показано и обработан
        # хотя бы один цикл событий. Один и тот же эффект есть и у кода ДО этого
        # изменения (idle/active-прозрачность при перетаскивании рамки мышью),
        # так что это не имеет отношения к set_reading(): просто «довыставляем»
        # обычную прозрачность явно, чтобы тесты ниже начинали с известного,
        # предсказуемого состояния, а не зависели от этой особенности площадки.
        self.overlay._set_alpha(IDLE_ALPHA)
        self.root.update()

    def tearDown(self):
        # CustomTkinter планирует собственные внутренние after(...)-проверки
        # (например, масштаб DPI); если их не отменить явно, они могут сработать
        # уже после destroy() и напечатать безобидное, но лишнее "invalid command
        # name" — та же причина и то же решение, что и в test_gui.py.
        try:
            for after_id in self.root.tk.eval("after info").split():
                self.root.after_cancel(after_id)
        except tkinter.TclError:
            pass
        try:
            self.overlay.destroy()
            self.root.destroy()
        except tkinter.TclError:
            pass

    def test_reading_true_makes_the_overlay_fully_transparent(self):
        self.overlay.set_reading(True)
        self.root.update()
        self.assertAlmostEqual(self.overlay.attributes("-alpha"), READING_ALPHA, places=6)

    def test_overlay_stays_viewable_while_transparent(self):
        # "Прозрачная" — не то же самое, что "скрытая" (withdraw): окно должно
        # оставаться viewable, иначе get_roi() перестанет сообщать актуальные
        # координаты, пока идёт озвучка.
        self.overlay.set_reading(True)
        self.root.update()
        self.assertTrue(self.overlay.is_visible())

    def test_reading_false_restores_idle_alpha(self):
        self.overlay.set_reading(True)
        self.root.update()
        self.overlay.set_reading(False)
        self.root.update()
        self.assertAlmostEqual(self.overlay.attributes("-alpha"), IDLE_ALPHA, places=6)

    def test_does_nothing_while_hidden_by_the_user(self):
        self.overlay.hide()
        self.root.update()
        self.overlay.set_reading(True)
        self.root.update()
        self.assertFalse(self.overlay.is_visible(), "скрытая пользователем рамка не должна появляться")

    def test_does_nothing_while_being_dragged(self):
        # Пока пользователь тащит рамку (_on_press уже выставил ACTIVE_ALPHA),
        # set_reading не должен вмешиваться в прозрачность поверх этого.
        self.overlay._mode = "move"
        self.overlay._set_alpha(ACTIVE_ALPHA)
        self.root.update()
        self.overlay.set_reading(True)
        self.root.update()
        self.assertAlmostEqual(self.overlay.attributes("-alpha"), ACTIVE_ALPHA, places=6)
        self.overlay._mode = ""

    def test_reading_state_is_tracked_even_while_hidden(self):
        # Внутреннее состояние ._reading должно отражать последний вызов, даже
        # если видимого эффекта сейчас нет (рамка скрыта) — на случай, если
        # понадобится в будущем опереться на него при повторном show().
        self.overlay.hide()
        self.root.update()
        self.overlay.set_reading(True)
        self.assertTrue(self.overlay._reading)


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class ExcludeFromCaptureTests(unittest.TestCase):
    """
    _exclude_from_capture() — это необязательное улучшение (исключает рамку из
    снимка mss на Windows 10 2004+/11). На этой платформе (не Windows) он
    обязан тихо ничего не делать и не бросать исключений — рамка должна
    продолжать нормально работать и там, где это улучшение недоступно.
    """

    def setUp(self):
        self.root = ctk.CTk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except tkinter.TclError:
            pass

    def test_no_op_and_no_exception_on_non_windows_platform(self):
        self.assertNotEqual(sys.platform, "win32", "этот тест предполагает запуск не на Windows")
        overlay = SelectionOverlay(self.root, (10, 10, 300, 100), index=0)
        # Конструктор уже вызвал _exclude_from_capture() один раз, не упав —
        # иначе бы setUp/создание рамки выше уже провалилось. Проверяем также
        # повторный прямой вызов — для надёжности и для явности намерения теста.
        overlay._exclude_from_capture()
        overlay.destroy()


if __name__ == "__main__":
    unittest.main()
