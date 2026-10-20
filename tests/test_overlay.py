# -*- coding: utf-8 -*-
"""
Тесты SelectionOverlay.set_reading() — прозрачность рамки на время озвучки её
области — и платформенной защиты _exclude_from_capture().

Требуют рабочего графического дисплея (создают настоящее окно CustomTkinter) —
если дисплея нет, весь модуль пропускается, а не падает (см. test_gui.py: то же
самое решение и по той же причине — headless Linux без Xvfb не редкость).
"""

import contextlib
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

# overlay.py сам по себе не требует дисплея для импорта — он импортирует
# tkinter на верхнем уровне (чтобы дать типы вроде tkinter.Toplevel), но не
# создаёт окно при импорте модуля. Clamp01Tests ниже тестирует _clamp01() —
# чистую функцию без побочных эффектов — и по докстрингу класса намеренно
# не требует дисплея, в отличие от остальных тестов файла. Поэтому сам
# import overlay (и то, что из него берётся для Clamp01Tests) не прячем за
# _DISPLAY_AVAILABLE — иначе Clamp01Tests ломается именно там, где
# задумывался как самый нетребовательный тест в файле.
import overlay as overlay_module
from overlay import IDLE_ALPHA, _clamp01

if _DISPLAY_AVAILABLE:
    import customtkinter as ctk

    from overlay import ACTIVE_ALPHA, READING_ALPHA, SelectionOverlay


class _FakeUser32:
    """
    Имитация ctypes.windll.user32 для тестов _exclude_from_capture() — реальный
    windll существует только на настоящей Windows, а этот набор тестов должен
    корректно проверять ОБЕ ветки (Windows и не-Windows) независимо от того, на
    какой платформе он в данный момент реально выполняется (см. docstring
    ExcludeFromCaptureTests ниже — этого же правила не хватало в прежней версии
    теста, из-за чего он был неверно завязан на реальную платформу раннера).
    """

    def __init__(self, set_display_affinity_results=(1,), get_parent_result=0):
        self.set_display_affinity_calls = []
        self.get_parent_calls = []
        self._results = list(set_display_affinity_results)
        self._get_parent_result = get_parent_result

    def SetWindowDisplayAffinity(self, hwnd, flag):
        self.set_display_affinity_calls.append((hwnd, flag))
        # Каждый вызов забирает следующий заготовленный результат по очереди;
        # когда заготовки кончаются, дальнейшие вызовы считаются успешными.
        if self._results:
            return self._results.pop(0)
        return 1

    def GetParent(self, hwnd):
        self.get_parent_calls.append(hwnd)
        return self._get_parent_result


class _BoomingUser32:
    """Имитация user32, у которой сам вызов Win32 API неожиданно падает."""

    def SetWindowDisplayAffinity(self, hwnd, flag):
        raise OSError("симуляция непредвиденного сбоя вызова Win32 API")


@contextlib.contextmanager
def _simulated_platform(platform, user32=None):
    """
    На время блока подменяет overlay_module.sys.platform и, если передан
    user32, overlay_module.ctypes.windll.user32 — и гарантированно возвращает
    оба обратно в исходное состояние по выходу из блока, даже при исключении
    внутри него.
    """
    original_platform = overlay_module.sys.platform
    had_windll = hasattr(overlay_module.ctypes, "windll")
    original_windll = getattr(overlay_module.ctypes, "windll", None)
    overlay_module.sys.platform = platform
    if user32 is not None:
        overlay_module.ctypes.windll = type("FakeWindll", (), {"user32": user32})()
    try:
        yield
    finally:
        overlay_module.sys.platform = original_platform
        if had_windll:
            overlay_module.ctypes.windll = original_windll
        elif hasattr(overlay_module.ctypes, "windll"):
            del overlay_module.ctypes.windll


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


class Clamp01Tests(unittest.TestCase):
    """
    _clamp01() — чистая функция, тест без tkinter и без дисплея (в отличие от
    остальных тестов этого файла, которым нужно настоящее окно).
    """

    def test_within_range_is_unchanged(self):
        self.assertEqual(overlay_module._clamp01(0.42), 0.42)

    def test_below_zero_is_clamped_to_zero(self):
        self.assertEqual(overlay_module._clamp01(-1.0), 0.0)

    def test_above_one_is_clamped_to_one(self):
        self.assertEqual(overlay_module._clamp01(5.0), 1.0)

    def test_non_numeric_falls_back_to_idle_alpha(self):
        self.assertEqual(overlay_module._clamp01("не число"), overlay_module.IDLE_ALPHA)

    def test_none_falls_back_to_idle_alpha(self):
        self.assertEqual(overlay_module._clamp01(None), overlay_module.IDLE_ALPHA)


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class OverlayAlphaSettingTests(unittest.TestCase):
    """
    Настраиваемая прозрачность рамки вне момента озвучки (ползунок
    «Прозрачность рамки» в окне программы, см. gui.py и config.overlay_alpha):
    конструктор принимает начальное значение, а set_idle_alpha() меняет его
    «вживую», с учётом того, в каком состоянии рамка находится прямо сейчас.
    """

    def tearDown(self):
        try:
            for after_id in self.root.tk.eval("after info").split():
                self.root.after_cancel(after_id)
        except (tkinter.TclError, AttributeError):
            pass
        try:
            self.overlay.destroy()
            self.root.destroy()
        except (tkinter.TclError, AttributeError):
            pass

    def _make_overlay(self, idle_alpha=None):
        # idle_alpha=None, а не =IDLE_ALPHA: значения по умолчанию аргументов
        # вычисляются в момент определения функции, то есть при самом импорте
        # модуля — раньше, чем успевает сработать @unittest.skipUnless на
        # строке 199. В headless-окружении IDLE_ALPHA никогда не импортирован
        # (см. блок if _DISPLAY_AVAILABLE выше), и такой default уронил бы
        # весь модуль NameError'ом при импорте вместо того, чтобы тесты этого
        # класса были просто пропущены. Резолвим реальное значение здесь,
        # внутри тела метода, — сюда код попадает, только когда дисплей уже
        # есть и IDLE_ALPHA гарантированно доступен.
        if idle_alpha is None:
            idle_alpha = IDLE_ALPHA
        self.root = ctk.CTk()
        self.root.withdraw()
        self.overlay = SelectionOverlay(self.root, (10, 10, 300, 100), index=0, idle_alpha=idle_alpha)
        self.overlay.show()
        self.root.update()
        # См. комментарий в SetReadingTests.setUp — тот же приём для той же
        # особенности некоторых X11-окружений без compositor'а.
        self.overlay._set_alpha(idle_alpha)
        self.root.update()
        return self.overlay

    def test_constructor_applies_the_given_idle_alpha(self):
        overlay = self._make_overlay(idle_alpha=0.65)
        self.assertAlmostEqual(overlay.attributes("-alpha"), 0.65, places=6)

    def test_constructor_clamps_an_out_of_range_value(self):
        overlay = self._make_overlay(idle_alpha=3.0)
        self.assertAlmostEqual(overlay.attributes("-alpha"), 1.0, places=6)

    def test_set_idle_alpha_applies_immediately_in_idle_state(self):
        overlay = self._make_overlay()
        overlay.set_idle_alpha(0.0)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), 0.0, places=6)
        self.assertTrue(overlay.is_visible(), "0.0 делает рамку невидимой, но не withdraw()")

    def test_set_idle_alpha_is_clamped(self):
        overlay = self._make_overlay()
        overlay.set_idle_alpha(-2.0)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), 0.0, places=6)

    def test_set_idle_alpha_while_reading_does_not_change_visible_alpha_yet(self):
        overlay = self._make_overlay()
        overlay.set_reading(True)
        self.root.update()
        overlay.set_idle_alpha(0.9)
        self.root.update()
        # Пока идёт чтение, видимый эффект — всё ещё READING_ALPHA (рамка не
        # видна), а не только что установленное значение.
        self.assertAlmostEqual(overlay.attributes("-alpha"), READING_ALPHA, places=6)

    def test_new_idle_alpha_applies_once_reading_ends(self):
        overlay = self._make_overlay()
        overlay.set_reading(True)
        self.root.update()
        overlay.set_idle_alpha(0.9)
        overlay.set_reading(False)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), 0.9, places=6)

    def test_set_idle_alpha_while_dragging_does_not_change_visible_alpha_yet(self):
        overlay = self._make_overlay()
        overlay._mode = "move"
        overlay._set_alpha(ACTIVE_ALPHA)
        self.root.update()
        overlay.set_idle_alpha(0.9)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), ACTIVE_ALPHA, places=6)
        overlay._mode = ""

    def test_new_idle_alpha_applies_on_release(self):
        overlay = self._make_overlay()
        overlay._mode = "move"
        overlay._set_alpha(ACTIVE_ALPHA)
        self.root.update()
        overlay.set_idle_alpha(0.9)
        overlay._on_release(None)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), 0.9, places=6)

    def test_release_while_reading_keeps_the_overlay_invisible(self):
        # Если пользователь как-то умудрился отпустить кнопку мыши именно в
        # момент, когда область уже снова читается (редкий, но возможный по
        # времени случай) — отпускание не должно «выключить» READING_ALPHA.
        overlay = self._make_overlay()
        overlay._reading = True
        overlay._mode = "move"
        overlay._set_alpha(ACTIVE_ALPHA)
        self.root.update()
        overlay._on_release(None)
        self.root.update()
        self.assertAlmostEqual(overlay.attributes("-alpha"), READING_ALPHA, places=6)


@unittest.skipUnless(_DISPLAY_AVAILABLE, "нет графического дисплея — тест окна пропущен")
class ExcludeFromCaptureTests(unittest.TestCase):
    """
    _exclude_from_capture() — это необязательное улучшение (исключает рамку из
    снимка mss на Windows 10 2004+/11, см. overlay._WDA_EXCLUDEFROMCAPTURE). На
    любой другой платформе он обязан тихо ничего не делать, а на самой Windows —
    пытаться вызвать нужный Win32 API и не падать, даже если тот ведёт себя
    неожиданно.

    Этот тестовый набор (как и весь проект) реально гоняется и на Linux, и на
    Windows — см. .github/workflows/build.yml. Поэтому обе ветки платформы
    здесь проверяются ЯВНОЙ подменой overlay.sys.platform и overlay.ctypes.windll
    (см. _simulated_platform выше), а не полагаются на то, какая платформа в
    данный момент реально запускает тесты: прежняя версия этого теста как раз
    на этом и спотыкалась (успешно работала здесь, на Linux, но была написана
    так, что обязана была провалиться на самой Windows — ровно там, где эта
    защита и должна быть проверена жёстче всего).
    """

    def setUp(self):
        self.root = ctk.CTk()
        self.root.withdraw()
        self.overlay = SelectionOverlay(self.root, (10, 10, 300, 100), index=0)

    def tearDown(self):
        try:
            self.overlay.destroy()
            self.root.destroy()
        except tkinter.TclError:
            pass

    def test_no_op_on_non_windows_platform(self):
        # Платформа явно НЕ "win32" — функция обязана вернуться сразу же, даже
        # не заглядывая в ctypes.windll (на реальном Linux/macOS его вообще нет).
        with _simulated_platform("darwin"):
            self.overlay._exclude_from_capture()

    def test_calls_the_windows_api_with_the_right_flag(self):
        user32 = _FakeUser32(set_display_affinity_results=(1,))
        with _simulated_platform("win32", user32):
            self.overlay._exclude_from_capture()
        self.assertEqual(len(user32.set_display_affinity_calls), 1)
        _hwnd, flag = user32.set_display_affinity_calls[0]
        self.assertEqual(flag, overlay_module._WDA_EXCLUDEFROMCAPTURE)
        self.assertEqual(user32.get_parent_calls, [], "GetParent не нужен, если первая попытка удалась")

    def test_falls_back_to_get_parent_when_the_direct_call_fails(self):
        user32 = _FakeUser32(set_display_affinity_results=(0, 1), get_parent_result=4242)
        with _simulated_platform("win32", user32):
            self.overlay._exclude_from_capture()
        self.assertEqual(len(user32.set_display_affinity_calls), 2, "должна быть вторая попытка через GetParent")
        self.assertEqual(user32.get_parent_calls, [user32.set_display_affinity_calls[0][0]])
        self.assertEqual(user32.set_display_affinity_calls[1][0], 4242)

    def test_does_not_call_get_parent_when_get_parent_itself_returns_falsy(self):
        # GetParent может вернуть 0 (нет родителя) — тогда повторной попытки
        # быть не должно, а не падать с бессмысленным SetWindowDisplayAffinity(0, ...).
        user32 = _FakeUser32(set_display_affinity_results=(0,), get_parent_result=0)
        with _simulated_platform("win32", user32):
            self.overlay._exclude_from_capture()
        self.assertEqual(len(user32.set_display_affinity_calls), 1)
        self.assertEqual(user32.get_parent_calls, [user32.set_display_affinity_calls[0][0]])

    def test_any_exception_from_the_api_is_swallowed_quietly(self):
        with _simulated_platform("win32", _BoomingUser32()):
            self.overlay._exclude_from_capture()  # не должен поднять исключение наружу


if __name__ == "__main__":
    unittest.main()
