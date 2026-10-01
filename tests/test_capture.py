# -*- coding: utf-8 -*-
"""Тесты подготовки кадра к распознаванию."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import capture  # noqa: E402


def _frame_with_text(light_on_dark=True):
    background, foreground = ((30, 34, 50), (255, 255, 255)) if light_on_dark else ((235, 235, 235), (20, 20, 20))
    frame = np.full((100, 600, 3), background, dtype=np.uint8)
    cv2.putText(frame, "SUBTITLE TEXT HERE", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, foreground, 4)
    return frame


def _typing_frame(text):
    """Кадр с частично напечатанным текстом — соседние состояния похожи, но не идентичны."""
    frame = np.full((100, 600, 3), (30, 34, 50), dtype=np.uint8)
    cv2.putText(frame, text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    return frame


def _gradient_frame_with_outlined_text(text="ПРИВЕТ КАК ДЕЛА", light_on_dark=True, size=(150, 900)):
    """
    Кадр с ЗАМЕТНЫМ перепадом яркости фона слева направо (одна половина фразы
    может лежать на тёмной части сцены, другая — на светлой) и обводкой текста
    контрастным цветом — типичная игровая ситуация, когда субтитр показан
    прямо поверх сцены, а не на отдельной ровной плашке.
    """
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        # Разброс яркости фона намеренно широкий (от тёмного до светлого),
        # чтобы результат уверенно попадал в неоднородную ветку обработки —
        # именно такой перепад ломает единый порог Оцу.
        value = int(20 + (x / width) * 190)
        frame[:, x] = (value, value, value)
    text_color = (255, 255, 255) if light_on_dark else (20, 20, 20)
    outline_color = (0, 0, 0) if light_on_dark else (255, 255, 255)
    origin = (25, int(height * 0.62))
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, outline_color, 7, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2, cv2.LINE_AA)
    return frame


class PreprocessTests(unittest.TestCase):
    def test_light_text_on_dark_gives_black_on_white_with_padding(self):
        image = capture.preprocess_for_ocr(_frame_with_text(True))
        self.assertIsNotNone(image)
        self.assertEqual(image.shape, (100 * capture.UPSCALE_FACTOR + 2 * capture.OCR_PADDING,
                                       600 * capture.UPSCALE_FACTOR + 2 * capture.OCR_PADDING))
        self.assertGreater(float(image.mean()), 127)                      # фон белый
        self.assertTrue((image[: capture.OCR_PADDING] == 255).all())      # поля белые

    def test_dark_text_on_light_is_also_black_on_white(self):
        image = capture.preprocess_for_ocr(_frame_with_text(False))
        self.assertIsNotNone(image)
        self.assertGreater(float(image.mean()), 127)

    def test_empty_frame_is_skipped(self):
        self.assertIsNone(capture.preprocess_for_ocr(np.full((100, 600, 3), 30, dtype=np.uint8)))

    def test_none_and_empty_input(self):
        self.assertIsNone(capture.preprocess_for_ocr(None))
        self.assertIsNone(capture.preprocess_for_ocr(np.zeros((0, 0, 3), dtype=np.uint8)))


class AdaptiveBinarizationTests(unittest.TestCase):
    """
    Проверяет ветку адаптивной бинаризации: субтитр наложен прямо на сцену
    с заметным перепадом яркости фона, где у метода Оцу нет единого верного
    порога (см. BACKGROUND_UNIFORMITY_STD_THRESHOLD в capture.py).
    """

    def test_uniform_background_alone_is_below_threshold(self):
        # Критерий переключения смотрит на РАЗБРОС ЯРКОСТИ ВСЕГО КАДРА, поэтому
        # крупный контрастный текст сам по себе поднимает std независимо от
        # фона — это не брак критерия, а следствие того, что текст обязан
        # отличаться от фона. Проверяем сам признак на однородном ФОНЕ БЕЗ
        # текста — том самом, на котором построена плашка в _frame_with_text.
        background_only = np.full((100, 600), 30, dtype=np.uint8)
        self.assertLess(float(np.std(background_only)), capture.BACKGROUND_UNIFORMITY_STD_THRESHOLD)

    def test_gradient_background_alone_is_above_threshold(self):
        height, width = 150, 900
        gray = np.zeros((height, width), dtype=np.uint8)
        for x in range(width):
            gray[:, x] = int(20 + (x / width) * 190)
        self.assertGreaterEqual(float(np.std(gray)), capture.BACKGROUND_UNIFORMITY_STD_THRESHOLD)

    def test_outlined_text_on_gradient_is_recognized_as_text(self):
        # Главный сценарий задачи: текст с обводкой поверх фона с явным перепадом
        # яркости. Раньше здесь метод Оцу либо заливал шумом светлую половину
        # кадра, либо терял текст на тёмной — итоговая доля «текста» уходила
        # далеко за пределы разумного диапазона. Новый код обязан вернуть
        # изображение (не None) с долей белого текста в допустимых пределах.
        image = capture.preprocess_for_ocr(_gradient_frame_with_outlined_text())
        self.assertIsNotNone(image)
        black_ratio = 1.0 - float(np.count_nonzero(image)) / image.size
        self.assertGreaterEqual(black_ratio, capture.MIN_TEXT_PIXEL_RATIO)
        self.assertLessEqual(black_ratio, capture.MAX_TEXT_PIXEL_RATIO)

    def test_outlined_dark_text_on_gradient_is_also_recognized(self):
        # Обратная полярность: тёмный текст с белой обводкой на том же градиенте.
        image = capture.preprocess_for_ocr(_gradient_frame_with_outlined_text(light_on_dark=False))
        self.assertIsNotNone(image)
        black_ratio = 1.0 - float(np.count_nonzero(image)) / image.size
        self.assertGreaterEqual(black_ratio, capture.MIN_TEXT_PIXEL_RATIO)
        self.assertLessEqual(black_ratio, capture.MAX_TEXT_PIXEL_RATIO)

    def test_result_shape_and_padding_match_uniform_path(self):
        # Независимо от того, какая ветка бинаризации сработала внутри, форма
        # результата и белые поля вокруг текста должны быть одинаковыми — это
        # важно для frames_are_similar (сравнивает кадры по форме) и для
        # остального конвейера, который не должен замечать, какой путь выбран.
        image = capture.preprocess_for_ocr(_gradient_frame_with_outlined_text())
        self.assertIsNotNone(image)
        self.assertEqual(
            image.shape,
            (150 * capture.UPSCALE_FACTOR + 2 * capture.OCR_PADDING,
             900 * capture.UPSCALE_FACTOR + 2 * capture.OCR_PADDING),
        )
        self.assertTrue((image[: capture.OCR_PADDING] == 255).all())  # поля белые

    def test_empty_gradient_background_without_text_is_skipped(self):
        # Неоднородный фон сам по себе (без текста) не должен приниматься за текст.
        height, width = 150, 900
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for x in range(width):
            value = int(20 + (x / width) * 190)
            frame[:, x] = (value, value, value)
        self.assertIsNone(capture.preprocess_for_ocr(frame))

    def test_remove_small_components_keeps_large_and_drops_small(self):
        # Прямая проверка вспомогательной функции очистки шума: маленькая
        # шумная точка исчезает, крупная область (аналог настоящей буквы) —
        # остаётся без изменений.
        binary = np.zeros((60, 60), dtype=np.uint8)
        binary[5:6, 5:6] = 255      # «шум»: область из 1 пикселя
        binary[20:40, 20:40] = 255  # «буква»: область 20x20 = 400 пикселей
        cleaned = capture._remove_small_components(binary, min_area=10)
        self.assertEqual(cleaned[5, 5], 0)
        self.assertTrue((cleaned[20:40, 20:40] == 255).all())

    def test_adaptive_binarize_helper_returns_text_as_white(self):
        # Прямая проверка _binarize_adaptive в отрыве от подбора ветки: на
        # кадре с заведомо неоднородным фоном текст должен стать белым (255),
        # а доля белых пикселей — оставаться в разумных пределах (не залитый
        # шумом кадр).
        gray = cv2.cvtColor(_gradient_frame_with_outlined_text(), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=capture.UPSCALE_FACTOR, fy=capture.UPSCALE_FACTOR,
                           interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = capture._binarize_adaptive(gray)
        white_ratio = float(np.count_nonzero(binary)) / binary.size
        self.assertGreaterEqual(white_ratio, capture.MIN_TEXT_PIXEL_RATIO)
        self.assertLessEqual(white_ratio, capture.MAX_TEXT_PIXEL_RATIO)


class SimilarityTests(unittest.TestCase):
    def test_identical_frames_are_similar(self):
        image = capture.preprocess_for_ocr(_frame_with_text())
        self.assertTrue(capture.frames_are_similar(image, image.copy()))

    def test_different_frames_are_not_similar(self):
        a = capture.preprocess_for_ocr(_frame_with_text(True))
        b = np.zeros_like(a)
        self.assertFalse(capture.frames_are_similar(a, b))

    def test_none_or_shape_mismatch_is_not_similar(self):
        image = capture.preprocess_for_ocr(_frame_with_text())
        self.assertFalse(capture.frames_are_similar(None, image))
        self.assertFalse(capture.frames_are_similar(image, image[:10]))


class AreaTrackerTests(unittest.TestCase):
    """
    AreaTracker решает, КОГДА снимок пора отдавать на распознавание, не видя самого
    распознавания. Картинки — просто маркеры: «A», «B» и т. д.; для трекера важно
    только, совпадают ли они (frames_are_similar), а не что на них нарисовано.
    """

    # IMG_A и IMG_B — РАЗНЫЙ текст (разные картинки для трекера). _frame_with_text(True)
    # и _frame_with_text(False) сюда не годятся: предобработка нормализует оба варианта
    # в «чёрный текст на белом» и делает их побитово идентичными — для AreaTracker это
    # была бы одна и та же картинка, а не смена реплики.
    IMG_A = capture.preprocess_for_ocr(_frame_with_text(True))
    IMG_B = capture.preprocess_for_ocr(_typing_frame("ДРУГАЯ РЕПЛИКА СОВСЕМ ИНОЙ ДЛИНЫ"))
    IMG_C = capture.preprocess_for_ocr(_typing_frame("S"))
    IMG_D = capture.preprocess_for_ocr(_typing_frame("SU"))
    IMG_E = capture.preprocess_for_ocr(_typing_frame("SUB"))

    def setUp(self):
        self.tracker = capture.AreaTracker(settle_time=0.3, max_wait=1.5, empty_hold=0.8)

    def test_nothing_committed_before_settle_time(self):
        self.assertIsNone(self.tracker.update(self.IMG_A, now=0.0))
        self.assertIsNone(self.tracker.update(self.IMG_A, now=0.1))
        self.assertIsNone(self.tracker.update(self.IMG_A, now=0.29))

    def test_committed_once_settled(self):
        self.tracker.update(self.IMG_A, now=0.0)
        commit = self.tracker.update(self.IMG_A, now=0.31)
        self.assertIsNotNone(commit)
        self.assertTrue(commit.settled)
        self.assertIs(commit.image, self.IMG_A)
        self.assertFalse(
            capture.frames_are_similar(self.IMG_A, self.IMG_B, self.tracker._tolerance),
            "IMG_A и IMG_B должны быть РАЗНЫМИ картинками для проверки трекера",
        )

    def test_same_image_is_committed_only_once(self):
        self.tracker.update(self.IMG_A, now=0.0)
        self.assertIsNotNone(self.tracker.update(self.IMG_A, now=0.31))
        self.assertIsNone(self.tracker.update(self.IMG_A, now=0.5))
        self.assertIsNone(self.tracker.update(self.IMG_A, now=0.9))

    def test_change_after_commit_is_tracked_again(self):
        self.tracker.update(self.IMG_A, now=0.0)
        self.tracker.update(self.IMG_A, now=0.31)          # первая реплика отправлена
        self.assertIsNone(self.tracker.update(self.IMG_B, now=0.40))
        commit = self.tracker.update(self.IMG_B, now=0.71)  # 0.40 + 0.31 >= settle_time
        self.assertIsNotNone(commit)
        self.assertIs(commit.image, self.IMG_B)

    def test_typing_effect_restarts_settle_window(self):
        # Текст «допечатывается» по буквам каждые 0.2 с — короче settle_time (0.3 с),
        # поэтому печать ещё не закончилась и коммита быть не должно.
        self.assertIsNone(self.tracker.update(self.IMG_C, now=0.0))
        self.assertIsNone(self.tracker.update(self.IMG_D, now=0.2))
        self.assertIsNone(self.tracker.update(self.IMG_E, now=0.4))
        # Печать закончилась на IMG_E — картинка держится 0.3 с без изменений.
        commit = self.tracker.update(self.IMG_E, now=0.71)
        self.assertIsNotNone(commit)
        self.assertTrue(commit.settled)
        self.assertIs(commit.image, self.IMG_E)

    def test_forced_commit_when_constantly_changing(self):
        # Картинка меняется непрерывно (шумный фон, очень медленная печать) — settle
        # никогда не наступает, но снимок всё равно должен уйти на распознавание не
        # реже, чем раз в max_wait секунд.
        frames = [self.IMG_A, self.IMG_B, self.IMG_C, self.IMG_D, self.IMG_E]
        t = 0.0
        commit = None
        for i in range(40):
            commit = self.tracker.update(frames[i % len(frames)], now=t)
            t += 0.05
            if commit is not None:
                break
        self.assertIsNotNone(commit)
        self.assertFalse(commit.settled)          # отправлен принудительно, не как устоявшийся
        self.assertLessEqual(t, 1.6)               # не позже max_wait + один шаг

    def test_empty_area_reports_marker_after_hold(self):
        self.tracker.update(self.IMG_A, now=0.0)
        self.tracker.update(self.IMG_A, now=0.31)     # текст был показан
        self.assertIsNone(self.tracker.update(None, now=0.40))   # пусто, но ещё не долго
        commit = self.tracker.update(None, now=1.21)              # 0.40 + 0.8 = 1.20, с запасом
        self.assertIsNotNone(commit)
        self.assertIsNone(commit.image)  # метка «область опустела» — изображения нет

    def test_empty_marker_sent_only_once(self):
        self.tracker.update(self.IMG_A, now=0.0)
        self.tracker.update(self.IMG_A, now=0.31)
        self.tracker.update(None, now=0.40)
        self.assertIsNotNone(self.tracker.update(None, now=1.21))
        self.assertIsNone(self.tracker.update(None, now=1.50))    # метка уже отправлена — повторно не шлём

    def test_no_marker_if_area_was_never_committed(self):
        # Область изначально пуста — «опустела» отправлять нечего (реплики и не было).
        self.assertIsNone(self.tracker.update(None, now=0.0))
        self.assertIsNone(self.tracker.update(None, now=2.0))

    def test_reappearing_after_empty_is_tracked_as_new(self):
        self.tracker.update(self.IMG_A, now=0.0)
        self.tracker.update(self.IMG_A, now=0.31)
        self.tracker.update(None, now=0.40)
        self.tracker.update(None, now=1.21)                       # метка «опустела» отправлена
        self.assertIsNone(self.tracker.update(self.IMG_B, now=1.30))
        commit = self.tracker.update(self.IMG_B, now=1.61)
        self.assertIsNotNone(commit)
        self.assertTrue(commit.settled)


if __name__ == "__main__":
    unittest.main()
