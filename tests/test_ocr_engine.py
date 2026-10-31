# -*- coding: utf-8 -*-
"""
Тесты downscale_for_easyocr — дополнительного уменьшения кадра
непосредственно перед передачей в EasyOCR (см. подробный комментарий к
config.EASYOCR_DOWNSCALE_FACTOR и докстринг самой функции в ocr_engine.py).

Сам класс EasyOcrEngine здесь не тестируется — он требует установленную
библиотеку easyocr с её тяжёлыми зависимостями (torch и т.п.), которых в
окружении для быстрых тестов проекта нет. downscale_for_easyocr — чистая,
не зависящая от самого движка функция, и её поведение проверяется
изолированно.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ocr_engine import downscale_for_easyocr  # noqa: E402


def _image(height, width):
    """Одноканальное (как после capture.preprocess_for_ocr) изображение с
    узнаваемым градиентом — чтобы отличить реальное уменьшение от, например,
    случайно обрезанного или залитого одним цветом результата."""
    row = np.linspace(0, 255, num=width, dtype=np.uint8)
    return np.tile(row, (height, 1))


class DownscaleForEasyocrTests(unittest.TestCase):
    def test_factor_one_returns_image_unchanged(self):
        image = _image(200, 2000)
        result = downscale_for_easyocr(image, 1.0)
        self.assertIs(result, image)   # та же ссылка — копирования не было вовсе

    def test_factor_below_one_returns_image_unchanged(self):
        # factor < 1.0 означало бы УВЕЛИЧЕНИЕ — это не входит в задачу функции
        # (увеличением занимается отдельно capture.preprocess_for_ocr), любое
        # значение не больше 1.0 трактуется как «доп. уменьшение отключено».
        image = _image(200, 2000)
        result = downscale_for_easyocr(image, 0.5)
        self.assertIs(result, image)

    def test_factor_above_one_shrinks_each_side_proportionally(self):
        image = _image(300, 2250)   # близко к реальному размеру области субтитров
        result = downscale_for_easyocr(image, 1.5)
        self.assertEqual(result.shape, (200, 1500))   # 300/1.5=200, 2250/1.5=1500

    def test_result_stays_single_channel(self):
        image = _image(248, 1328)
        result = downscale_for_easyocr(image, 1.5)
        self.assertEqual(result.ndim, 2)   # как и вход — одноканальное изображение
        self.assertEqual(result.dtype, image.dtype)

    def test_default_project_factor_matches_config(self):
        import config
        image = _image(248, 1328)
        result = downscale_for_easyocr(image, config.EASYOCR_DOWNSCALE_FACTOR)
        expected_h = round(248 / config.EASYOCR_DOWNSCALE_FACTOR)
        expected_w = round(1328 / config.EASYOCR_DOWNSCALE_FACTOR)
        self.assertEqual(result.shape, (expected_h, expected_w))
        # Текущее значение по умолчанию (см. комментарий в config.py) — 1.5:
        # явная проверка на случай, если кто-то случайно изменит константу,
        # не обновив сопровождающий её комментарий и расчёты в нём.
        self.assertEqual(config.EASYOCR_DOWNSCALE_FACTOR, 1.5)

    def test_very_small_image_does_not_collapse_to_zero(self):
        # На всякий случай: если на вход попадёт совсем маленькое изображение
        # (например, крошечная тестовая область), результат не должен
        # схлопнуться до 0 по какой-либо стороне — cv2.resize с нулевым
        # размером стороны падает с ошибкой.
        image = _image(2, 2)
        result = downscale_for_easyocr(image, 3.0)
        self.assertGreaterEqual(result.shape[0], 1)
        self.assertGreaterEqual(result.shape[1], 1)

    def test_shrunk_image_is_smaller_in_byte_size(self):
        # Содержательная проверка эффекта, а не только геометрии: уменьшенное
        # изображение занимает заметно меньше памяти — а значит и свёрточной
        # сети EasyOCR предстоит обработать заметно меньше данных.
        image = _image(300, 2250)
        result = downscale_for_easyocr(image, 1.5)
        self.assertLess(result.nbytes, image.nbytes // 2)


if __name__ == "__main__":
    unittest.main()
