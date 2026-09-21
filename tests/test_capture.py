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


if __name__ == "__main__":
    unittest.main()
