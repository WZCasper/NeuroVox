# -*- coding: utf-8 -*-
"""Тесты очистки текста и фильтра повторов."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_filter import SubtitleFilter, clean_text, similarity  # noqa: E402


class CleanTextTests(unittest.TestCase):
    def test_removes_ui_junk(self):
        self.assertEqual(clean_text("Привет ] | как дела"), "Привет как дела")

    def test_fixes_latin_lookalikes_inside_russian_word(self):
        self.assertEqual(clean_text("Пpивет, мир"), "Привет, мир")

    def test_keeps_pure_english_text(self):
        self.assertEqual(clean_text("Hello there"), "Hello there")

    def test_drops_timers_and_numbers(self):
        self.assertEqual(clean_text("12:45"), "")

    def test_drops_too_short_text(self):
        self.assertEqual(clean_text("а"), "")

    def test_collapses_repeated_punctuation(self):
        self.assertEqual(clean_text("Стой!!! Не двигайся"), "Стой! Не двигайся")

    def test_empty_and_blank(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   "), "")


class SimilarityTests(unittest.TestCase):
    def test_same_text_with_punctuation_is_identical(self):
        self.assertGreaterEqual(similarity("Куда ты пропал? Мы искали тебя.", "Куда ты пропал Мы искали тебя"), 85)

    def test_different_texts_are_far_apart(self):
        self.assertLess(similarity("Куда ты пропал?", "Совсем другая фраза здесь"), 85)

    def test_ignores_case_and_yo(self):
        self.assertGreaterEqual(similarity("Привет, Ёжик", "привет ежик"), 95)


class SubtitleFilterTests(unittest.TestCase):
    @staticmethod
    def _spoken(frames, **kwargs):
        flt = SubtitleFilter(threshold=85, stable_frames=2, **kwargs)
        return [phrase for phrase in (flt.process(frame) for frame in frames) if phrase]

    def test_stable_phrase_is_spoken_once(self):
        self.assertEqual(self._spoken(["Куда ты пропал?"] * 6), ["Куда ты пропал?"])

    def test_typing_effect_does_not_speak_fragments(self):
        frames = ["Куда", "Куда ты", "Куда ты проп", "Куда ты пропал?", "Куда ты пропал?", "Куда ты пропал?"]
        self.assertEqual(self._spoken(frames), ["Куда ты пропал?"])

    def test_ocr_noise_does_not_create_duplicates(self):
        frames = ["Не оборачивайся, они рядом"] * 2 + ["Не оборачивайся они радом"] * 3 + ["Не оборачивайся, они рядом"] * 3
        self.assertEqual(len(self._spoken(frames)), 1)

    def test_two_different_phrases_are_both_spoken(self):
        frames = ["Первая реплика героя"] * 3 + [""] * 2 + ["Совершенно другой ответ"] * 3
        self.assertEqual(len(self._spoken(frames)), 2)

    def test_same_phrase_after_pause_is_spoken_again(self):
        frames = ["Беги отсюда"] * 3 + [""] * 3 + ["Беги отсюда"] * 3
        self.assertEqual(self._spoken(frames), ["Беги отсюда", "Беги отсюда"])

    def test_same_phrase_without_pause_is_not_repeated(self):
        frames = ["Беги отсюда"] * 3 + [""] * 1 + ["Беги отсюда"] * 3
        self.assertEqual(self._spoken(frames), ["Беги отсюда"])


if __name__ == "__main__":
    unittest.main()
