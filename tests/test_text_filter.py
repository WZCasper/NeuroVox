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

    def test_drops_glued_together_ocr_garbage(self):
        # Реальный пример из журнала пользователя: несколько слов субтитра,
        # которые OCR слепил в одно без единого пробела, — такую «кашу»
        # Silero озвучил бы неразборчивым набором звуков.
        self.assertEqual(clean_text("почемубьтебененрисестьменадолго"), "")

    def test_keeps_long_real_russian_word(self):
        # Самые длинные настоящие русские слова не должны задеваться
        # порогом _MAX_SINGLE_WORD_LETTERS, рассчитанным на то, чтобы ловить
        # именно склейку НЕСКОЛЬКИХ слов, а не редкое длинное слово.
        self.assertEqual(clean_text("человеконенавистничество"), "человеконенавистничество")

    def test_keeps_natural_phrase_with_many_short_words(self):
        # Диалоговые фразы с обилием коротких местоимений и союзов —
        # обычное дело в субтитрах и не должны ложно приниматься за мусор.
        text = "Он и она, ты и я, мы и они — вот и всё, что есть у нас"
        self.assertEqual(clean_text(text), text)


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

    def test_same_phrase_after_forget_is_spoken_again(self):
        # Область субтитров опустела (реплика закончилась) — в реальном
        # конвейере (workers.OcrWorker) об этом сообщает не пустая строка в
        # process(), а отдельный вызов forget(), когда снимок помечен как
        # пустой (frame.is_marker). Та же фраза после этого — новое появление.
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        frames_before = ["Беги отсюда"] * 3
        frames_after = ["Беги отсюда"] * 3
        spoken = [p for p in (flt.process(f) for f in frames_before) if p]
        flt.forget()
        spoken += [p for p in (flt.process(f) for f in frames_after) if p]
        self.assertEqual(spoken, ["Беги отсюда", "Беги отсюда"])

    def test_same_phrase_without_forget_is_not_repeated(self):
        # Без forget() между повторами — это один и тот же, ещё не
        # закончившийся показ реплики, повторно озвучивать не нужно.
        frames = ["Беги отсюда"] * 3 + ["Беги отсюда"] * 3
        self.assertEqual(self._spoken(frames), ["Беги отсюда"])

    def test_empty_string_is_safe_and_does_not_forget_last_spoken(self):
        # process() с пустой строкой больше не пытается распознать «реплика
        # закончилась» (это больше не происходит в текущем конвейере — см.
        # docstring process()), но не должен падать и не должен сам по себе
        # сбрасывать last_spoken — это теперь исключительно работа forget().
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertIsNotNone(flt.process("Беги отсюда", settled=True))
        for _ in range(10):
            self.assertIsNone(flt.process(""))
        # Без forget() та же фраза всё ещё считается тем же показом реплики.
        self.assertIsNone(flt.process("Беги отсюда", settled=True))


class SettledFrameTests(unittest.TestCase):
    """Кадры, которые уже признаны устоявшимися по картинке, повторно подтверждать не нужно."""

    def test_settled_text_is_spoken_immediately(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertEqual(flt.process("Куда ты пропал?", settled=True), "Куда ты пропал?")

    def test_settled_repeat_is_not_spoken_twice(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertIsNotNone(flt.process("Куда ты пропал?", settled=True))
        self.assertIsNone(flt.process("Куда ты пропал?", settled=True))

    def test_different_settled_phrases_are_all_spoken(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        spoken = [flt.process(text, settled=True) for text in ("Первая реплика", "Вторая реплика героя", "Третья реплика")]
        self.assertEqual([s for s in spoken if s], ["Первая реплика", "Вторая реплика героя", "Третья реплика"])

    def test_unsettled_text_still_needs_confirmation(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertIsNone(flt.process("Куда ты пропал?", settled=False))
        self.assertEqual(flt.process("Куда ты пропал?", settled=False), "Куда ты пропал?")

    def test_settled_noise_or_junk_is_still_dropped(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertIsNone(flt.process("12:45", settled=True))
        self.assertIsNone(flt.process("", settled=True))

    def test_forget_allows_the_same_phrase_again(self):
        flt = SubtitleFilter(threshold=85, stable_frames=2)
        self.assertIsNotNone(flt.process("Беги отсюда", settled=True))
        self.assertIsNone(flt.process("Беги отсюда", settled=True))
        flt.forget()
        self.assertEqual(flt.process("Беги отсюда", settled=True), "Беги отсюда")


if __name__ == "__main__":
    unittest.main()
