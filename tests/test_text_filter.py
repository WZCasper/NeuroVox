# -*- coding: utf-8 -*-
"""Тесты очистки текста и фильтра повторов."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pronunciation import SpeakerNameList  # noqa: E402
from text_filter import SubtitleFilter, clean_text, strip_speaker_name, similarity  # noqa: E402


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


class StripSpeakerNameTests(unittest.TestCase):
    """
    strip_speaker_name() и его интеграция в clean_text() — отсечение имени
    говорящего в начале реплики («Хозия: Несите его сюда!»), см. docstring
    strip_speaker_name в text_filter.py и config.SPEAKER_NAME_MODES.
    """

    def test_off_mode_does_not_touch_text(self):
        self.assertEqual(strip_speaker_name("Хозия: Несите его сюда!", "off"), "Хозия: Несите его сюда!")

    def test_auto_mode_strips_a_name_before_colon(self):
        self.assertEqual(strip_speaker_name("Хозия: Несите его сюда!", "auto"), "Несите его сюда!")

    def test_auto_mode_strips_a_name_before_dash(self):
        self.assertEqual(strip_speaker_name("Датч — Ты нам нужен сильным.", "auto"), "Ты нам нужен сильным.")

    def test_auto_mode_strips_a_two_word_name(self):
        self.assertEqual(
            strip_speaker_name("Артур Морган: Ладно, идём.", "auto"), "Ладно, идём."
        )

    def test_auto_mode_also_strips_a_non_name_word_by_the_same_shape(self):
        # Задокументированный компромисс режима "auto": формально неотличимо
        # от настоящего имени (см. docstring strip_speaker_name и config.py),
        # поэтому тест фиксирует это поведение как ожидаемое, а не как баг.
        self.assertEqual(
            strip_speaker_name("Запомни: никогда не доверяй чужакам.", "auto"),
            "никогда не доверяй чужакам.",
        )

    def test_auto_mode_leaves_text_without_a_name_shaped_prefix_untouched(self):
        self.assertEqual(strip_speaker_name("Беги, у нас мало времени!", "auto"), "Беги, у нас мало времени!")

    def test_auto_mode_does_not_strip_lowercase_prefix(self):
        # Имя всегда с заглавной буквы — иначе это не похоже на имя вовсе.
        self.assertEqual(strip_speaker_name("просто: текст", "auto"), "просто: текст")

    def test_list_mode_strips_a_known_name(self):
        names = SpeakerNameList(path=None)
        names._names = {"хозия", "датч"}   # минуя файл — тестируем только логику сравнения
        self.assertEqual(strip_speaker_name("Хозия: Несите его сюда!", "list", names), "Несите его сюда!")

    def test_list_mode_leaves_an_unknown_name_untouched(self):
        names = SpeakerNameList(path=None)
        names._names = {"датч"}
        self.assertEqual(
            strip_speaker_name("Запомни: никогда не доверяй чужакам.", "list", names),
            "Запомни: никогда не доверяй чужакам.",
        )

    def test_list_mode_without_names_object_does_nothing(self):
        self.assertEqual(strip_speaker_name("Хозия: Несите его сюда!", "list", None), "Хозия: Несите его сюда!")

    def test_list_mode_name_match_is_case_insensitive(self):
        # Регистр самого ИМЕНИ в списке не важен (пользователь мог вписать
        # его строчными буквами) — но первая буква имени В ТЕКСТЕ субтитра
        # по-прежнему должна быть заглавной, как и пишут имена говорящих на
        # практике (см. все примеры на скриншотах пользователя); test_auto_
        # mode_does_not_strip_lowercase_prefix выше фиксирует эту же границу
        # для режима "auto".
        # Нормализация регистра имени в самом списке — ответственность
        # _reload_if_changed() при чтении файла (см. SpeakerNameListFileTests
        # в этом файле, которые проверяют именно это через реальный файл);
        # здесь, подменяя _names напрямую в обход файла, кладём уже
        # нормализованное значение, как это сделал бы реальный путь чтения.
        names = SpeakerNameList(path=None)
        names._names = {"хозия"}
        self.assertEqual(strip_speaker_name("Хозия: Несите его сюда!", "list", names), "Несите его сюда!")

    def test_clean_text_applies_speaker_name_mode(self):
        self.assertEqual(clean_text("Хозия: Несите его сюда!", "auto"), "Несите его сюда!")
        self.assertEqual(clean_text("Хозия: Несите его сюда!", "off"), "Хозия: Несите его сюда!")

    def test_clean_text_default_mode_is_off(self):
        # Вызов без третьего аргумента (как во всех остальных тестах этого
        # файла) не должен менять поведение, которое было до появления этой
        # настройки.
        self.assertEqual(clean_text("Хозия: Несите его сюда!"), "Хозия: Несите его сюда!")

    def test_short_reply_after_name_is_not_dropped_as_too_short(self):
        # Имя убирается ДО проверки минимальной длины — короткая, но настоящая
        # реплика после имени («Нет.») не должна отбрасываться как короткая.
        self.assertEqual(clean_text("Датч: Нет.", "auto"), "Нет.")

    def test_subtitle_filter_uses_speaker_name_mode(self):
        flt = SubtitleFilter(threshold=85, stable_frames=1, speaker_name_mode="auto")
        self.assertEqual(flt.process("Хозия: Несите его сюда!", settled=True), "Несите его сюда!")


if __name__ == "__main__":
    unittest.main()
