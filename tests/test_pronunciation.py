# -*- coding: utf-8 -*-
"""Тесты подготовки произношения (pronunciation.py): капс и словарь ударений."""

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pronunciation import (  # noqa: E402
    StressDictionary,
    normalize_caps,
    parse_dictionary,
)


class NormalizeCapsTests(unittest.TestCase):
    def test_full_caps_sentence_is_lowercased_with_capital_start(self):
        self.assertEqual(
            normalize_caps("КУДА ТЫ ПРОПАЛ? МЫ ИСКАЛИ ТЕБЯ."),
            "Куда ты пропал? Мы искали тебя.",
        )

    def test_normal_case_is_left_untouched(self):
        self.assertEqual(normalize_caps("Куда ты пропал?"), "Куда ты пропал?")

    def test_mixed_case_is_left_untouched(self):
        # Есть хотя бы одна строчная русская буква — считаем регистр осмысленным.
        self.assertEqual(normalize_caps("США и ФБР начали Расследование"), "США и ФБР начали Расследование")

    def test_short_all_caps_acronym_is_kept(self):
        self.assertEqual(normalize_caps("ФБР"), "ФБР")
        self.assertEqual(normalize_caps("США"), "США")

    def test_all_words_short_is_kept(self):
        # Все слова короче 4 букв — типично для аббревиатур/восклицаний, не трогаем.
        self.assertEqual(normalize_caps("ДА, ЭТО Я"), "ДА, ЭТО Я")

    def test_sentence_with_long_word_is_normalized(self):
        self.assertEqual(normalize_caps("ВНИМАНИЕ!"), "Внимание!")

    def test_multiple_sentences_each_get_capital(self):
        self.assertEqual(
            normalize_caps("ОСТОРОЖНО! ВПЕРЕДИ ЗАСАДА."),
            "Осторожно! Впереди засада.",
        )

    def test_latin_text_is_untouched(self):
        self.assertEqual(normalize_caps("HELLO THERE"), "HELLO THERE")

    def test_mixed_latin_and_cyrillic_caps(self):
        # Кириллица вся заглавная и есть длинное слово -> нормализуем кириллицу,
        # латиницу (NPC) не трогаем.
        self.assertEqual(normalize_caps("ВЫ ЗДЕСЬ, NPC?"), "Вы здесь, NPC?")

    def test_empty_string(self):
        self.assertEqual(normalize_caps(""), "")

    def test_no_cyrillic_letters_at_all(self):
        self.assertEqual(normalize_caps("12345!!!"), "12345!!!")


class ParseDictionaryTests(unittest.TestCase):
    def test_simple_entries(self):
        entries, problems = parse_dictionary("геральт = гер+альт\nзамок = з+амок\n")
        self.assertEqual(entries, {"геральт": "гер+альт", "замок": "з+амок"})
        self.assertEqual(problems, [])

    def test_comments_and_blank_lines_are_skipped(self):
        entries, problems = parse_dictionary("# заголовок\n\n   \nслово = сл+ово\n# конец\n")
        self.assertEqual(entries, {"слово": "сл+ово"})
        self.assertEqual(problems, [])

    def test_missing_equals_sign_is_reported(self):
        entries, problems = parse_dictionary("это не запись словаря\n")
        self.assertEqual(entries, {})
        self.assertEqual(len(problems), 1)
        self.assertIn("строка 1", problems[0])

    def test_empty_left_or_right_side_is_reported(self):
        entries, problems = parse_dictionary("= пусто_слева\nслово_без_ударения =\n")
        self.assertEqual(entries, {})
        self.assertEqual(len(problems), 2)

    def test_key_normalization_ignores_case_and_plus(self):
        entries, _ = parse_dictionary("ГеРаЛьТ = гер+альт\n")
        self.assertIn("геральт", entries)

    def test_multi_word_key_normalizes_whitespace(self):
        entries, _ = parse_dictionary("серый   волк = серый в+олк\n")
        self.assertIn("серый волк", entries)

    def test_byte_order_mark_on_first_line_is_stripped(self):
        entries, problems = parse_dictionary("\ufeffслово = сл+ово\n")
        self.assertEqual(entries, {"слово": "сл+ово"})
        self.assertEqual(problems, [])


class StressDictionaryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "stress_dict.txt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_apply_with_no_file_returns_text_unchanged(self):
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Привет, мир"), "Привет, мир")
        self.assertEqual(len(sd), 0)

    def test_ensure_file_creates_template_with_instructions(self):
        sd = StressDictionary(self.path)
        created = sd.ensure_file()
        self.assertEqual(created, self.path)
        self.assertTrue(self.path.exists())
        content = self.path.read_text(encoding="utf-8-sig")
        self.assertIn("Словарь ударений", content)

    def test_ensure_file_does_not_overwrite_existing_content(self):
        self.path.write_text("моё = м+оё\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        sd.ensure_file()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "моё = м+оё\n")

    def test_simple_word_replacement(self):
        self.path.write_text("геральт = гер+альт\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Геральт пришёл в замок."), "Гер+альт пришёл в замок.")

    def test_case_is_preserved_on_replacement(self):
        self.path.write_text("геральт = гер+альт\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Геральт здесь."), "Гер+альт здесь.")
        self.assertEqual(sd.apply("геральт здесь."), "гер+альт здесь.")

    def test_full_caps_word_gives_full_caps_replacement(self):
        self.path.write_text("геральт = гер+альт\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("ГЕРАЛЬТ здесь."), "ГЕР+АЛЬТ здесь.")

    def test_yo_and_ye_are_distinct(self):
        # «е» и «ё» — разные буквы: «все» не должно превращаться в «всё».
        self.path.write_text("всё = вс+ё\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Это всё, а не все."), "Это вс+ё, а не все.")

    def test_multi_word_phrase_with_variable_whitespace(self):
        self.path.write_text("серый волк = серый в+олк\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Тут серый   волк рыщет."), "Тут серый в+олк рыщет.")

    def test_word_boundaries_are_respected(self):
        # «замок» не должен находиться внутри «замок-крепость» иначе, чем как отдельное слово,
        # а «за+мок» уже расставленное ударение не должно замениться повторно.
        self.path.write_text("замок = з+амок\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Уже стоит з+амок."), "Уже стоит з+амок.")
        self.assertEqual(sd.apply("Замки и замок."), "Замки и з+амок.")

    def test_longer_keys_take_priority_over_shorter_ones(self):
        self.path.write_text("волк = в+олк\nсерый волк = серый в+олк-вожак\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("Серый волк воет."), "Серый в+олк-вожак воет.")
        self.assertEqual(sd.apply("Волк воет."), "В+олк воет.")

    def test_reload_picks_up_changes_after_modification(self):
        self.path.write_text("слово = сл+ово\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("слово"), "сл+ово")
        time.sleep(0.02)   # гарантируем, что mtime файла реально изменится
        self.path.write_text("слово = слов+о\n", encoding="utf-8")
        self.assertEqual(sd.apply("слово"), "слов+о")

    def test_reload_handles_file_deletion(self):
        self.path.write_text("слово = сл+ово\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(len(sd), 1)
        self.path.unlink()
        self.assertEqual(sd.apply("слово"), "слово")
        self.assertEqual(len(sd), 0)

    def test_malformed_lines_are_skipped_without_crashing(self):
        self.path.write_text("хорошо = хор+ошо\nплохая строка без равно\n= тоже плохая\n", encoding="utf-8")
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply("хорошо"), "хор+ошо")
        self.assertEqual(len(sd), 1)

    def test_empty_text_input(self):
        sd = StressDictionary(self.path)
        self.assertEqual(sd.apply(""), "")


if __name__ == "__main__":
    unittest.main()
