# -*- coding: utf-8 -*-
"""Тесты вспомогательных функций озвучки (без PyTorch и без скачивания моделей)."""

import contextlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import config  # noqa: E402
import tts_engine as tts  # noqa: E402
from pronunciation import StressDictionary  # noqa: E402


class PrepareTextTests(unittest.TestCase):
    def test_removes_unsupported_symbols(self):
        self.assertEqual(tts.prepare_text("Привет ★ мир ▲ !"), "Привет мир !")

    def test_only_junk_gives_empty(self):
        self.assertEqual(tts.prepare_text("★▲"), "")


class SplitTests(unittest.TestCase):
    def test_all_chunks_within_limit_and_text_preserved(self):
        long_text = ("Это очень длинное предложение. " * 30).strip()
        chunks = tts.split_into_chunks(long_text)
        self.assertTrue(all(len(c) <= tts.MAX_CHUNK_CHARS for c in chunks))
        self.assertEqual(" ".join(chunks), long_text)

    def test_word_longer_than_limit_is_cut(self):
        word = "а" * 700
        chunks = tts.split_into_chunks(word)
        self.assertTrue(all(len(c) <= tts.MAX_CHUNK_CHARS for c in chunks))
        self.assertEqual("".join(chunks), word)

    def test_short_text_is_untouched(self):
        self.assertEqual(tts.split_into_chunks("Привет."), ["Привет."])


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.sin(np.linspace(0, 100, 48000)).astype(np.float32)

    def test_double_speed_halves_length(self):
        self.assertLessEqual(abs(tts.change_speed(self.audio, 2.0).size - 24000), 1)

    def test_half_speed_doubles_length(self):
        self.assertLessEqual(abs(tts.change_speed(self.audio, 0.5).size - 96000), 1)

    def test_normal_speed_is_untouched(self):
        self.assertIs(tts.change_speed(self.audio, 1.0), self.audio)

    def test_result_is_float32(self):
        self.assertEqual(tts.change_speed(self.audio, 1.3).dtype, np.float32)


class DownloadTests(unittest.TestCase):
    def test_broken_small_file_is_not_a_model(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(config, "MODELS_DIR", pathlib.Path(folder)):
            (pathlib.Path(folder) / "v4_ru.pt").write_bytes(b"x" * 100)
            self.assertFalse(tts.is_model_downloaded("v4_ru"))

    def test_unknown_model_gives_clear_error(self):
        with self.assertRaises(tts.TtsError) as ctx:
            tts.download_model("no_such_model")
        self.assertIn("Неизвестная модель", str(ctx.exception))

    def test_unreachable_server_gives_russian_error_and_leaves_no_junk(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(config, "MODELS_DIR", pathlib.Path(folder)), \
                mock.patch.dict(config.SILERO_MODELS, {"bad": "http://127.0.0.1:9/none.pt"}):
            with self.assertRaises(tts.TtsError) as ctx:
                tts.download_model("bad")
            self.assertIn("интернет", str(ctx.exception))
            self.assertEqual([p.name for p in pathlib.Path(folder).glob("*.part")], [])


class FakeAudio:
    """Имитация тензора: достаточно методов detach/cpu/numpy."""

    def __init__(self, samples):
        self._samples = samples

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._samples


class FakeTorch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def _fake_samples():
    return np.ones(4800, dtype=np.float64) * 0.1


class ModelWithExtras:
    def __init__(self):
        self.calls = []

    def apply_tts(self, text, speaker, sample_rate, put_accent=False, put_yo=False):
        self.calls.append({"put_accent": put_accent, "put_yo": put_yo})
        return FakeAudio(_fake_samples())


class ModelWithoutExtras:
    def __init__(self):
        self.calls = 0

    def apply_tts(self, text, speaker, sample_rate):
        self.calls += 1
        return FakeAudio(_fake_samples())


class ModelV5:
    """Как v5_5_ru из официального примера Silero: ударения и разбор омографов."""

    def __init__(self):
        self.calls = []
        self.texts = []

    def apply_tts(self, text, speaker, sample_rate, put_accent=False, put_yo=False,
                  put_stress_homo=False, put_yo_homo=False):
        self.texts.append(text)
        self.calls.append({"put_accent": put_accent, "put_yo": put_yo,
                           "put_stress_homo": put_stress_homo, "put_yo_homo": put_yo_homo})
        return FakeAudio(_fake_samples())


class ModelWithKwargs:
    def apply_tts(self, text, speaker, sample_rate, **kwargs):
        self.kwargs = kwargs
        return FakeAudio(_fake_samples())


def _engine_with(model):
    engine = tts.SileroTts()
    engine._model = model
    engine._torch = FakeTorch()
    engine._optional_kwargs = tts.detect_optional_kwargs(model)
    return engine


@contextlib.contextmanager
def _fake_torch_environment(model):
    """
    Подменяет модуль ``torch`` целиком, чтобы можно было вызвать настоящий
    ``SileroTts.load()`` (а не копировать его логику в тесте) без реального
    PyTorch и без скачивания файла модели.
    """
    model.to = lambda device: None
    fake_package = types.SimpleNamespace(
        PackageImporter=lambda path: types.SimpleNamespace(
            load_pickle=lambda group, name: model
        )
    )
    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda n: None,
        device=lambda name: name,
        package=fake_package,
        inference_mode=FakeTorch.inference_mode,
    )
    with mock.patch.object(tts, "download_model", return_value=pathlib.Path("fake_model.pt")), \
            mock.patch.dict(sys.modules, {"torch": fake_torch, "torch.package": fake_package}):
        yield


class SynthesizeTests(unittest.TestCase):
    def test_without_load_gives_clear_error(self):
        with self.assertRaises(tts.TtsError) as ctx:
            tts.SileroTts().synthesize("привет", "baya")
        self.assertIn("не загружена", str(ctx.exception))

    def test_speakers_without_model_come_from_config(self):
        self.assertEqual(set(tts.SileroTts().available_speakers()), set(config.SILERO_SPEAKERS))

    def test_detects_supported_optional_arguments(self):
        self.assertEqual(tts.detect_optional_kwargs(ModelWithExtras()), ("put_accent", "put_yo"))
        self.assertEqual(tts.detect_optional_kwargs(ModelWithoutExtras()), ())
        self.assertEqual(tts.detect_optional_kwargs(ModelWithKwargs()), tts._OPTIONAL_TTS_KWARGS)
        self.assertEqual(tts.detect_optional_kwargs(ModelV5()), tts._OPTIONAL_TTS_KWARGS)

    def test_passes_optional_arguments_when_supported(self):
        model = ModelWithExtras()
        fragments = _engine_with(model).synthesize("Привет, мир!", "baya")
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0].dtype, np.float32)
        self.assertEqual(model.calls[0], {"put_accent": True, "put_yo": True})

    def test_works_with_model_that_has_no_optional_arguments(self):
        model = ModelWithoutExtras()
        self.assertEqual(len(_engine_with(model).synthesize("Привет, мир!", "baya")), 1)
        self.assertEqual(model.calls, 1)

    def test_falls_back_when_model_rejects_optional_arguments(self):
        model = ModelWithoutExtras()
        engine = _engine_with(model)
        engine._optional_kwargs = ("put_accent", "put_yo")  # как если бы определение параметров не сработало
        self.assertEqual(len(engine.synthesize("Привет, мир!", "baya")), 1)
        self.assertEqual(engine._optional_kwargs, ())

    def test_empty_text_gives_no_audio(self):
        self.assertEqual(_engine_with(ModelWithExtras()).synthesize("★▲", "baya"), [])

    def test_homograph_flags_are_enabled_for_v5_model(self):
        model = ModelV5()
        _engine_with(model).synthesize("Мы открыли замок.", "baya")
        self.assertEqual(
            model.calls[0],
            {"put_accent": True, "put_yo": True, "put_stress_homo": True, "put_yo_homo": True},
        )

    def test_old_model_keeps_accent_flags_and_only_loses_homograph_ones(self):
        # v4_ru: ударения есть, разбора омографов нет. Отказ от одного параметра
        # не должен отключать остальные (раньше отключались все сразу).
        model = ModelWithExtras()
        engine = _engine_with(model)
        engine._optional_kwargs = tts._OPTIONAL_TTS_KWARGS       # как если бы определение параметров не сработало
        self.assertEqual(len(engine.synthesize("Привет, мир!", "baya")), 1)
        self.assertEqual(model.calls[0], {"put_accent": True, "put_yo": True})
        self.assertEqual(engine._optional_kwargs, ("put_accent", "put_yo"))

    def test_fallback_survives_unparseable_error_message(self):
        class Strange:
            def __init__(self):
                self.calls = 0

            def apply_tts(self, text, speaker, sample_rate, **kwargs):
                self.calls += 1
                if kwargs:
                    raise TypeError("что-то пошло не так")
                return FakeAudio(_fake_samples())

        model = Strange()
        engine = _engine_with(model)
        self.assertEqual(len(engine.synthesize("Привет, мир!", "baya")), 1)
        self.assertEqual(engine._optional_kwargs, ())

    def test_capital_phrase_is_normalized_before_synthesis(self):
        model = ModelV5()
        _engine_with(model).synthesize("КУДА ТЫ ПРОПАЛ?", "baya")
        self.assertEqual(model.texts[0], "Куда ты пропал?")

    def test_user_stress_dictionary_is_applied(self):
        model = ModelV5()
        engine = _engine_with(model)
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "stress.txt"
            path.write_text("геральт = гер+альт\n", encoding="utf-8")
            engine._dictionary = StressDictionary(path)
            engine.synthesize("Геральт пришёл.", "baya")
        self.assertEqual(model.texts[0], "Гер+альт пришёл.")

    def test_repeated_phrase_does_not_call_model_again(self):
        # Основное требование: повторная фраза (тот же текст, голос, скорость)
        # отдаётся из кэша, а model.apply_tts вызывается только один раз.
        model = ModelV5()
        engine = _engine_with(model)
        first = engine.synthesize("Осторожно, засада!", "baya")
        second = engine.synthesize("Осторожно, засада!", "baya")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_repeated_phrase_with_different_speed_calls_model_again(self):
        # Скорость входит в ключ кэша: то же самое произнесённое быстрее или
        # медленнее — это другой результат, а не то же самое аудио.
        model = ModelV5()
        engine = _engine_with(model)
        engine.synthesize("Осторожно, засада!", "baya", speed=1.0)
        engine.synthesize("Осторожно, засада!", "baya", speed=1.4)
        self.assertEqual(len(model.calls), 2)

    def test_repeated_phrase_with_different_speaker_calls_model_again(self):
        model = ModelV5()
        engine = _engine_with(model)
        engine.synthesize("Осторожно, засада!", "baya")
        engine.synthesize("Осторожно, засада!", "aidar")
        self.assertEqual(len(model.calls), 2)

    def test_cache_key_uses_text_after_stress_dictionary_and_caps_normalization(self):
        # Ключ кэша строится из ИТОГОВОГО текста (после normalize_caps и словаря
        # ударений): два разных сырых текста, дающих одинаковый итоговый текст,
        # обязаны считаться одной и той же фразой и не приводить ко второму
        # обращению к модели.
        model = ModelV5()
        engine = _engine_with(model)
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "stress.txt"
            path.write_text("геральт = гер+альт\n", encoding="utf-8")
            engine._dictionary = StressDictionary(path)
            engine.synthesize("Геральт пришёл.", "baya")
            # КАПС даёт тот же нормализованный текст после normalize_caps + словаря.
            engine.synthesize("ГЕРАЛЬТ ПРИШЁЛ.", "baya")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.texts[0], "Гер+альт пришёл.")

    def test_cache_survives_reload_of_same_model(self):
        # Повторный load() с тем же именем модели — обычный no-op (модель уже
        # загружена), а не «смена модели»: кэш не должен опустошаться.
        model = ModelV5()
        engine = _engine_with(model)
        engine.synthesize("Привет!", "baya")
        with _fake_torch_environment(model):
            engine.load()  # self._model уже не None -> сразу вернётся, ничего не изменив
        self.assertEqual(len(engine._cache), 1)

    def test_loading_different_model_clears_cache(self):
        # Настоящий сквозной сценарий: SileroTts.load() с другим model_name
        # должен опустошить кэш — старое аудио синтезировано другой моделью.
        model_a = ModelV5()
        engine = tts.SileroTts("v5_5_ru")
        with _fake_torch_environment(model_a):
            engine.load()
        engine.synthesize("Привет!", "baya")
        self.assertEqual(len(engine._cache), 1)

        model_b = ModelV5()
        engine.model_name = "v4_ru"
        engine._model = None  # иначе load() сочтёт, что модель уже загружена, и ничего не сделает
        with _fake_torch_environment(model_b):
            engine.load()
        self.assertEqual(len(engine._cache), 0)

        engine.synthesize("Привет!", "baya")
        self.assertEqual(len(model_b.calls), 1)  # обратились к НОВОЙ модели, не к кэшу старой


class SynthesisCacheTests(unittest.TestCase):
    """Прямые тесты _SynthesisCache — вытеснение по объёму, а не по числу записей."""

    def _make_fragment(self, num_samples):
        return [np.ones(num_samples, dtype=np.float32)]

    def test_get_missing_key_returns_none(self):
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        self.assertIsNone(cache.get(("нет такой фразы",)))

    def test_put_then_get_returns_equal_but_independent_copy(self):
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        original = self._make_fragment(100)
        cache.put(("фраза",), original)
        retrieved = cache.get(("фраза",))
        np.testing.assert_array_equal(retrieved[0], original[0])
        self.assertIsNot(retrieved[0], original[0])   # копия, а не та же ссылка
        # Мутация возвращённой копии не должна повредить то, что лежит в кэше.
        retrieved[0][0] = -999.0
        self.assertEqual(cache.get(("фраза",))[0][0], 1.0)

    def test_eviction_drops_least_recently_used_entry(self):
        # Лимит в 1000 семплов; три записи по 400 не помещаются одновременно —
        # при добавлении третьей должна уйти первая (наименее давно нужная).
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        cache.put(("a",), self._make_fragment(400))
        cache.put(("b",), self._make_fragment(400))
        cache.put(("c",), self._make_fragment(400))
        self.assertIsNone(cache.get(("a",)))
        self.assertIsNotNone(cache.get(("b",)))
        self.assertIsNotNone(cache.get(("c",)))

    def test_get_refreshes_recency_and_protects_from_eviction(self):
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        cache.put(("x",), self._make_fragment(300))
        cache.put(("y",), self._make_fragment(300))
        cache.put(("z",), self._make_fragment(300))
        cache.get(("x",))  # «освежаем» x — теперь y самый давний
        cache.put(("w",), self._make_fragment(300))  # должен вытеснить y, не x
        self.assertIsNotNone(cache.get(("x",)))
        self.assertIsNone(cache.get(("y",)))
        self.assertIsNotNone(cache.get(("z",)))
        self.assertIsNotNone(cache.get(("w",)))

    def test_updating_existing_key_does_not_double_count_its_size(self):
        # Повторная запись по тому же ключу заменяет старое значение, а не
        # добавляется поверх него — суммарный объём не должен «раздуваться».
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        cache.put(("a",), self._make_fragment(400))
        cache.put(("a",), self._make_fragment(400))
        self.assertEqual(cache._total_samples, 400)
        self.assertEqual(len(cache), 1)

    def test_fragment_larger_than_whole_cache_is_not_stored(self):
        # Одна фраза длиннее лимита всего кэша целиком — хранить её бессмысленно
        # (она одна вытеснила бы всё остальное), поэтому она просто не кэшируется.
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        cache.put(("огромная фраза",), self._make_fragment(1001))
        self.assertIsNone(cache.get(("огромная фраза",)))
        self.assertEqual(len(cache), 0)

    def test_clear_empties_cache(self):
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=1.0)
        cache.put(("a",), self._make_fragment(100))
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(("a",)))

    def test_zero_max_seconds_disables_caching(self):
        # Лимит 0 — кэш полностью выключен (например, для отладки), а не ошибка.
        cache = tts._SynthesisCache(sample_rate=1000, max_seconds=0.0)
        cache.put(("a",), self._make_fragment(10))
        self.assertIsNone(cache.get(("a",)))

    def test_default_limit_matches_config(self):
        cache = tts._SynthesisCache(sample_rate=config.TTS_SAMPLE_RATE)
        expected = int(config.TTS_CACHE_MAX_SECONDS * config.TTS_SAMPLE_RATE)
        self.assertEqual(cache._max_samples, expected)


if __name__ == "__main__":
    unittest.main()
