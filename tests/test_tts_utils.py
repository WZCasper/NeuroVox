# -*- coding: utf-8 -*-
"""Тесты вспомогательных функций озвучки (без PyTorch и без скачивания моделей)."""

import contextlib
import pathlib
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
