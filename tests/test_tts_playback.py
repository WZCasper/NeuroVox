# -*- coding: utf-8 -*-
"""
Тесты потока озвучки (TtsPlaybackWorker).

Два уровня, как и в test_workers.py:
  * SkipStaleDuplicatesTests — прямые тесты workers.skip_stale_duplicates,
    без потоков и реального времени: чистая логика работы с очередью.
  * TtsBacklogCatchUpIntegrationTests / SpeakingEventsIntegrationTests —
    сквозные тесты через настоящие потоки на подделках экрана, OCR и звука
    (pipeline_fakes.run_pipeline).
"""

import pathlib
import queue
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from workers import SpeechItem, skip_stale_duplicates  # noqa: E402

from tests.pipeline_fakes import run_pipeline, segments  # noqa: E402


def _item(area: int, text: str, created_at: float = 0.0) -> SpeechItem:
    return SpeechItem(area=area, text=text, created_at=created_at)


class SkipStaleDuplicatesTests(unittest.TestCase):
    """
    Прямые тесты skip_stale_duplicates: если для области только что взятой на
    озвучку реплики в очереди уже ждёт более новая — переходим сразу к ней;
    реплики ДРУГИХ областей это не затрагивает и не меняет их относительный
    порядок.
    """

    def test_nothing_to_skip_when_queue_is_empty(self):
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        item = _item(0, "первая")
        result, skipped = skip_stale_duplicates(q, item)
        self.assertIs(result, item)
        self.assertEqual(skipped, 0)
        self.assertEqual(q.qsize(), 0)

    def test_single_waiting_item_of_another_area_is_left_untouched(self):
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        other = _item(1, "Б — единственная")
        q.put_nowait(other)
        result, skipped = skip_stale_duplicates(q, _item(0, "А — единственная"))
        self.assertEqual(skipped, 0)
        self.assertEqual(result.text, "А — единственная")
        # Реплика другой области вернулась в очередь нетронутой.
        self.assertEqual(q.qsize(), 1)
        self.assertIs(q.get_nowait(), other)

    def test_collapses_same_area_backlog_to_the_latest(self):
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        older = _item(0, "вторая")
        newest = _item(0, "третья")
        q.put_nowait(older)
        q.put_nowait(newest)
        result, skipped = skip_stale_duplicates(q, _item(0, "первая"))
        self.assertEqual(skipped, 2)
        self.assertIs(result, newest)
        self.assertEqual(q.qsize(), 0)

    def test_other_areas_keep_their_relative_order(self):
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        other_first = _item(1, "Б — первая")
        same_area_newer = _item(0, "А — новая")
        other_second = _item(1, "Б — вторая")
        q.put_nowait(other_first)
        q.put_nowait(same_area_newer)
        q.put_nowait(other_second)
        result, skipped = skip_stale_duplicates(q, _item(0, "А — старая"))
        self.assertEqual(skipped, 1)
        self.assertIs(result, same_area_newer)
        # Реплики области 1 остались в очереди, в прежнем порядке.
        self.assertEqual(q.qsize(), 2)
        self.assertIs(q.get_nowait(), other_first)
        self.assertIs(q.get_nowait(), other_second)

    def test_two_other_areas_are_both_preserved(self):
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        area1 = _item(1, "Б")
        area0_newer = _item(0, "А-новая")
        area2 = _item(2, "В")
        q.put_nowait(area1)
        q.put_nowait(area0_newer)
        q.put_nowait(area2)
        result, skipped = skip_stale_duplicates(q, _item(0, "А-старая"))
        self.assertEqual(skipped, 1)
        self.assertIs(result, area0_newer)
        self.assertEqual(q.qsize(), 2)
        self.assertIs(q.get_nowait(), area1)
        self.assertIs(q.get_nowait(), area2)

    def test_created_at_of_the_oldest_item_is_not_reused(self):
        # Важно: после пропуска устаревших возраст считается по created_at
        # САМОЙ СВЕЖЕЙ реплики (latest), а не отброшенной старой — иначе
        # актуальную реплику могли бы по ошибке счесть устаревшей сразу
        # после того, как мы только что до неё добрались.
        q: "queue.Queue[SpeechItem]" = queue.Queue()
        newest = _item(0, "свежая", created_at=100.0)
        q.put_nowait(newest)
        result, _skipped = skip_stale_duplicates(q, _item(0, "старая", created_at=0.0))
        self.assertEqual(result.created_at, 100.0)


class TtsBacklogCatchUpIntegrationTests(unittest.TestCase):
    """
    Сквозные тесты через настоящие потоки (run_pipeline, реальное время).

    Синтез и воспроизведение одной реплики на реальном железе без видеокарты
    занимают секунды; если за это время субтитры успевают смениться ещё
    несколько раз, озвучка не должна накапливать нарастающее отставание —
    вместо этого она догоняет текущие субтитры (см. workers.TtsPlaybackWorker
    и skip_stale_duplicates), жертвуя устаревшими репликами ради актуальности.
    """

    # Заметно РАЗНЫЕ по смыслу и длине реплики. Специально не варианты одной и
    # той же фразы с одним изменённым словом: такие слишком похожи друг на
    # друга попиксельно и сами по себе не проходят порог «кадр изменился» в
    # capture.frames_are_similar — это отдельная, не относящаяся к данному
    # тесту особенность, которую мы не хотим здесь случайно проверять.
    _TEXTS = (
        "Нам нужно спешить, скоро рассвет",
        "Осторожно, там ловушка!",
        "Ты уверен, что это сработает?",
        "Хорошо, идём за мной",
        "Что происходит? Я ничего не вижу",
        "Беги! Они нас заметили!",
        "Спасибо тебе за помощь",
    )

    def test_lag_stays_bounded_instead_of_growing_during_a_dialogue_burst(self):
        cadence = 1.0
        texts = list(self._TEXTS[:5])
        items = [(i * cadence, (i + 1) * cadence, texts[i]) for i in range(len(texts))]
        at = segments(items)
        timeline = lambda area, t: at(t)  # noqa: E731

        started = time.monotonic()
        player, _ocr, _buffer, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=texts,
            duration=len(texts) * cadence + 4.0,
            ocr_latency=0.2, play_seconds=1.8, area_pause=0.2, capture_fps=20.0,
        )

        self.assertGreater(len(player.events), 0, "должна прозвучать хотя бы одна реплика")

        appeared_at = {text: start for start, _end, text in items}
        for text, begin, _end in player.events:
            lag = (begin - started) - appeared_at[text]
            self.assertLess(
                lag, 4.5,
                f"озвучка «{text}» отстала от появления субтитра на {lag:.2f} с — "
                "отставание не должно накапливаться на протяжении сцены",
            )

        # И хотя бы одна устаревшая реплика должна была быть пропущена — иначе
        # тест ничего не проверяет (с этими параметрами нагрузка заведомо
        # превышает то, что поток озвучки успевает проговаривать вовремя).
        skipped_events = [e for e in bus.poll(limit=500) if e.kind == "skipped"]
        self.assertTrue(skipped_events, "ожидался хотя бы один пропуск устаревшей реплики")

    def test_normal_unhurried_cadence_speaks_every_line_as_before(self):
        # Контрольный случай: субтитр держится заметно дольше, чем вся обработка
        # одной фразы занимает в сумме — ничего пропускаться не должно, поведение
        # должно остаться таким же, как до появления skip_stale_duplicates.
        cadence = 2.0
        texts = list(self._TEXTS[:3])
        items = [(i * cadence, (i + 1) * cadence, texts[i]) for i in range(len(texts))]
        at = segments(items)
        timeline = lambda area, t: at(t)  # noqa: E731

        player, _ocr, _buffer, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=texts,
            duration=len(texts) * cadence + 3.0,
            ocr_latency=0.2, play_seconds=1.0, area_pause=0.2, capture_fps=20.0,
        )

        spoken_texts = [text for text, _begin, _end in player.events]
        self.assertEqual(spoken_texts, texts)
        skipped_events = [e for e in bus.poll(limit=500) if e.kind == "skipped"]
        self.assertEqual(skipped_events, [])

    def test_backlog_in_one_area_does_not_affect_another_area(self):
        # Область 0 перегружена частыми репликами (будет догонять себя).
        # Область 1 — редкие, обычные реплики: не должны пострадать.
        area0_items = [
            (0.0, 1.0, "Бежим скорее, за нами погоня"),
            (1.0, 2.0, "Держись, уже почти добрались"),
            (2.0, 3.0, "Вот он, тайный проход в стене"),
            (3.0, 4.0, "Факел гаснет, нужен новый"),
            (4.0, 5.0, "Слышишь этот скрежет вдали?"),
        ]
        area1_items = [
            (1.5, 3.5, "Капитан ожидает доклада на мостике"),
            (5.5, 7.5, "Груз благополучно доставлен на склад"),
        ]
        at0, at1 = segments(area0_items), segments(area1_items)

        def timeline(area, t):
            return at0(t) if area == 0 else at1(t)

        all_texts = [t for _, _, t in area0_items] + [t for _, _, t in area1_items]
        expected_area1_order = [t for _, _, t in area1_items]

        player, _ocr, _buffer, _bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40], [0, 100, 100, 40]], texts=all_texts,
            duration=10.0, ocr_latency=0.2, play_seconds=1.5, area_pause=0.3, capture_fps=20.0,
        )

        spoken_area1 = [text for text, _b, _e in player.events if text in expected_area1_order]
        self.assertEqual(
            spoken_area1, expected_area1_order,
            "перегрузка области 0 не должна пропускать или переставлять реплики области 1",
        )


class SpeakingEventsIntegrationTests(unittest.TestCase):
    """
    "speaking_start"/"speaking_end" — события, которыми TtsPlaybackWorker
    обрамляет именно звучание реплики (не синтез и не паузу перед ним).
    Интерфейс использует их, чтобы на это время прятать рамку области
    (см. overlay.SelectionOverlay.set_reading и gui.MainWindow).
    """

    def test_events_bracket_playback_with_the_correct_area(self):
        texts = ["Единственная реплика в этом тесте"]
        items = [(0.0, 10.0, texts[0])]
        at = segments(items)
        timeline = lambda area, t: at(t)  # noqa: E731

        _player, _ocr, _buffer, bus = run_pipeline(
            timeline, rois=[[0, 0, 100, 40]], texts=texts,
            duration=2.0, ocr_latency=0.1, play_seconds=0.5, area_pause=0.2, capture_fps=20.0,
        )

        events = bus.poll(limit=500)
        starts = [e for e in events if e.kind == "speaking_start"]
        ends = [e for e in events if e.kind == "speaking_end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertEqual(int(starts[0].value), 0)
        self.assertEqual(int(ends[0].value), 0)

        # "speaking_start" должен прийти раньше "speaking_end" и раньше, чем
        # среди событий появится "stopped" (конца теста). Порядок внутри самой
        # шины совпадает с порядком emit(), поэтому сравниваем позиции в списке.
        self.assertLess(events.index(starts[0]), events.index(ends[0]))


if __name__ == "__main__":
    unittest.main()
