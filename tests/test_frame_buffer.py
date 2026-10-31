# -*- coding: utf-8 -*-
"""Тесты буфера снимков (frame_buffer.FrameBuffer)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from frame_buffer import FrameBuffer  # noqa: E402


class _Clock:
    """Управляемые «часы» для теста: время не идёт само, а переводится вручную."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _img(marker: int = 1) -> np.ndarray:
    return np.full((4, 4), marker, dtype=np.uint8)


class OrderingTests(unittest.TestCase):
    """Снимки должны выдаваться строго в порядке появления, независимо от области."""

    def setUp(self):
        self.clock = _Clock()
        self.buffer = FrameBuffer(max_frames=64, retention=5.0, max_unread_age=30.0, clock=self.clock)

    def test_take_next_returns_frames_in_arrival_order(self):
        self.buffer.put(area=0, image=_img(1))
        self.buffer.put(area=1, image=_img(2))
        self.buffer.put(area=0, image=_img(3))
        order = [self.buffer.take_next().seq for _ in range(3)]
        self.assertEqual(order, [1, 2, 3])

    def test_take_next_marks_frame_as_read(self):
        self.buffer.put(area=0, image=_img())
        frame = self.buffer.take_next()
        self.assertTrue(frame.is_read)
        self.assertIsNotNone(frame.read_at)

    def test_take_next_returns_none_on_timeout_when_empty(self):
        self.assertIsNone(self.buffer.take_next(timeout=0.01))

    def test_take_next_skips_already_read_frames(self):
        self.buffer.put(area=0, image=_img(1))
        self.buffer.put(area=0, image=_img(2))
        first = self.buffer.take_next()
        second = self.buffer.take_next()
        self.assertNotEqual(first.seq, second.seq)
        self.assertIsNone(self.buffer.take_next(timeout=0.01))

    def test_unread_count_ignores_markers(self):
        self.buffer.put(area=0, image=_img())
        self.buffer.put(area=1, image=None)   # метка «опустела» — не «непрочитанный кадр»
        self.assertEqual(self.buffer.unread_count(), 1)


class PendingReplacementTests(unittest.TestCase):
    """Непрочитанный, ещё не устоявшийся снимок заменяется более новым той же области."""

    def setUp(self):
        self.clock = _Clock()
        self.buffer = FrameBuffer(max_frames=64, retention=5.0, max_unread_age=30.0, clock=self.clock)

    def test_unsettled_unread_frame_of_same_area_is_replaced(self):
        self.buffer.put(area=0, image=_img(1), settled=False)
        self.buffer.put(area=0, image=_img(2), settled=False)
        self.assertEqual(len(self.buffer), 1)
        frame = self.buffer.take_next()
        self.assertEqual(frame.area, 0)
        self.assertTrue(np.array_equal(frame.image, _img(2)))

    def test_settled_frame_is_not_replaced(self):
        self.buffer.put(area=0, image=_img(1), settled=True)
        self.buffer.put(area=0, image=_img(2), settled=True)
        self.assertEqual(len(self.buffer), 2)

    def test_read_frame_is_not_replaced(self):
        self.buffer.put(area=0, image=_img(1), settled=False)
        self.buffer.take_next()   # прочитан — больше не «ждущий»
        self.buffer.put(area=0, image=_img(2), settled=False)
        self.assertEqual(len(self.buffer), 2)

    def test_replacement_does_not_affect_other_areas(self):
        self.buffer.put(area=0, image=_img(1), settled=False)
        self.buffer.put(area=1, image=_img(2), settled=False)
        self.buffer.put(area=0, image=_img(3), settled=False)
        self.assertEqual(len(self.buffer), 2)   # у области 0 — замена, у области 1 — как было


class SkipToLatestTests(unittest.TestCase):
    """
    skip_to_latest — явный запрос «догнать» более свежий снимок той же
    области, если он успел накопиться, пока текущий ждал своей очереди.
    В отличие от замены при put() (см. PendingReplacementTests выше), эта
    замена не происходит сама по себе — её явно вызывает OcrWorker перед
    распознаванием (см. workers.py), и только устоявшиеся снимки, которые
    put() сознательно НЕ вытесняет друг другом, тоже могут быть пропущены
    этим путём.
    """

    def setUp(self):
        self.clock = _Clock()
        self.buffer = FrameBuffer(max_frames=64, retention=5.0, max_unread_age=30.0, clock=self.clock)

    def test_returns_none_when_no_newer_frame_of_same_area(self):
        # after_seq и так единственный/самый свежий кандидат своей области —
        # он будет распознан как обычно, ничего не пропущено.
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=first.seq)
        self.assertIsNone(frame)
        self.assertEqual(skipped, 0)

    def test_returns_latest_and_correct_skipped_count(self):
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        self.buffer.put(area=0, image=_img(2), settled=True)
        third = self.buffer.put(area=0, image=_img(3), settled=True)
        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=first.seq)
        self.assertEqual(frame.seq, third.seq)
        self.assertTrue(np.array_equal(frame.image, _img(3)))
        # Из-за перехода к third НЕ будут распознаны и, значит, не прозвучат:
        # сам first (ради него и вызвали skip_to_latest) и second (кандидат
        # между ними) — итого 2, та же семантика, что у
        # workers.skip_stale_duplicates (см. её докстринг и тест ниже).
        self.assertEqual(skipped, 2)

    def test_skipped_frames_become_read_and_leave_buffer_on_purge(self):
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        second = self.buffer.put(area=0, image=_img(2), settled=True)
        self.buffer.skip_to_latest(area=0, after_seq=first.seq)
        # Пропущенный (второй) снимок помечен прочитанным — purge() заберёт
        # его так же, как обычно прочитанный снимок, через retention секунд.
        self.assertTrue(second.is_read)
        self.clock.advance(5.1)
        self.buffer.purge()
        self.assertEqual(len(self.buffer), 1)   # остался только «латест» (третий put не делали)

    def test_does_not_affect_other_areas(self):
        # Сценарий как в OcrWorker.run(): кадр area=0/seq=1 уже взят на
        # распознавание через take_next() — именно ЕГО актуальность и
        # проверяется дальше через skip_to_latest(after_seq=1).
        self.buffer.put(area=0, image=_img(1), settled=True)
        current = self.buffer.take_next()
        self.assertEqual(current.seq, 1)

        area1_frame = self.buffer.put(area=1, image=_img(2), settled=True)
        self.buffer.put(area=0, image=_img(3), settled=True)

        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=current.seq)
        # Пропущен ровно 1 снимок — сам current (area=0/seq=1): из-за перехода
        # к более свежему seq=3 он тоже не будет распознан и озвучен. Кадр
        # ДРУГОЙ области (area=1/seq=2), оказавшийся между ними по времени,
        # в этот счётчик не входит вовсе — это проверяется ниже отдельно.
        self.assertEqual(skipped, 1)
        self.assertEqual(frame.seq, 3)
        # Снимок области 1 остаётся нетронутым и непрочитанным — отдаётся как обычно,
        # следующим по очереди (он старше третьего снимка области 0).
        self.assertFalse(area1_frame.is_read)
        self.assertEqual(self.buffer.take_next().seq, area1_frame.seq)

    def test_ignores_already_read_frames(self):
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        self.buffer.take_next()   # читаем первый снимок обычным путём
        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=first.seq)
        self.assertIsNone(frame)   # второго снимка нет, а первый уже прочитан — не считается
        self.assertEqual(skipped, 0)

    def test_ignores_marker_frames_as_candidates(self):
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        self.buffer.put(area=0, image=None)   # область опустела — это метка, не кандидат
        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=first.seq)
        self.assertIsNone(frame)
        self.assertEqual(skipped, 0)

    def test_ignores_frames_at_or_before_after_seq(self):
        first = self.buffer.put(area=0, image=_img(1), settled=True)
        second = self.buffer.put(area=0, image=_img(2), settled=True)
        # Запрашиваем «новее second» — сам second (и всё, что было раньше) не кандидат.
        frame, skipped = self.buffer.skip_to_latest(area=0, after_seq=second.seq)
        self.assertIsNone(frame)
        self.assertEqual(skipped, 0)
        self.assertFalse(first.is_read)   # первый снимок при этом не тронут


class PurgeTests(unittest.TestCase):
    """Прочитанные снимки удаляются через retention, непрочитанные устаревшие — через max_unread_age."""

    def setUp(self):
        self.clock = _Clock()
        self.buffer = FrameBuffer(max_frames=64, retention=2.0, max_unread_age=10.0, clock=self.clock)

    def test_read_frame_survives_before_retention(self):
        self.buffer.put(area=0, image=_img())
        self.buffer.take_next()
        self.clock.advance(1.9)
        self.assertEqual(self.buffer.purge(), 0)
        self.assertEqual(len(self.buffer), 1)

    def test_read_frame_removed_after_retention(self):
        self.buffer.put(area=0, image=_img())
        self.buffer.take_next()
        self.clock.advance(2.1)
        self.assertEqual(self.buffer.purge(), 1)
        self.assertEqual(len(self.buffer), 0)

    def test_unread_frame_survives_before_max_age(self):
        self.buffer.put(area=0, image=_img())
        self.clock.advance(9.9)
        self.assertEqual(self.buffer.purge(), 0)

    def test_unread_frame_removed_after_max_age(self):
        self.buffer.put(area=0, image=_img())
        self.clock.advance(10.1)
        self.assertEqual(self.buffer.purge(), 1)
        self.assertEqual(len(self.buffer), 0)

    def test_marker_frame_is_never_purged_as_stale(self):
        # Метка «область опустела» не должна теряться из-за старения — иначе фильтр
        # повторов не узнает, что реплика закончилась, и не озвучит её снова, если
        # она появится опять.
        self.buffer.put(area=0, image=None)
        self.clock.advance(1000.0)
        self.assertEqual(self.buffer.purge(), 0)
        self.assertEqual(len(self.buffer), 1)

    def test_purge_keeps_unexpired_frames(self):
        self.buffer.put(area=0, image=_img(1))
        self.buffer.take_next()
        self.clock.advance(2.1)
        self.buffer.put(area=0, image=_img(2))   # свежий непрочитанный
        self.assertEqual(self.buffer.purge(), 1)  # удалён только первый (прочитанный, просрочен)
        self.assertEqual(len(self.buffer), 1)


class OverflowTests(unittest.TestCase):
    """Буфер не должен расти бесконечно: при переполнении освобождает место."""

    def setUp(self):
        self.clock = _Clock()
        self.buffer = FrameBuffer(max_frames=4, retention=100.0, max_unread_age=100.0, clock=self.clock)

    def test_read_frames_are_evicted_first(self):
        self.buffer.put(area=0, image=_img(1))
        self.buffer.take_next()                    # прочитан — первый кандидат на вытеснение
        for marker in (2, 3, 4):
            self.buffer.put(area=0, image=_img(marker))
        self.assertEqual(len(self.buffer), 4)
        # Прочитанный снимок должен исчезнуть, три непрочитанных — остаться.
        self.assertEqual(self.buffer.unread_count(), 3)

    def test_oldest_unread_is_dropped_when_all_unread(self):
        for marker in (1, 2, 3, 4, 5):
            self.buffer.put(area=0, image=_img(marker))
        self.assertEqual(len(self.buffer), 4)
        frame = self.buffer.take_next()
        # Самый первый (marker=1) должен был уступить место самому новому.
        self.assertFalse(np.array_equal(frame.image, _img(1)))

    def test_stats_report_dropped_overflow(self):
        for marker in range(6):
            self.buffer.put(area=0, image=_img(marker))
        self.assertGreater(self.buffer.stats()["dropped_overflow"], 0)


class ClearAndStatsTests(unittest.TestCase):
    def test_clear_empties_the_buffer(self):
        clock = _Clock()
        buffer = FrameBuffer(clock=clock)
        buffer.put(area=0, image=_img())
        buffer.put(area=1, image=_img())
        buffer.clear()
        self.assertEqual(len(buffer), 0)

    def test_stats_shape(self):
        clock = _Clock()
        buffer = FrameBuffer(clock=clock)
        buffer.put(area=0, image=_img())
        frame_keys = set(buffer.stats())
        self.assertEqual(frame_keys, {"total", "unread", "read", "dropped_stale", "dropped_overflow"})


if __name__ == "__main__":
    unittest.main()
