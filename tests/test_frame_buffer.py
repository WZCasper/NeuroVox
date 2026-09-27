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
