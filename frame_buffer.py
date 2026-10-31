# -*- coding: utf-8 -*-
"""
Буфер снимков экрана.

Снимок экрана занимает единицы миллисекунд, а распознавание текста — сотни. Если
делать их в одном цикле, субтитр, который сменился за время распознавания
предыдущего, пропадает. Поэтому снимки делаются быстро и складываются сюда, а
распознавание берёт их отсюда строго по очереди, в порядке появления.

Правила буфера:
  * снимки читаются по очереди (самый старый непрочитанный — первым), поэтому
    реплики из разных областей не смешиваются и не теряются;
  * прочитанный снимок хранится ещё ``retention`` секунд и затем удаляется;
  * непрочитанный снимок, который ждёт дольше ``max_unread_age`` секунд, удаляется
    как устаревший (реплика полуминутной давности в игре уже никому не нужна);
  * «недоустоявшийся» снимок (сделан в момент, когда картинка ещё менялась)
    заменяется более новым снимком той же области — читать промежуточные
    состояния анимации нет смысла;
  * устоявшиеся снимки сами по себе НЕ вытесняют друг друга при добавлении
    (put) — несколько разных устоявшихся субтитров одной области могут ждать
    своей очереди одновременно, и каждый будет распознан по очереди, в
    порядке появления. Если распознавание не успевает за скоростью субтитров
    на экране, вызывающий код (workers.OcrWorker) может явно запросить переход
    к самому свежему снимку через skip_to_latest — так из буфера читается
    то, что происходит в игре СЕЙЧАС, а не история с нарастающей задержкой;
  * размер буфера ограничен, чтобы память не росла при долгой игре.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("neurovox.buffer")


@dataclass
class CapturedFrame:
    """Один снимок области экрана, ожидающий распознавания."""

    seq: int                          # порядковый номер (определяет очерёдность чтения)
    area: int                         # номер области (0, 1, 2 ...)
    image: Optional[np.ndarray]       # подготовленный кадр для OCR; None — метка «область опустела»
    captured_at: float                # время снимка (time.monotonic)
    settled: bool = True              # True — картинка устоялась, текст показан целиком
    read_at: Optional[float] = None   # когда снимок взят на распознавание

    @property
    def is_marker(self) -> bool:
        """Метка «область опустела»: субтитр исчез, изображения нет."""
        return self.image is None

    @property
    def is_read(self) -> bool:
        return self.read_at is not None


class FrameBuffer:
    """Потокобезопасный буфер снимков с очерёдностью и удалением по времени."""

    def __init__(
        self,
        max_frames: int = config.FRAME_BUFFER_MAX_FRAMES,
        retention: float = config.FRAME_RETENTION_SECONDS,
        max_unread_age: float = config.FRAME_MAX_UNREAD_AGE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_frames = max(2, int(max_frames))
        self._retention = float(retention)
        self._max_unread_age = float(max_unread_age)
        self._clock = clock
        self._frames: List[CapturedFrame] = []
        self._cond = threading.Condition()
        self._seq = 0
        self._dropped_stale = 0
        self._dropped_overflow = 0

    # -- добавление ---------------------------------------------------------

    def put(
        self,
        area: int,
        image: Optional[np.ndarray],
        settled: bool = True,
        captured_at: Optional[float] = None,
    ) -> CapturedFrame:
        """
        Кладёт снимок в буфер.

        image=None — служебная метка «область опустела» (нужна, чтобы фильтр повторов
        знал: реплика закончилась, и такая же фраза позже — это новая реплика).
        """
        with self._cond:
            self._seq += 1
            frame = CapturedFrame(
                seq=self._seq,
                area=area,
                image=image,
                captured_at=self._clock() if captured_at is None else captured_at,
                settled=settled,
            )
            if image is not None:
                # Непрочитанные промежуточные снимки этой области устарели: их вытесняет новый.
                self._frames = [
                    f for f in self._frames
                    if not (f.area == area and not f.settled and not f.is_read and not f.is_marker)
                ]
            self._frames.append(frame)
            self._enforce_limit_locked()
            self._cond.notify_all()
            return frame

    # -- чтение -------------------------------------------------------------

    def take_next(self, timeout: Optional[float] = None) -> Optional[CapturedFrame]:
        """
        Берёт на распознавание самый старый непрочитанный снимок и помечает его прочитанным.

        Если снимков нет, ждёт до ``timeout`` секунд (None — ждать бесконечно) и
        возвращает None, когда время вышло.
        """
        with self._cond:
            self._cond.wait_for(self._has_unread_locked, timeout=timeout)
            for frame in self._frames:
                if not frame.is_read:
                    frame.read_at = self._clock()
                    return frame
            return None

    # -- обслуживание -------------------------------------------------------

    def purge(self) -> int:
        """
        Удаляет прочитанные снимки старше ``retention`` и устаревшие непрочитанные.

        Метки «область опустела» не несут изображения и как устаревшие не удаляются —
        иначе фильтр повторов мог бы навсегда «запомнить» уже закончившуюся реплику.
        Возвращает число удалённых снимков.
        """
        with self._cond:
            now = self._clock()
            kept: List[CapturedFrame] = []
            removed = 0
            for frame in self._frames:
                if frame.is_read:
                    expired = now - frame.read_at >= self._retention
                else:
                    expired = (not frame.is_marker) and (now - frame.captured_at >= self._max_unread_age)
                    if expired:
                        self._dropped_stale += 1
                if expired:
                    removed += 1
                else:
                    kept.append(frame)
            if removed:
                self._frames = kept
                logger.debug("Буфер снимков: удалено %d, осталось %d", removed, len(kept))
            return removed

    def clear(self) -> None:
        """Удаляет все снимки (при остановке слежения)."""
        with self._cond:
            self._frames = []
            self._cond.notify_all()

    # -- сведения -----------------------------------------------------------

    def __len__(self) -> int:
        with self._cond:
            return len(self._frames)

    def unread_count(self) -> int:
        """Сколько снимков ещё ждёт распознавания (метки не считаются)."""
        with self._cond:
            return sum(1 for f in self._frames if not f.is_read and not f.is_marker)

    def skip_to_latest(self, area: int, after_seq: int) -> "Tuple[Optional[CapturedFrame], int]":
        """
        Если, пока снимок ``after_seq`` ждал своей очереди, для ТОЙ ЖЕ области
        уже накопился более новый непрочитанный снимок — возвращает самый
        свежий из них и помечает прочитанными все промежуточные (чтобы не
        распознавать их тоже и не засорять буфер).

        Если более нового снимка нет (``after_seq`` и так самый свежий
        непрочитанный снимок своей области) — возвращает (None, 0), ничего
        не меняя: вызывающий код должен распознать именно его как обычно.

        Возвращает (самый свежий снимок области или None, число снимков той
        же области, которые из-за перехода к latest не будут распознаны —
        это и накопившиеся «между» ``after_seq`` и latest снимки, и сам снимок
        ``after_seq`` (ведь раз вызывающий код обратился за заменой для него,
        значит и его распознавать не стали — он тоже не будет озвучен); 0,
        если более нового снимка не нашлось и ``after_seq`` остаётся
        единственным кандидатом на распознавание, как обычно).

        Семантика числа пропущенных — та же, что у workers.skip_stale_duplicates
        (см. её докстринг): каждый «обгон» более новым кандидатом считается
        отдельно, включая сам стартовый элемент как первого обойдённого.

        Это та же идея, что и workers.skip_stale_duplicates (см. её докстринг
        — там то же самое происходит с уже РАСПОЗНАННЫМИ репликами перед
        озвучкой), только здесь применяется РАНЬШЕ, перед самим распознаванием:
        распознавание на процессоре без видеокарты может занимать секунды, и
        если за это время на экране уже дважды сменился субтитр той же
        области, нет смысла распознавать то, что уже устарело, — лучше сразу
        перейти к самому свежему и тем самым нагнать текущий момент на экране,
        а не читать историю с нарастающей задержкой.

        Помеченные здесь как прочитанные снимки не теряются бесследно: тот
        факт, что реплика была ПОКАЗАНА на экране, фиксируется (это те же
        данные, что видел бы игрок), а время на их распознавание и озвучку
        тратить не приходится — то же самое уже происходит на уровне очереди
        озвучки (put_dropping_oldest, skip_stale_duplicates) для чуть более
        позднего этапа конвейера.
        """
        with self._cond:
            same_area_unread = [
                f for f in self._frames
                if f.area == area and not f.is_read and not f.is_marker and f.seq > after_seq
            ]
            if not same_area_unread:
                return None, 0
            latest = max(same_area_unread, key=lambda f: f.seq)
            # same_area_unread не включает сам after_seq-снимок (он уже взят
            # вызывающим кодом и в буфере либо отсутствует, либо помечен
            # прочитанным) — но раз вызывающий код обратился сюда вместо
            # того, чтобы распознать его, значит и он тоже не будет
            # озвучен. Поэтому в счётчик пропущенных он засчитывается
            # наравне с каждым промежуточным снимком списка — отсюда без
            # вычитания единицы, в отличие от простого len(same_area_unread) - 1.
            skipped = len(same_area_unread)
            now = self._clock()
            for frame in same_area_unread:
                if frame.seq != latest.seq:
                    frame.read_at = now
            latest.read_at = now
            self._cond.notify_all()
            return latest, skipped

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {
                "total": len(self._frames),
                "unread": sum(1 for f in self._frames if not f.is_read and not f.is_marker),
                "read": sum(1 for f in self._frames if f.is_read),
                "dropped_stale": self._dropped_stale,
                "dropped_overflow": self._dropped_overflow,
            }

    # -- внутреннее ---------------------------------------------------------

    def _has_unread_locked(self) -> bool:
        return any(not f.is_read for f in self._frames)

    def _enforce_limit_locked(self) -> None:
        """Держит размер в пределах лимита: сначала выбрасывает прочитанные, потом самые старые."""
        overflow = len(self._frames) - self._max_frames
        if overflow <= 0:
            return
        # 1) прочитанные снимки уже не нужны — удаляем самые старые из них
        kept: List[CapturedFrame] = []
        for frame in self._frames:
            if overflow > 0 and frame.is_read:
                overflow -= 1
                continue
            kept.append(frame)
        # 2) если места всё равно нет — жертвуем самыми старыми непрочитанными снимками
        while overflow > 0:
            for index, frame in enumerate(kept):
                if not frame.is_marker:
                    logger.warning(
                        "Буфер снимков переполнен — пропущен снимок области %d (распознавание не успевает).",
                        frame.area + 1,
                    )
                    del kept[index]
                    self._dropped_overflow += 1
                    overflow -= 1
                    break
            else:
                # остались только метки — их немного, просто прекращаем
                break
        self._frames = kept
