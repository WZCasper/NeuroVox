# -*- coding: utf-8 -*-
"""
Скачивание модели озвучки (tts_engine.download_model) с локальным HTTP-сервером.

Интернет не нужен: сервер поднимается в этом же процессе и по сценарию отдаёт полный
файл, обрывает ответ посередине или возвращает 404. Так проверяются повторные попытки,
сверка размера и отсутствие «битого» файла — без обращения к models.silero.ai.
"""

import http.server
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import tts_engine  # noqa: E402

# 4096 байт: больше порога, который в тестах снижен до 1 КБ.
PAYLOAD = bytes(range(256)) * 16


class _Scenario:
    """Что отвечать на очередной запрос: по одному шагу на запрос, после списка — полный файл."""

    def __init__(self, steps):
        self._steps = list(steps)
        self._lock = threading.Lock()
        self.hits = 0

    def next_step(self) -> str:
        with self._lock:
            self.hits += 1
            return self._steps.pop(0) if self._steps else "ok"


def _make_handler(scenario: _Scenario):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format, *args):  # noqa: A002 — имя из базового класса
            return

        def do_GET(self):  # noqa: N802 — имя задаёт http.server
            step = scenario.next_step()
            if step == "404":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            if step == "truncated":
                self.wfile.write(PAYLOAD[: len(PAYLOAD) // 2])  # обрыв посередине
            else:
                self.wfile.write(PAYLOAD)

    return Handler


class DownloadModelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models_dir = pathlib.Path(tmp.name)
        overrides = [
            mock.patch.object(config, "MODELS_DIR", self.models_dir),
            mock.patch.object(config, "MIN_MODEL_SIZE_BYTES", 1024),
            mock.patch.object(config, "MODEL_DOWNLOAD_RETRY_DELAY_S", 0),
            mock.patch.object(config, "MODEL_DOWNLOAD_TIMEOUT_S", 5),
            mock.patch.object(config, "MODEL_DOWNLOAD_ATTEMPTS", 3),
        ]
        for override in overrides:
            override.start()
            self.addCleanup(override.stop)

    def _serve(self, steps):
        scenario = _Scenario(steps)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(scenario))
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/test_model.pt"
        registry = mock.patch.dict(config.SILERO_MODELS, {"test_model": url})
        registry.start()
        self.addCleanup(registry.stop)
        return scenario, url

    def test_success_on_first_attempt(self):
        scenario, _ = self._serve([])
        path = tts_engine.download_model("test_model")
        self.assertEqual(scenario.hits, 1)
        self.assertEqual(path.read_bytes(), PAYLOAD)

    def test_recovers_after_truncated_download(self):
        scenario, _ = self._serve(["truncated"])
        path = tts_engine.download_model("test_model")
        self.assertEqual(scenario.hits, 2)
        self.assertEqual(path.read_bytes(), PAYLOAD)

    def test_failure_leaves_no_partial_file_and_explains_manual_download(self):
        scenario, url = self._serve(["truncated"] * 3)
        with self.assertRaises(tts_engine.TtsError) as caught:
            tts_engine.download_model("test_model")
        self.assertEqual(scenario.hits, 3)
        self.assertEqual(list(self.models_dir.iterdir()), [])
        message = str(caught.exception)
        self.assertIn(url, message)
        self.assertIn(str(self.models_dir / "test_model.pt"), message)

    def test_http_404_is_not_retried(self):
        scenario, _ = self._serve(["404"])
        with self.assertRaises(tts_engine.TtsError):
            tts_engine.download_model("test_model")
        self.assertEqual(scenario.hits, 1)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(tts_engine.TtsError):
            tts_engine.download_model("no_such_model")


if __name__ == "__main__":
    unittest.main()
