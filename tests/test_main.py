# -*- coding: utf-8 -*-
"""Тест запуска без консоли (как в собранном .exe)."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoConsoleTests(unittest.TestCase):
    def test_missing_std_streams_are_replaced(self):
        code = (
            "import sys\n"
            "sys.stdout = None\n"
            "sys.stderr = None\n"
            "import main\n"
            "main._ensure_std_streams()\n"
            "print('текст в никуда')\n"
            "sys.stderr.write('ошибка в никуда')\n"
            "sys.exit(0 if (sys.stdout is not None and sys.stderr is not None) else 3)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))

    def test_unknown_arguments_do_not_break_parsing(self):
        sys.path.insert(0, str(ROOT))
        import main

        args = main._parse_args(["--selftest", "--что-то-неизвестное", "файл.txt"])
        self.assertTrue(args.selftest)
        self.assertFalse(args.strict)


if __name__ == "__main__":
    unittest.main()
