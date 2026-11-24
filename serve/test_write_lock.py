"""F-04: every command line the server writes to the engine's stdin goes through the write lock (wlock), and two
writers never interleave one line into another.  Fake engine process, no GPU.

    python -m unittest serve.test_write_lock -v
"""
import io
import queue
import threading
import time
import unittest
from unittest import mock

from serve.server import EngineDied, StrataEngine


class CheckedLock:
    """A lock that knows whether the calling code holds it (the stdin check below asks it)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.held = False

    def __enter__(self):
        self._lock.acquire()
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        self._lock.release()
        return False


class CheckingStdin:
    """Records each write; fails the test when a write is made without the lock held."""

    def __init__(self, lock):
        self.lock, self.writes = lock, []

    def write(self, s):
        assert self.lock.held, f"stdin written without wlock: {s!r}"
        self.writes.append(s)
        return len(s)

    def flush(self):
        pass


class SlowStdin:
    """Yields between characters, so two unlocked writers would interleave their lines."""

    def __init__(self):
        self.text = []

    def write(self, s):
        for ch in s:
            self.text.append(ch)
            time.sleep(0)
        return len(s)

    def flush(self):
        pass


def engine_with(replies):
    e = StrataEngine("missing-executable", [], lazy=True)
    e.proc = mock.Mock()
    e.proc.poll.return_value = None
    e.ended = False
    e.lines = queue.Queue()
    for r in replies:
        e.lines.put(r)
    e.wlock = CheckedLock()
    e.proc.stdin = CheckingStdin(e.wlock)
    return e


class EveryWriteUnderTheLock(unittest.TestCase):
    def test_vram_save_and_send_hold_the_lock_while_writing(self):
        e = engine_with(["VRAM reserve_mib=6000 expert_slots=1200\n"])
        e.vram(6000)
        self.assertEqual(e.proc.stdin.writes, ["VRAM 6000\n"])

        e = engine_with(["SAVED 62993 1234 401.5\n"])
        self.assertEqual(e.session_file("save", "/tmp/x.session")["tokens"], 62993)
        self.assertEqual(e.proc.stdin.writes, ["SAVE /tmp/x.session\n"])

        e = engine_with([])
        e._send("STOP")
        self.assertEqual(e.proc.stdin.writes, ["STOP\n"])

    def test_a_write_failure_keeps_each_sites_own_error(self):
        e = engine_with([])
        e.proc.stdin = mock.Mock()
        e.proc.stdin.write.side_effect = BrokenPipeError("gone")
        with self.assertRaises(BrokenPipeError):            # vram(): no handler there, as before
            e.vram(100)
        e.proc.stdin = mock.Mock()
        e.proc.stdin.write.side_effect = BrokenPipeError("gone")
        with self.assertRaisesRegex(EngineDied, "stopped unexpectedly"):   # _send / session_file: EngineDied
            e._send("STOP")

    def test_two_writers_never_split_each_others_lines(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.stdin = SlowStdin()
        e.wlock = threading.Lock()
        lines = [f"GEN 8 {','.join(str(i * 10 + k) for k in range(12))}" for i in range(40)]

        def writer(group):
            for line in group:
                e._write_line(line)

        threads = [threading.Thread(target=writer, args=(lines[0::2],)),
                   threading.Thread(target=writer, args=(lines[1::2],))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        written = "".join(e.proc.stdin.text).split("\n")
        self.assertEqual(written[-1], "")                    # every line ends with its newline
        self.assertEqual(sorted(written[:-1]), sorted(lines), "a line was split or interleaved")


if __name__ == "__main__":
    unittest.main()
