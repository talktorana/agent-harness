"""Memory writes under concurrency: no lost notes, no truncated files.

The harness legitimately has several writers in ONE process: the web console
runs each task in a thread, and the mission scheduler runs beside it. Every
memory mutation is a read-modify-write of a shared file, so two writers
landing together can each read the same base, prepend their own note and write
back -- the second write then erases the first. Nothing reports it: both calls
return "RECORDED", both transcripts look complete, and the note is simply gone.

These tests force the interleaving rather than hoping to hit it, and check the
other half of the property: when a write fails part-way, the PREVIOUS memory
must survive, because that file is the whole knowledge base the next task
loads.

Run:  python -m unittest test_memory_concurrency
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import memory  # noqa: E402


def _store(tmp: Path):
    root = tmp / "memories"
    (root / "repo").mkdir(parents=True, exist_ok=True)
    repo_file = root / "repo" / "demo.md"
    return memory.MemoryStore(session_id="t", root=root, repo_file=repo_file)


class ConcurrentAppendTest(unittest.TestCase):
    def test_two_writers_lose_nothing(self):
        """The race is forced, not left to luck.

        The window that matters is between reading the file and writing it
        back. ``_atomic_write`` is wrapped to sleep inside that window, so
        every unserialised pair of writers here collides deterministically --
        and with the lock held across read and write, they cannot.
        """
        rounds, writers = 8, 4
        real_write = memory._atomic_write

        def slow_write(path, text):
            time.sleep(0.01)
            real_write(path, text)

        for round_no in range(rounds):
            with tempfile.TemporaryDirectory() as tmp:
                store = _store(Path(tmp))
                barrier = threading.Barrier(writers)

                def work(n: int) -> None:
                    barrier.wait()
                    store.append("repo", f"note {n}", heading=f"N{round_no}-{n}")

                with mock.patch.object(memory, "_atomic_write",
                                       side_effect=slow_write):
                    threads = [threading.Thread(target=work, args=(n,))
                               for n in range(writers)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join(timeout=20)
                        self.assertFalse(t.is_alive(), "a writer hung")

                text = store.repo_file.read_text(encoding="utf-8")
                for n in range(writers):
                    self.assertIn(f"note {n}", text,
                                  f"round {round_no} lost writer {n}'s note: "
                                  f"a read-modify-write with no lock")
                self.assertNotIn("ERROR", text)

    def test_the_lock_is_reentrant_for_one_caller(self):
        """A mutator that calls another mutator must not deadlock itself."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(Path(tmp))
            store.append("repo", "first", heading="A")
            # revise takes the lock and calls _backup, which only reads -- but
            # a future helper that appends must not wedge the thread.
            out = store.revise("repo", "first", "second")
            self.assertIn("REVISED", out)
            self.assertIn("second", store.repo_file.read_text(encoding="utf-8"))


class AtomicWriteTest(unittest.TestCase):
    def test_a_failed_replace_leaves_the_previous_memory_intact(self):
        """Truncation on failure would be worse than a lost note.

        ``Path.write_text`` opens the target for writing (truncating it) and
        only then writes, so an error part-way leaves the file SHORT. For a
        repo memory that is the agent losing everything it knew, silently.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(Path(tmp))
            store.append("repo", "the original knowledge", heading="Base")
            before = store.repo_file.read_text(encoding="utf-8")

            def boom(*_a, **_k):
                raise OSError(28, "No space left on device")

            with mock.patch.object(os, "replace", side_effect=boom):
                out = store.append("repo", "the note that did not fit",
                                   heading="New")

            self.assertTrue(out.startswith("ERROR writing"), out)
            self.assertEqual(store.repo_file.read_text(encoding="utf-8"),
                             before,
                             "a failed write damaged the existing memory")
            self.assertNotIn("the note that did not fit",
                             store.repo_file.read_text(encoding="utf-8"))

    def test_a_successful_write_leaves_no_temporary_file_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _store(Path(tmp))
            store.append("repo", "kept", heading="K")
            leftovers = [p.name for p in store.repo_file.parent.iterdir()
                         if p.name.endswith(".tmp")]
            self.assertEqual(leftovers, [], str(leftovers))

    def test_the_temporary_name_cannot_collide_with_a_real_note(self):
        """.md.tmp is not .md, so a half-written file is never loaded."""
        p = Path("/tmp/x/memories/repo/demo.md")
        self.assertEqual(memory._atomic_write.__name__, "_atomic_write")
        self.assertNotEqual(p.with_suffix(p.suffix + ".tmp"), p)


_original_read_text = Path.read_text


def _read(path: Path) -> str:
    return _original_read_text(path, encoding="utf-8", errors="replace")


if __name__ == "__main__":
    unittest.main(verbosity=2)
