"""First-party memory regression: a revision/removal must not lose another note.

Uses only real MemoryStore operations and a controlled disk-read interleaving.
No agent, provider, process or external service is started.
"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import memory


class ConcurrentMutationTests(unittest.TestCase):
    def check_mutator_does_not_erase_concurrent_append(self, operation):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memories"
            file = root / "repo" / "demo.md"
            file.parent.mkdir(parents=True)
            store = memory.MemoryStore(session_id="race", root=root, repo_file=file)
            store.append("repo", "original note", heading="Base")
            read_started = threading.Event()
            allow_read_return = threading.Event()
            write_finished = threading.Event()
            errors = []
            original_read = Path.read_text

            def controlled_read(path, *args, **kwargs):
                result = original_read(path, *args, **kwargs)
                if threading.current_thread().name == "mutator" and path == file and not read_started.is_set():
                    read_started.set()
                    if not allow_read_return.wait(timeout=5):
                        raise RuntimeError("mutator fixture timed out")
                return result

            def mutate():
                try:
                    if operation == "revise":
                        store.revise("repo", "original note", "revised note")
                    else:
                        store.retire("repo", "Base")
                except Exception as exc:
                    errors.append(exc)

            def append():
                try:
                    store.append("repo", "concurrent note", heading="Concurrent")
                except Exception as exc:
                    errors.append(exc)
                finally:
                    write_finished.set()

            with mock.patch.object(Path, "read_text", controlled_read):
                mutator = threading.Thread(target=mutate, name="mutator")
                mutator.start()
                self.assertTrue(read_started.wait(timeout=5))
                writer = threading.Thread(target=append, name="writer")
                writer.start()
                # In the unsafe implementation the writer finishes while the
                # mutator holds a stale snapshot. In the fixed one it waits.
                write_finished.wait(timeout=0.6)
                allow_read_return.set()
                for thread in (mutator, writer):
                    thread.join(timeout=5)

            self.assertFalse(errors)
            self.assertFalse(mutator.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertIn("concurrent note", file.read_text(encoding="utf-8"))

    def test_revise_preserves_concurrent_append(self):
        self.check_mutator_does_not_erase_concurrent_append("revise")

    def test_retire_preserves_concurrent_append(self):
        self.check_mutator_does_not_erase_concurrent_append("retire")


if __name__ == "__main__":
    unittest.main()
