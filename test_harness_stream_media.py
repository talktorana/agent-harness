"""Regression tests for streaming, multimodal input, and browser tools.

Run:  python -m unittest test_harness_stream_media -v
No network calls: streaming is exercised against synthetic SSE chunks, and
the browser tools are exercised with Playwright absent (the degradation
path) plus a stub page for the snapshot/ref machinery.
"""
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import tools_browser  # noqa: E402
import tools_media  # noqa: E402
from agent import assemble_stream  # noqa: E402


def _chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta,
                                                    finish_reason=None)])


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name,
                                                    arguments=arguments))


class AssembleStreamTest(unittest.TestCase):
    def test_content_streams_and_emits(self):
        seen = []
        chunks = [_chunk("hel"), _chunk("lo "), _chunk("world")]
        msg = assemble_stream(chunks, emit=seen.append)
        self.assertEqual(msg.content, "hello world")
        self.assertEqual("".join(seen), "hello world")
        self.assertIsNone(msg.tool_calls)

    def test_fragmented_tool_call_reassembles(self):
        chunks = [
            _chunk(tool_calls=[_tc(0, id="call_1", name="read_file")]),
            _chunk(tool_calls=[_tc(0, arguments='{"pa')]),
            _chunk(tool_calls=[_tc(0, arguments='th": ')]),
            _chunk(tool_calls=[_tc(0, arguments='"a.py"}')]),
        ]
        msg = assemble_stream(chunks)
        self.assertEqual(len(msg.tool_calls), 1)
        tc = msg.tool_calls[0]
        self.assertEqual(tc.id, "call_1")
        self.assertEqual(tc.function.name, "read_file")
        self.assertEqual(tc.function.arguments, '{"path": "a.py"}')

    def test_two_parallel_tool_calls_keep_order(self):
        chunks = [
            _chunk(tool_calls=[_tc(0, id="a", name="git_status"),
                               _tc(1, id="b", name="git_diff")]),
            _chunk(tool_calls=[_tc(1, arguments="{}")]),
            _chunk(tool_calls=[_tc(0, arguments="{}")]),
        ]
        msg = assemble_stream(chunks)
        self.assertEqual([t.id for t in msg.tool_calls], ["a", "b"])

    def test_empty_stream_yields_empty_message(self):
        msg = assemble_stream(iter([]))
        self.assertIsNone(msg.content)
        self.assertIsNone(msg.tool_calls)


class MediaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(tools_media, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _png(self, name="x.png", size=None):
        p = self.root / name
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + (b"." * (size or 64)))
        return p

    def test_view_image_returns_envelope_and_round_trips(self):
        p = self._png()
        env = tools_media.view_image(str(p))
        payload = tools_media.parse_image_envelope(env)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["mime"], "image/png")
        self.assertEqual(base64.b64decode(payload["data"]), p.read_bytes())
        self.assertTrue(tools_media.data_url(payload)
                        .startswith("data:image/png;base64,"))
        self.assertIn("image", tools_media.caption(payload))

    def test_plain_text_is_not_an_envelope(self):
        self.assertIsNone(tools_media.parse_image_envelope("just text"))

    def test_unsupported_type_and_missing_file_refused(self):
        (self.root / "x.txt").write_text("not an image", encoding="utf-8")
        self.assertIn("UNSUPPORTED", tools_media.view_image("x.txt"))
        self.assertIn("NOT FOUND", tools_media.view_image("nope.png"))

    def test_oversized_image_refused(self):
        p = self._png("big.png", size=config.IMAGE_MAX_BYTES + 10)
        self.assertIn("TOO LARGE", tools_media.view_image(str(p)))

    def test_disabled_flag(self):
        with mock.patch.object(config, "IMAGES_ENABLED", False), \
                mock.patch.object(tools_media, "IMAGES_ENABLED", False):
            self.assertIn("DISABLED", tools_media.view_image("x.png"))


class BrowserDegradationTest(unittest.TestCase):
    """Playwright absent -> every tool says so, nothing crashes."""

    def setUp(self):
        tools_browser._STATE.update(pw=None, browser=None, page=None,
                                    refs={}, next=1)

    def test_tools_report_unavailable_without_playwright(self):
        try:
            import playwright  # noqa: F401
            self.skipTest("playwright installed; degradation path untested")
        except ImportError:
            pass
        for fn, kwargs in ((tools_browser.browser_open, {"url": "x"}),
                           (tools_browser.browser_snapshot, {}),
                           (tools_browser.browser_click, {"ref": "e1"}),
                           (tools_browser.browser_type, {"text": "a"}),
                           (tools_browser.browser_navigate, {"target": "x"}),
                           (tools_browser.browser_read, {}),
                           (tools_browser.browser_screenshot, {})):
            out = fn(**kwargs)
            self.assertIn("NOT AVAILABLE", out, fn.__name__)


class FakeElement:
    def __init__(self, tag, text="", visible=True, href=None):
        self._tag, self._text, self._visible, self._href = tag, text, visible, href
        self.clicked = 0
        self.filled = None

    def evaluate(self, _expr):
        return self._tag

    def inner_text(self):
        return self._text

    def is_visible(self):
        return self._visible

    def get_attribute(self, name):
        return self._href if name == "href" else None

    def click(self):
        self.clicked += 1

    def fill(self, text):
        self.filled = text

    def press(self, _key):
        pass

    def press_sequentially(self, text):
        self.filled = (self.filled or "") + text


class FakePage:
    def __init__(self, elements):
        self._elements = elements
        self.url = "http://test/"
        self.clicked = []

    def is_closed(self):
        return False

    def set_default_timeout(self, _ms):
        pass

    def query_selector_all(self, _sel):
        return self._elements

    def inner_text(self, _sel):
        return "page body text"

    def title(self):
        return "Test Page"

    def locator(self, selector):
        return SimpleNamespace(first=FakeElement("div"), click=lambda: None)

    def get_by_text(self, text, exact=False):
        return SimpleNamespace(first=FakeElement("button", text))

    def goto(self, url):
        self.url = url

    def screenshot(self, path=None, full_page=False):
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\nfake")


class BrowserStubTest(unittest.TestCase):
    def setUp(self):
        self.elements = [FakeElement("button", "Submit"),
                         FakeElement("input", ""),
                         FakeElement("a", "Docs", href="/docs"),
                         FakeElement("div", "hidden", visible=False)]
        self.page = FakePage(self.elements)
        tools_browser._STATE.update(pw=object(), browser=object(),
                                    page=self.page, refs={}, next=1)
        self.addCleanup(lambda: tools_browser._STATE.update(
            pw=None, browser=None, page=None, refs={}, next=1))

    def test_snapshot_numbers_visible_elements_only(self):
        out = tools_browser.browser_snapshot()
        self.assertIn("[e1]", out)
        self.assertIn("[e2]", out)
        self.assertIn("[e3]", out)
        self.assertNotIn("[e4]", out)          # hidden element skipped
        self.assertIn('"Submit"', out)
        self.assertIn("page body text", out)
        self.assertIn("url=http://test/", out)

    def test_click_by_ref_uses_snapshot_element(self):
        tools_browser.browser_snapshot()
        out = tools_browser.browser_click(ref="e1")
        self.assertIn("CLICKED", out)
        self.assertEqual(self.elements[0].clicked, 1)

    def test_stale_ref_is_refused(self):
        out = tools_browser.browser_click(ref="e9")
        self.assertIn("UNKNOWN REF", out)

    def test_type_fills_target(self):
        tools_browser.browser_snapshot()
        out = tools_browser.browser_type(ref="e2", text="hello")
        self.assertIn("TYPED", out)
        self.assertEqual(self.elements[1].filled, "hello")

    def test_screenshot_returns_image_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "shot.png"
            env = tools_browser.browser_screenshot(path=str(dest))
        payload = tools_media.parse_image_envelope(env)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["mime"], "image/png")
        self.assertIn("browser screenshot", payload["note"])


if __name__ == "__main__":
    unittest.main()
