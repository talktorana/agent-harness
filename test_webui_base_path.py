"""Regressions for mounting the web console under a URL prefix.

The console is a shell, so on a shared domain it belongs behind whatever
already authenticates that domain -- which means it is one PATH among others
(``/agent/...``) rather than a subdomain or a port of its own. Serving it there
needs two things to agree, and getting only one of them right is the interesting
failure:

* the HTML must FETCH the prefixed URLs, or the page loads and every call it
  makes goes to the domain's root and 404s -- a blank console with a working
  server, which looks like a JS bug;
* the router must answer UNDER the prefix and NOT at the root, or there are two
  paths to the same agent and only one of them is the one the proxy authenticates.

The regression guard here is structural rather than behavioural: it asserts that
no page contains a hardcoded absolute URL at all. That is the invariant that
actually holds this together, and it is the one a future edit is most likely to
break by adding a new ``fetch('/something')`` that works at the root and quietly
dies behind a proxy.

Run:  python -m unittest test_webui_base_path
"""
from __future__ import annotations

import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import webui  # noqa: E402

#: Any URL the browser resolves from the domain root: an absolute fetch, an
#: absolute href, or an EventSource/WebSocket pointed at "/".
ABSOLUTE_URL_IN_HTML = re.compile(
    r"""fetch\(\s*['"]/|href\s*=\s*["']/|EventSource\(\s*['"]/""")


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BasePathNormalisationTest(unittest.TestCase):
    """The prefix is written by hand in an env file; normalise it once."""

    def test_the_shapes_people_actually_write(self):
        for raw, want in (("", ""), ("/", ""), ("agent", "/agent"),
                          ("/agent", "/agent"), ("/agent/", "/agent"),
                          ("  /agent/  ", "/agent"), ("/a/b/", "/a/b")):
            with self.subTest(raw=raw):
                self.assertEqual(config._normalise_base_path(raw), want)

    def test_a_trailing_slash_cannot_survive_into_a_url(self):
        """/agent/ + /models is /agent//models: resolvable, and therefore a
        confusing intermittent-404 rather than a clean failure."""
        self.assertNotIn("//", config._normalise_base_path("/agent/") + "/x")


class PagesCarryNoRootRelativeUrlTest(unittest.TestCase):
    def test_neither_page_hardcodes_an_absolute_url(self):
        for name, html in (("PAGE", webui.PAGE), ("LIVE_PAGE", webui.LIVE_PAGE)):
            offenders = ABSOLUTE_URL_IN_HTML.findall(html)
            self.assertEqual(offenders, [],
                             f"{name} resolves URLs from the domain root, so it "
                             f"cannot be mounted under a prefix; use the "
                             f"{webui._BASE_TOKEN} token: {offenders}")

    def test_every_endpoint_the_pages_call_is_prefixed_with_the_token(self):
        """The token list is the contract; a new fetch() must join it."""
        for name, html in (("PAGE", webui.PAGE), ("LIVE_PAGE", webui.LIVE_PAGE)):
            prefixed = set(re.findall(
                re.escape(webui._BASE_TOKEN) + r"(/[a-z]+)", html))
            self.assertIn("/models", prefixed, name)
            self.assertIn("/chat", prefixed, name)
            #: The token regex captures the first path segment, so /task/ shows
            #: up as "/task". Assert the segment; the trailing slash is the
            #: concatenated task id.
            self.assertIn("/task", prefixed, name)

    def test_the_tokens_do_not_leak_into_served_html(self):
        with mock.patch.object(config, "WEBUI_BASE_PATH", "/agent"):
            for html in (webui._render(webui.PAGE),
                         webui._render(webui.LIVE_PAGE)):
                self.assertNotIn(webui._BASE_TOKEN, html)
                self.assertIn("/agent/models", html)


class RenderTest(unittest.TestCase):
    def test_no_prefix_renders_the_root_urls_unchanged(self):
        with mock.patch.object(config, "WEBUI_BASE_PATH", ""):
            page = webui._render(webui.PAGE)
        self.assertIn("fetch('/models')", page)
        self.assertIn('href="/live"', page)

    def test_a_prefix_is_joined_without_a_double_slash(self):
        with mock.patch.object(config, "WEBUI_BASE_PATH", "/agent"):
            for html in (webui._render(webui.PAGE),
                         webui._render(webui.LIVE_PAGE)):
                self.assertIn("fetch('/agent/models')", html)
                self.assertNotIn("/agent//", html)

    def test_the_prefix_is_read_at_render_time_not_at_import(self):
        """So one image serves both root and prefix with no rebuild."""
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(config, "WEBUI_BASE_PATH", ""))
            self.assertNotIn("/desk", webui._render(webui.LIVE_PAGE))
            stack.enter_context(
                mock.patch.object(config, "WEBUI_BASE_PATH", "/desk"))
            self.assertIn("fetch('/desk/models')",
                          webui._render(webui.LIVE_PAGE))


class RouteTest(unittest.TestCase):
    """The router: answer under the prefix, refuse everywhere else."""

    def _handler(self, path: str) -> webui.Handler:
        #: No socket: _route only reads self.path.
        handler = webui.Handler.__new__(webui.Handler)
        handler.path = path
        return handler

    def test_with_a_prefix_only_prefixed_paths_route(self):
        with mock.patch.object(config, "WEBUI_BASE_PATH", "/agent"):
            for path, want in (
                    ("/agent", "/"), ("/agent/", "/"),
                    ("/agent/live", "/live"),
                    #: A trailing slash is preserved: do_GET accepts both
                    #: "/live" and "/live/", so normalising here would be a
                    #: second opinion about a path the router already handles.
                    ("/agent/live/", "/live/"),
                    ("/agent/models", "/models"),
                    ("/agent/task/abc123", "/task/abc123"),
                    ("/agent/stop/abc123", "/stop/abc123"),
                    ("/agent/health", "/health"),
                    ("/agent/health?x=1", "/health"),
                    ("/agent/index.html", "/index.html"),
            ):
                with self.subTest(path=path):
                    self.assertEqual(self._handler(path)._route(), want)

    def test_the_root_is_refused_when_a_prefix_is_configured(self):
        """Two paths to one shell means one of them is unauthenticated."""
        with mock.patch.object(config, "WEBUI_BASE_PATH", "/agent"):
            for path in ("/", "/live", "/models", "/chat", "/health",
                         "/task/abc", "/index.html"):
                with self.subTest(path=path):
                    self.assertEqual(self._handler(path)._route(), "",
                                     f"{path} must not route behind a prefix")

    def test_a_prefix_is_not_matched_as_a_substring(self):
        """/agentextra is a different path; startswith('/agent') would say no."""
        with mock.patch.object(config, "WEBUI_BASE_PATH", "/agent"):
            for path in ("/agentextra", "/agentextra/live", "/agent-x/task/1"):
                with self.subTest(path=path):
                    self.assertEqual(self._handler(path)._route(), "")

    def test_without_a_prefix_everything_routes_at_the_root(self):
        with mock.patch.object(config, "WEBUI_BASE_PATH", ""):
            for path in ("/", "/live", "/models", "/task/abc", "/health"):
                with self.subTest(path=path):
                    self.assertEqual(self._handler(path)._route(), path)


class ServedUnderAPrefixTest(unittest.TestCase):
    """End to end over a real socket, as a proxy would reach it."""

    def setUp(self):
        self.port = _free_port()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), webui.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self._patch = mock.patch.object(config, "WEBUI_BASE_PATH", "/agent")
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 5)

    def get(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def test_the_prefixed_pages_are_served_with_prefixed_urls(self):
        for path, marker in (("/agent/live", "fetch('/agent/models')"),
                             ("/agent/", "fetch('/agent/models')")):
            status, body = self.get(path)
            self.assertEqual(status, 200, path)
            self.assertIn(marker, body)

    def test_the_bare_prefix_serves_the_console(self):
        status, _body = self.get("/agent")
        self.assertEqual(status, 200)

    def test_the_root_is_a_404_not_a_second_entrance(self):
        for path in ("/", "/live", "/health"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404, path)

    def test_health_answers_under_the_prefix_for_a_proxy_probe(self):
        status, body = self.get("/agent/health")
        self.assertEqual(status, 200)
        self.assertIn('"ok": true', body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
