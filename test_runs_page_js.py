"""The runs page's SCRIPT must actually render the runs it is given.

WHY THIS FILE EXISTS
--------------------
Live 2026-11-02, reported by the operator: "/agent/runs is broken, don't see any
history for past two hours".  `/agent/runs.json` was serving 54 runs correctly
and the newest session file was seconds old. The page was broken:

    try{ ...const j = await r.json(); ... }      // j is BLOCK-SCOPED to the try
    ...
    const held = j.held || {};                   // ReferenceError, every load

That line was added hours earlier, to report the retention window. It threw
before `tb.innerHTML` was assigned, so `loadRuns()` aborted and the table stayed
empty for every operator -- a healthy system rendered as if all history had been
wiped.

No Python test could have seen it: the string was present, the endpoint was
correct, and the page returned HTTP 200. The only thing wrong was inside
JavaScript that nothing ever RAN. So this suite executes the page's script under
a stub DOM and asserts the table is filled.

It runs the script exactly as the server serves it (extracted from
``webui.RUNS_PAGE``), so a change to the page is exercised, not a copy of it.

REQUIRES node. The sidecar image does not ship it, so where node is absent these
checks SKIP with a reason rather than passing -- a suite you could not run is a
stated gap, never a pass.

Run:  python test_runs_page_js.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import webui  # noqa: E402


def page_script(page: str) -> str:
    """The script the page actually ships, with the base-path placeholder filled."""
    start = page.index("<script>") + len("<script>")
    end = page.index("</script>", start)
    return page[start:end].replace("__AGENT_BASE__", "")


#: One scheduled run and one manual run, in the shape /runs.json returns.
PAYLOAD = {
    "runs": [
        {"session_id": "aaa111", "file": "2026-11-02T06-49-34_aaa111.jsonl",
         "started": "2026-11-02T06:49:34.502903+00:00", "status": "running",
         "steps": None, "elapsed_sec": None, "model": "m",
         "prompt": "x", "prompt_preview": "a scheduled pass",
         "answer_preview": "", "has_answer": False},
        {"session_id": "bbb222", "file": "2026-11-02T06-19-10_bbb222.jsonl",
         "started": "2026-11-02T06:19:10+00:00", "status": "ok",
         "steps": 56, "elapsed_sec": 728.0, "model": "m",
         "prompt": "y", "prompt_preview": "an earlier pass",
         "answer_preview": "done", "has_answer": True},
    ],
    "sessions_dir": "/agent-data/sessions",
    "held": {"keep": 2000, "max_bytes_mb": 200, "runs": 54,
             "oldest": "2026-11-01T02:06:19", "over_budget": False},
}

#: Shaped like /agent/schedule: the page reads id / interval / enabled /
#: last_run / next_due / seconds_until, NOT the raw declaration's keys.
#: The job id is deliberately GENERIC: this repo ships to more than one
#: deployment, and test_genericity.py fails the build if a consumer's own job
#: name appears in it (it caught exactly that here on the first run).
SCHEDULE = {"paused": False, "schedule_file": "ops/agent-schedule.json",
            "jobs": [{"id": "monitoring-pass",
                      "interval": "every 30 min", "enabled": True,
                      "last_run": "2026-11-02T06:49:34+00:00",
                      "next_due": "2026-11-02T07:19:34+00:00",
                      "seconds_until": 120, "overdue": False,
                      "in_progress": False}]}

#: A DOM stub that records what the page WROTE. Deliberately explicit: a Proxy
#: that swallows everything would let a script throw into a void and still
#: "pass", which is the failure this suite exists to catch.
HARNESS = r"""
const elements = {};
function el(id){ return elements[id] || (elements[id] = {innerHTML:'', textContent:''}); }
global.document = { getElementById: el };
global.window = global;
global.setInterval = () => 0;
const payloads = JSON.parse(process.argv[2]);
global.fetch = async (url) => {
  const body = url.includes('runs.json') ? payloads.runs : payloads.schedule;
  return { json: async () => body };
};
(async () => {
  __SCRIPT__
  // loadSchedule() and loadRuns() were invoked at the end of the page script;
  // they are async, so give the microtasks a turn before reading the DOM.
  await new Promise(r => setTimeout(r, 50));
  process.stdout.write(JSON.stringify({
    runs: elements['runs'] ? elements['runs'].innerHTML : null,
    keep: elements['keep'] ? elements['keep'].textContent : null,
    err:  elements['err']  ? elements['err'].innerHTML   : null,
    sched: elements['sched'] ? (elements['sched'].innerHTML || elements['sched'].textContent) : null,
  }));
})();
"""


def run_page(script: str, node: str) -> dict:
    """Execute the page script under the stub DOM; return what it rendered."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.js"
        path.write_text(HARNESS.replace("__SCRIPT__", script), encoding="utf-8")
        payload = json.dumps({"runs": PAYLOAD, "schedule": SCHEDULE})
        done = subprocess.run([node, str(path), payload], capture_output=True,
                              text=True, timeout=90)
    if done.returncode != 0:
        raise AssertionError(f"the page script threw:\n{done.stderr[-2000:]}")
    try:
        return json.loads(done.stdout)
    except ValueError as exc:
        raise AssertionError(
            f"page produced no readable result ({exc}); stdout="
            f"{done.stdout[:800]!r} stderr={done.stderr[:800]!r}") from exc


@unittest.skipIf(shutil.which("node") is None,
                 "node is not installed here (the sidecar image ships none); "
                 "the page script cannot be executed, so this is a STATED GAP, "
                 "not a pass")
class RunsPageJavascriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        cls.rendered = run_page(page_script(webui.RUNS_PAGE), cls.node)

    def test_the_table_is_filled_with_the_runs_it_was_given(self):
        """THE regression: a throw before innerHTML leaves the page empty.

        The reference error was in the retention line, which runs BEFORE the
        table is written, so the symptom was "no history" rather than a partial
        page -- which is why it read as data loss.
        """
        html = self.rendered["runs"]
        self.assertIsNotNone(html, "the runs table was never written")
        self.assertIn("aaa111", html)
        self.assertIn("bbb222", html)
        self.assertNotIn("no runs on disk yet", html)
        self.assertEqual(len(html.split("<tr")), 3,
                         "two runs should render two rows")

    def test_the_retention_line_still_works(self):
        """The feature whose addition caused the break."""
        keep = self.rendered["keep"] or ""
        self.assertIn("2 run(s)", keep)
        self.assertIn("2026-11-01 02:06", keep)
        self.assertIn("2000", keep)

    def test_a_fetch_failure_is_reported_not_swallowed(self):
        """An unreadable endpoint must say so where the table would be."""
        script = page_script(webui.RUNS_PAGE).replace(
            "const r=await fetch('", "const r=await fetch('//broken') && 0 || await fetch('")
        #: Simpler: drive the catch branch directly.
        script = script.replace("rows=j.runs||[];", "throw new Error('boom');")
        out = run_page(script, self.node)
        self.assertIn("could not read runs", out["err"] or "")

    def test_the_schedule_header_is_filled(self):
        self.assertTrue((self.rendered["sched"] or "").strip(),
                        "the schedule line was never written either")

    def test_the_row_is_clickable_through_a_data_attribute(self):
        """The fix for the served-script SyntaxError, pinned as a property.

        The id must NOT be spliced into an inline handler with nested quotes:
        in the Python source `\'` collapses to `'`, so the emitted JS became
        three adjacent string literals and the whole script failed to parse.
        A data attribute has no quotes to get wrong. This asserts the shape
        because the execution test above only proves the script RUNS -- a
        future edit could reintroduce a fragile quoting scheme that happens to
        work for alphanumeric ids and breaks on the first odd one.
        """
        html = self.rendered["runs"] or ""
        self.assertIn('data-sid="aaa111"', html)
        self.assertIn("showRun(this.dataset.sid)", html)
        self.assertNotIn("showRun('", html,
                         "the id is being spliced into the handler again")


if __name__ == "__main__":
    unittest.main(verbosity=2)
