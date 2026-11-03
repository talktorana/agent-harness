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

#: Shaped like /agent/model: what the harness will use for a run that names no
#: model, and which layer decided it. Two aliases, so "did the page preselect
#: the right one" is answerable -- with a single-alias provider it cannot be.
MODEL_STATE = {"provider": "inferhub",
               "model": "inferhub-deepseek-v4.1-flash",
               "model_id": "alicn/deepseek-v4.1-flash-1017",
               "source": "console",
               "reasoning_effort": "max", "max_tokens": 32000,
               "choice_file": "/agent-data/sessions/model_choice.json",
               "error": None}

CATALOG = {"providers": {"inferhub": {
               "base_url": "https://api.inferhub.dev/v1",
               "models": ["inferhub-deepseek-v4.1-flash",
                          "inferhub-deepseek-v4-pro-0813"],
               "model_ids": {"inferhub-deepseek-v4.1-flash":
                                 "alicn/deepseek-v4.1-flash-1017",
                             "inferhub-deepseek-v4-pro-0813":
                                 "ali/deepseek-v4-pro-0813"},
               "key_configured": True}},
           "default": ["inferhub", "inferhub-deepseek-v4.1-flash"]}

#: A DOM stub that records what the page WROTE. Deliberately explicit: a Proxy
#: that swallows everything would let a script throw into a void and still
#: "pass", which is the failure this suite exists to catch.
#:
#: fetch() also RECORDS every request and answers by URL, so the model panel's
#: save path can be driven and its request body inspected. Answering every URL
#: with the runs payload (the original stub) silently made the panel fall into
#: its "catalogue unavailable" branch and proved nothing about it.
HARNESS = r"""
const elements = {}, requests = [];
function mk(tag){ return { tagName: tag, innerHTML:'', textContent:'', value:'',
  className:'', children:[],
  appendChild(c){ this.children.push(c); return c; } }; }
function el(id){ return elements[id] || (elements[id] = mk('div')); }
global.document = { getElementById: el, createElement: mk };
global.window = global;
global.setInterval = () => 0;
const payloads = JSON.parse(process.argv[2]);
global.fetch = async (url, opts) => {
  const method = (opts && opts.method) || 'GET';
  requests.push({url: String(url), method: method,
                 body: (opts && opts.body) || null});
  let body;
  if (String(url).endsWith('/model')) {
    // A POST answers with the report, then the page re-reads the state.
    body = method === 'POST'
      ? Object.assign({ok: true, saved: true,
                       saved_model_id: payloads.model.model_id},
                      payloads.model_after || payloads.model)
      : payloads.model;
  }
  else if (String(url).endsWith('/models')) body = payloads.catalog;
  else if (String(url).includes('runs.json')) body = payloads.runs;
  else body = payloads.schedule;
  return { json: async () => body };
};
(async () => {
  __SCRIPT__
  // The page's own loaders were invoked at the end of the script and are
  // async. Settle them BEFORE the driven action, or a handler called from
  // `after` runs against a DOM the page has not drawn yet -- which under this
  // stub means reading placeholder elements and testing nothing.
  await new Promise(r => setTimeout(r, 50));
  __AFTER__
  await new Promise(r => setTimeout(r, 50));
  process.stdout.write(JSON.stringify({
    runs: elements['runs'] ? elements['runs'].innerHTML : null,
    keep: elements['keep'] ? elements['keep'].textContent : null,
    err:  elements['err']  ? elements['err'].innerHTML   : null,
    sched: elements['sched'] ? (elements['sched'].innerHTML || elements['sched'].textContent) : null,
    model: elements['model'] ? (elements['model'].innerHTML || elements['model'].textContent) : null,
    mmsg: elements['mmsg'] ? elements['mmsg'].textContent : null,
    mprov: elements['mprov'] ? elements['mprov'].value : null,
    mmod: elements['mmod'] ? elements['mmod'].value : null,
    requests: requests,
  }));
})();
"""


def run_page(script: str, node: str, *, model=None, model_after=None,
             catalog=None, after: str = "") -> dict:
    """Execute the page script under the stub DOM; return what it rendered.

    ``after`` is spliced in before the settle delay, which is how the panel's
    click handlers are driven: they are function declarations in the page
    script, so they are callable from that scope and nowhere else.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.js"
        path.write_text(
            HARNESS.replace("__SCRIPT__", script).replace("__AFTER__", after),
            encoding="utf-8")
        payload = json.dumps({"runs": PAYLOAD, "schedule": SCHEDULE,
                              "model": MODEL_STATE if model is None else model,
                              "model_after": model_after,
                              "catalog": CATALOG if catalog is None else catalog})
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


@unittest.skipIf(shutil.which("node") is None,
                 "node is not installed here (the sidecar image ships none); "
                 "the page script cannot be executed, so this is a STATED GAP, "
                 "not a pass")
class RunsPageModelChoiceTest(unittest.TestCase):
    """The "model for runs" panel must show and save the REAL default.

    WHY THIS IS EXECUTED RATHER THAN GREPPED
    ----------------------------------------
    The panel is the only thing standing between an operator and a model they
    cannot otherwise change without shell access: the catalogue is mounted
    read-only and the applier refuses to rewrite it. Two failures are possible
    and neither is visible in the HTML:

      * the dropdown preselects the WRONG alias, so pressing Save silently
        switches the model away from the one in force -- the operator sees a
        success message and a different model runs;
      * Save POSTs the wrong field names, so nothing is persisted at all.

    Both are properties of JavaScript that runs, so these tests run it.
    """

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        cls.rendered = run_page(page_script(webui.RUNS_PAGE), cls.node)

    def test_the_panel_reports_the_model_in_force_and_where_it_came_from(self):
        html = self.rendered["model"] or ""
        self.assertIn("inferhub-deepseek-v4.1-flash", html)
        self.assertIn("set here in the console", html,
                      "the source of the default is not stated; the page could "
                      "then present a catalogue value as a saved one")

    def test_the_panel_shows_the_wire_id_so_the_model_is_identifiable(self):
        """The operator looks for the id the provider actually receives."""
        self.assertIn("alicn/deepseek-v4.1-flash-1017",
                      self.rendered["model"] or "")

    def test_the_dropdown_preselects_the_model_that_is_in_force(self):
        """THE failure mode: Save must not mean "change the model".

        The state says the flash alias is in force. If the select defaulted to
        the first alias in the catalogue instead, an operator who opened the
        page and pressed Save would switch models without being told.
        """
        self.assertEqual(self.rendered["mprov"], "inferhub")
        self.assertEqual(self.rendered["mmod"], "inferhub-deepseek-v4.1-flash")

    def test_the_panel_says_a_model_choice_does_not_move_the_token_budget(self):
        """reasoning_effort and max_tokens are process-wide environment
        settings. The panel must not let a model selection imply otherwise."""
        html = self.rendered["model"] or ""
        self.assertIn("max", html)
        self.assertIn("32000", html)

    def test_saving_posts_the_selected_pair_as_json(self):
        out = run_page(page_script(webui.RUNS_PAGE), self.node,
                       after="await saveModel();")
        posts = [r for r in out["requests"]
                 if r["method"] == "POST" and r["url"].endswith("/model")]
        self.assertEqual(len(posts), 1, f"expected exactly one save POST, got "
                                        f"{[r['url'] for r in out['requests']]}")
        body = json.loads(posts[0]["body"])
        self.assertEqual(body, {"provider": "inferhub",
                                "model": "inferhub-deepseek-v4.1-flash"})

    def test_a_refused_save_is_reported_and_not_shown_as_saved(self):
        """A typo must not look like a success.

        The server rejects unusable choices (unknown alias, missing key) and
        leaves the previous one in force. If the page rendered its optimistic
        message, the operator would believe a model was set that will never
        start a mission.
        """
        refused = dict(MODEL_STATE)
        out = run_page(page_script(webui.RUNS_PAGE), self.node,
                       model=refused, model_after={"ok": False,
                                                   "error": "ProviderError: "
                                                            "no such alias"},
                       after="await saveModel();")
        msg = out["mmsg"] or ""
        self.assertIn("NOT saved", msg)
        self.assertIn("no such alias", msg)
        self.assertNotIn("saved:", msg.replace("NOT saved:", ""))

    def test_an_empty_selection_is_refused_without_a_request(self):
        """A save that cannot succeed must not be sent at all.

        Under a half-drawn DOM the selects can exist with no value; posting
        that earns a server rejection, which reads like the operator did
        something wrong rather than the page not being ready.
        """
        out = run_page(page_script(webui.RUNS_PAGE), self.node,
                       after="document.getElementById('mmod').value='';\n"
                             "await saveModel();")
        self.assertIn("choose a provider and a model", out["mmsg"] or "")
        self.assertEqual([r for r in out["requests"] if r["method"] == "POST"],
                         [], "an empty selection was POSTed anyway")

    def test_clearing_sends_explicit_clear_and_says_nothing_was_set(self):
        """The honest no-op: reporting "cleared" when nothing was set would
        train an operator to distrust the message."""
        out = run_page(page_script(webui.RUNS_PAGE), self.node,
                       model_after={"ok": True, "cleared": False},
                       after="await clearModelChoice();")
        posts = [r for r in out["requests"]
                 if r["method"] == "POST" and r["url"].endswith("/model")]
        self.assertEqual(json.loads(posts[0]["body"]), {"clear": True})
        self.assertIn("already in force", out["mmsg"] or "")

    def test_an_unavailable_catalogue_disables_choosing_instead_of_offering_it(self):
        """With no catalogue the page cannot render a model list. Offering a
        Save button anyway would post an empty selection."""
        out = run_page(page_script(webui.RUNS_PAGE), self.node, catalog={})
        html = out["model"] or ""
        self.assertIn("cannot choose a model here", html)
        self.assertNotIn("save as default", html)

    def test_a_broken_default_is_surfaced_on_the_page(self):
        broken = dict(MODEL_STATE, error="ProviderError: catalogue unreadable",
                      provider=None, model=None, model_id=None, source="catalogue")
        out = run_page(page_script(webui.RUNS_PAGE), self.node, model=broken)
        self.assertIn("catalogue unreadable", out["model"] or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
