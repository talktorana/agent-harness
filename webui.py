"""Zero-dependency web chat UI: the harness in a browser tab.

One stdlib HTTP server, one page, polling instead of websockets -- the goal
is a chat surface that works in a container with no extra packages and no
build step, not a framework. Tasks run in background threads; the page polls
for the streamed-so-far transcript and the final answer.

    python webui.py            # http://127.0.0.1:8787
    AGENT_WEBUI_HOST=0.0.0.0 AGENT_WEBUI_PORT=9000 python webui.py

Security posture: this is a SHELL-CAPABLE agent behind a plain HTTP port. Bind
it to loopback, or put it behind your existing authenticated reverse proxy
(the trading stack's Caddy). It deliberately has no auth of its own -- adding
a fake one would be worse than documenting the requirement.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

#: Placeholder the two pages use for their URL prefix. Substituted at serve
#: time from ``config.WEBUI_BASE_PATH`` rather than baked into the constant, so
#: one image can be mounted at the root on a laptop and under ``/agent`` on a
#: shared domain with no rebuild.
_BASE_TOKEN = "__AGENT_BASE__"


def _render(html: str) -> str:
    """Substitute the configured prefix into every URL the page fetches.

    Read from config on each call (not captured at import) so a deployment can
    set it late and a test can vary it.
    """
    base = getattr(config, "WEBUI_BASE_PATH", "")
    return html.replace(_BASE_TOKEN, base or "")


# ======================================================== run history
#
# Scheduled runs happen while nobody is watching, and the console showed only
# the task in THIS tab. So "is it running every 30 minutes?" and "what did it
# actually do at 03:17?" were unanswerable from the UI -- the evidence existed
# on disk and nowhere else.
#
# Two readers, both read-only and both bounded: a run's JSONL can be ~70 KB and
# there are dozens, so the list reads the HEAD and TAIL of each file rather than
# the whole thing. The detail view reads one file, on demand.

def _sessions_dir() -> Path:
    return Path(getattr(config, "SESSIONS_DIR", "") or "sessions")


def _session_files(limit: int) -> List[Path]:
    """Newest first. The filename starts with a sortable UTC timestamp."""
    root = _sessions_dir()
    try:
        files = [p for p in root.glob("*.jsonl") if p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.name, reverse=True)
    return files[:max(1, limit)]


def _session_id(path: Path) -> str:
    """``<stamp>_<id>.jsonl`` -> ``<id>``."""
    stem = path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _read_lines(path: Path, limit: int, *, tail: bool) -> List[Dict[str, Any]]:
    """Parse up to ``limit`` JSONL records from one end of a file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    picked = lines[-limit:] if tail else lines[:limit]
    out: List[Dict[str, Any]] = []
    for line in picked:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue          # a record being written right now
        if isinstance(record, dict):
            out.append(record)
    return out


def _one_line(text: Any, limit: int = 200) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "\u2026"


def _last_activity(path: Path, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the run was DOING when the file stopped growing.

    A missing ``session_end`` now means one thing -- the process went away
    mid-run -- but "what was it doing?" is what identifies the cause. A file
    that stops on ``model_call`` means it died during an API call; stopping on
    ``action`` means it died executing a tool. Without this the console could
    only say "no end record", which is the point where an investigation stops.
    """
    last = {}
    for record in reversed(records):
        if record.get("event") in ("session_start", "session_end"):
            continue
        last = record
        break
    kind = last.get("event") or ""
    tool = str(last.get("tool") or "")
    if kind == "action":
        phrase = "during a tool call" + (f" ({tool})" if tool else "")
    else:
        phrase = {
            "model_call": "during a model call",
            "assistant": "while the model was replying",
            "answer": "after answering",
            "api_retry": "while retrying an API call",
            "tool": "during a tool call",
        }.get(kind, f"on a {kind} event" if kind else "before writing anything")
    return {"last_event": kind or None,
            "last_event_at": last.get("at"),
            "died_during": phrase}


def run_summaries(limit: int = 60) -> List[Dict[str, Any]]:
    """One row per run: when, what it was told, how it ended.

    The status comes from the ``session_end`` record. Absence of one is
    meaningful and is reported as such: the process died mid-run, which is
    exactly the case an operator needs to see and could not previously.
    """
    now = datetime.now(timezone.utc)
    rows: List[Dict[str, Any]] = []
    for path in _session_files(limit):
        head = _read_lines(path, 3, tail=False)
        start = next((r for r in head if r.get("event") == "session_start"), {})
        tail = _read_lines(path, 8, tail=True)
        end = next((r for r in reversed(tail)
                    if r.get("event") == "session_end"), None)
        answer = next((r for r in reversed(tail)
                       if r.get("event") == "answer"), None)

        started = start.get("at") or ""
        if not end:
            status = "running" if _recent(path, now) else "no_end_record"
        else:
            status = str(end.get("status") or "ok")

        row = {
            "session_id": _session_id(path),
            "file": path.name,
            "started": started,
            "status": status,
            "steps": (end or {}).get("steps"),
            "elapsed_sec": (end or {}).get("elapsed_sec"),
            "model": start.get("model"),
            "prompt": str(start.get("task") or ""),
            "prompt_preview": _one_line(start.get("task"), 140),
            "answer_preview": _one_line((answer or {}).get("text"), 200),
            "has_answer": bool(answer),
        }
        if not end:
            row.update(_last_activity(path, tail))
        rows.append(row)
    return rows


def _scheduler_view() -> Dict[str, Any]:
    """The scheduler's verdict on whether a run is live, or why it is unknown.

    Never raises and never guesses: the page is how a human decides whether to
    trust the agent, so a scheduler that cannot answer must say so rather than
    leave the field out (an absent key reads as "no run in flight" to any
    caller doing `.get()`), and must never be reported as idle.
    """
    try:
        import scheduler
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "reason": f"cannot import the scheduler: {type(exc).__name__}: {exc}"}
    try:
        doc = scheduler.status()
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "reason": f"scheduler.status() raised "
                          f"{type(exc).__name__}: {exc}"}
    jobs = doc.get("jobs") or []
    running = [j.get("id") for j in jobs if j.get("running")]
    current = next((j for j in jobs if j.get("running")), None) or \
        (jobs[0] if jobs else {})
    return {
        "available": True,
        "running": bool(running),
        "running_jobs": running,
        "in_progress_sec": current.get("in_progress_sec"),
        "in_progress_since": current.get("in_progress_since"),
        "next_due": current.get("next_due"),
        "paused": bool(doc.get("paused")),
        "any_enabled": bool(doc.get("any_enabled")),
    }


def _recent(path: Path, now: datetime, window_sec: float = 600.0) -> bool:
    """Was this file touched recently enough that the run may still be going?"""
    try:
        return (now - datetime.fromtimestamp(path.stat().st_mtime,
                                             timezone.utc)
                ).total_seconds() < window_sec
    except OSError:
        return False


def _stamp_of(path: Path) -> str:
    """``2026-11-02T04-31-41_ab12cd34.jsonl`` -> ``2026-11-02T04:31:41``.

    The filename uses dashes in the TIME too, so the whole stamp cannot simply
    have dashes replaced: that yields ``2026:11:02T04:31:41``, which is not a
    date anything can parse and was the first thing this helper got wrong.
    """
    stamp = path.stem.split("_", 1)[0]
    date, _, clock = stamp.partition("T")
    return f"{date}T{clock.replace('-', ':')}" if clock else stamp


def retention_facts() -> Dict[str, Any]:
    """What the retention bound actually is, and how far back it reaches.

    Reported, not inferred, and for a reason that cost a real investigation:
    the count bound was 50 on a deployment that runs a mission every thirty
    minutes, so the visible window was about a day. Nothing on the page said so,
    and "where did yesterday go?" is not a question anybody should have to
    answer by reading `storage.prune_sessions`.
    """
    facts: Dict[str, Any] = {
        "keep": getattr(config, "SESSIONS_KEEP", 0),
        "max_bytes": getattr(config, "SESSIONS_MAX_BYTES", 0),
        "max_bytes_mb": getattr(config, "SESSIONS_MAX_MB", 0),
        "runs": 0, "oldest": None, "newest": None, "bytes": 0,
        #: Which bound is actually deleting things. Two knobs can both be set
        #: and only one decides; naming it is the difference between "we keep
        #: 2000 runs" and "we keep a month".
        "over_budget": False,
    }
    try:
        #: Same enumeration the list uses, so the numbers quoted here cannot
        #: disagree with the rows above them. Imported from storage would be a
        #: second source of truth for "what counts as a run".
        paths = _session_files(100000)
    except Exception:  # noqa: BLE001 - a reporting helper must never 500 a page
        return facts
    facts["runs"] = len(paths)
    if paths:
        facts["oldest"] = _stamp_of(paths[-1])
        facts["newest"] = _stamp_of(paths[0])
        total = 0
        for stem in (p.stem for p in paths):
            for suffix in (".jsonl", ".md"):
                try:
                    total += (_sessions_dir() / f"{stem}{suffix}").stat().st_size
                except OSError:
                    pass
        facts["bytes"] = total
        keep = int(facts["keep"] or 0)
        limit = int(facts["max_bytes"] or 0)
        facts["over_budget"] = bool(
            (keep and len(paths) >= keep) or (limit and total >= limit))
    return facts


def run_detail(session_id: str) -> Optional[Dict[str, Any]]:
    """The full prompt, every step, and the final answer of ONE run."""
    for path in _session_files(200):
        if _session_id(path) != session_id and path.stem != session_id:
            continue
        records = _read_lines(path, 100000, tail=False)
        events: List[Dict[str, Any]] = []
        prompt = ""
        answer = ""
        for record in records:
            kind = record.get("event")
            if kind == "session_start":
                prompt = str(record.get("task") or "")
                #: The start record IS the prompt; listing it as a step too
                #: would show a phantom action before anything happened.
                continue
            if kind == "session_end":
                continue
            if kind == "answer":
                answer = str(record.get("text") or "")
            events.append(record)
        return {"session_id": _session_id(path), "file": path.name,
                "prompt": prompt, "answer": answer, "events": events}
    return None

RUNS_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>agent-harness runs</title>
<style>
 body{font:13px/1.5 ui-monospace,Consolas,monospace;margin:0;background:#111;
      color:#ddd}
 #wrap{max-width:1180px;margin:0 auto;padding:16px}
 h3{margin:0 0 4px}
 a{color:#7fd1ff}
 .m{color:#888}.ok{color:#8ee08e}.bad{color:#ff9d9d}.warn{color:#ffd27f}
 #sched{background:#181818;border:1px solid #333;border-radius:6px;padding:10px;
        margin:12px 0}
 #sched .row{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline}
 .pill{padding:1px 8px;border-radius:10px;border:1px solid #444;font-size:12px}
 .pill.on{background:#1f4d2f;border-color:#2a6}
 .pill.off{background:#4d1f1f;border-color:#6a2a2a}
 table{width:100%;border-collapse:collapse;margin-top:8px}
 th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #262626;
       vertical-align:top}
 th{color:#888;font-weight:normal;font-size:12px}
 tr.run{cursor:pointer}
 tr.run:hover{background:#1a1a1a}
 pre{white-space:pre-wrap;background:#181818;border:1px solid #333;
     border-radius:6px;padding:10px;max-height:46vh;overflow:auto;margin:6px 0}
 #detail{margin-top:14px}
 .tab{display:inline-block;padding:3px 10px;border:1px solid #333;
      border-radius:5px;cursor:pointer;margin-right:6px;font-size:12px}
 .tab.sel{background:#2a2a2a;border-color:#555}
 .clip{max-height:3.6em;overflow:hidden}
 .nav{margin:2px 0 10px}
 .nav a{display:inline-block;margin-right:6px;padding:3px 10px;font-size:12px;
        color:#7fd1ff;background:#16202c;border:1px solid #2f4f75;
        border-radius:5px;text-decoration:none}
 .nav a:hover{background:#1e2c3c}
</style></head><body><div id="wrap">
<h3>agent-harness <span class="m">schedule &amp; runs</span></h3>
<div class="nav">
  <a href="__AGENT_BASE__/live">live view</a>
  <a href="__AGENT_BASE__/">console</a>
</div>
<div id="sched" class="m">loading schedule...</div>
<div id="err"></div>
<div id="keep" class="m"></div>
<table><thead><tr>
 <th style="width:150px">started (UTC)</th><th style="width:90px">status</th>
 <th style="width:56px">steps</th><th style="width:70px">took</th>
 <th>prompt / task</th><th>answer</th>
</tr></thead><tbody id="runs"></tbody></table>
<div id="detail"></div>
<script>
const NICE={ok:'ok',success:'ok',running:'running',no_end_record:'no end record',
            error:'error',failed:'failed',timeout:'timeout',crashed:'crashed',
            step_limit_with_answer:'step limit (partial)',
            step_limit:'step limit (no answer)'};
// 'step_limit_with_answer' really occurs: the agent ran out of steps but still
// produced a report. Colouring that red would train an operator to ignore red,
// which is the opposite of what a status column is for.
function cls(s){return s==='ok'||s==='success'?'ok':
                    (s==='running'||s==='step_limit_with_answer'?'warn':'bad');}
function ago(iso){
  if(!iso)return '';
  const t=Date.parse(iso); if(isNaN(t))return iso;
  const d=Math.max(0,Math.round((Date.now()-t)/1000));
  if(d<60)return d+'s ago';
  if(d<3600)return Math.round(d/60)+'m ago';
  return Math.round(d/3600)+'h '+Math.round((d%3600)/60)+'m ago';
}
function secs(v){return (v==null)?'':
  (v<60?v.toFixed(0)+'s':(v/60).toFixed(1)+'m');}
function esc(s){return (s==null?'':String(s))
  // Quotes too, not just angle brackets: this value is interpolated into the
  // onclick attribute below, and an id containing a quote would otherwise end
  // the attribute and inject markup. Escaping them is one line and closes the
  // class rather than the instance.
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');}

async function loadSchedule(){
  let j=null;
  try{const r=await fetch('__AGENT_BASE__/schedule');j=await r.json();}catch(e){}
  const el=document.getElementById('sched');
  if(!j){el.textContent='schedule unavailable (scheduler not readable here)';
          el.className='bad';return;}
  let h='';
  if(j.paused){
    h+='<div class="bad">PAUSED by an operator hold file'
      +(j.hold&&j.hold.reason?' -- '+esc(j.hold.reason):'')
      +(j.hold&&j.hold.since?' (since '+esc(j.hold.since)+')':'')+'</div>';
  }
  if(!j.jobs||!j.jobs.length){
    h+='<div class="warn">no jobs declared in '+esc(j.schedule_file)+'</div>';
  }
  (j.jobs||[]).forEach(function(job){
    const statePill=job.enabled?'<span class="pill on">enabled</span>'
                               :'<span class="pill off">disabled</span>';
    let next='';
    if(job.next_due){
      const s=job.seconds_until;
      // Say WHAT the stamp means. A future next_due next to "running now" is
      // otherwise ambiguous -- the 09:20 pass read both and reported its own
      // next-wake time as UNVERIFIED -- so the label states the basis instead
      // of leaving the reader to work it out.
      next='next due <b>'+(s!=null&&s>0?(s>=60?Math.round(s/60)+'m '+Math.round(s%60)+'s':
            s+'s'):'now')+'</b> <span class="m">('+esc(job.next_due)+')</span>'
        +' <span class="m">= last run + '+esc(job.every_min||'?')+' min</span>';
      if(job.overdue) next+=' <span class="warn">overdue</span>';
    }
    h+='<div class="row"><b>'+esc(job.id)+'</b>'
      +'<span>'+esc(job.interval)+'</span>'+statePill
      +'<span>last run <b>'+esc(job.last_run||'never')+'</b>'
      +(job.last_run?' <span class="m">('+ago(job.last_run)+')</span>':'')+'</span>'
      // A run in progress has no duration yet: finished still describes the
      // PREVIOUS run, and showing a difference would print a negative number.
      // It gets its OWN elapsed clock instead, so "running now" is a fact with
      // a number next to it rather than a word to interpret.
      +(job.running?('<span class="warn">running now'
        +(job.in_progress_sec!=null?' for <b>'+secs(job.in_progress_sec)+'</b>':'')
        +(job.in_progress_since?' <span class="m">(since '
          +esc(job.in_progress_since)+')</span>':'')
        +'</span>')
        :(job.duration_sec!=null?'<span>last took <b>'+secs(job.duration_sec)
          +'</b></span>':''))
      +(job.last_run?'<span>last outcome <span class="'+cls(job.last_ok?'ok':'bad')
        +'">'+(job.last_ok===false?'FAILED':'ok')+'</span></span>':'')
      +'<span>'+next+'</span></div>';
    if(job.last_error) h+='<div class="bad">last error: '+esc(job.last_error)+'</div>';
    h+='<div class="m" style="margin-top:4px">prompt sent each run ('
      +job.prompt_chars+' chars): '+(job.prompt
        ? '<span class="clip">'+esc(job.prompt_preview)+'</span>'
        : '<span class="bad">EMPTY</span>')+'</div>';
    if(job.prompt){
      h+='<div><span class="tab" onclick="tog(this)">show full prompt</span>'
        +'<div style="display:none"><pre>'+esc(job.prompt)+'</pre></div></div>';
    }
  });
  el.innerHTML=h; el.className='';
  document.getElementById('schedh').textContent='';
}
function tog(el){
  const body=el.nextElementSibling;
  const open=body.style.display!=='none';
  body.style.display=open?'none':'block';
  el.textContent=open?'show full prompt':'hide prompt';
}

async function loadRuns(){
  // `held` is declared HERE, beside `rows`, and NOT with `const` inside the try.
  //
  // It was `const j = await r.json()` inside the try and then `j.held` after it,
  // so the retention line referenced a block-scoped name that no longer exists
  // outside the block: a ReferenceError, thrown on every load, which aborted
  // loadRuns BEFORE the table was filled. The page therefore rendered "no runs"
  // for every operator while /runs.json happily served 54 of them -- a display
  // defect that makes a healthy system look wiped. Nothing caught it because
  // this page had no test that ever EXECUTED its script.
  let rows=[], held={};
  try{const r=await fetch('__AGENT_BASE__/runs.json');const j=await r.json();
      rows=j.runs||[]; held=j.held||{};}catch(e){
    document.getElementById('err').innerHTML=
      '<div class="bad">could not read runs: '+esc(e)+'</div>';return;}
  const tb=document.getElementById('runs');
  // Say HOW MUCH history exists and WHY it stops where it does. The page used
  // to show a list and nothing else, so an operator who noticed that yesterday
  // was missing had to read the retention code to find out whether the data was
  // lost, never written, or deleted on purpose. Naming the oldest run and the
  // bound turns that into one line they can act on.
  const keep=document.getElementById('keep');
  if(keep){
    keep.textContent=(held.keep||held.max_bytes)
      ? ('history: '+rows.length+' run(s)'+(held.oldest?' back to '
          +String(held.oldest).replace('T',' ').slice(0,16)+' UTC':'')
         +'\u2014older runs are deleted by retention (keep '+held.keep
         +' runs, '+held.max_bytes_mb+' MB budget)'
         +(held.over_budget?' \u2014 byte budget is the binding limit':''))
      : '';
  }
  if(!rows.length){tb.innerHTML='<tr><td colspan="6" class="m">'
    +'no runs on disk yet</td></tr>';return;}
  tb.innerHTML=rows.map(function(r){
    // An unfinished run says what it was DOING when the file stopped growing:
    // 'no end record' alone is where an investigation stops, and the last event
    // is what distinguishes a container restart from a crash.
    const died=(r.status==='no_end_record'&&r.died_during)
      ? '<div class="bad">died '+esc(r.died_during)
        +(r.last_event_at?' at '+esc(r.last_event_at.slice(11,19)):'')+'</div>'
      : '';
    // The id travels in a DATA ATTRIBUTE, and the handler reads it back.
    //
    // This was built by concatenating quotes into the onclick string:
    //     '...onclick="showRun(\''+r.session_id+'\')">'
    // The `\'` survives Python as a plain `'`, so the SERVED script read
    //     '...showRun(''+r.session_id+'')...'
    // -- three adjacent string literals with no operators between them, a
    // SyntaxError that stopped the WHOLE page script from parsing. Not one
    // line failed: nothing ran at all, no schedule, no rows, and an operator
    // saw an empty table with data sitting behind it (reported 2026-11-02 as
    // "no history for the past two hours").
    //
    // A data attribute has no nested quotes to get wrong, and the value is
    // escaped, so it cannot break out of the attribute either.
    return '<tr class="run" data-sid="'+esc(r.session_id)+'"'
      +' onclick="showRun(this.dataset.sid)">'
      +'<td>'+esc(r.started?r.started.replace('T',' ').slice(0,19):'?')
      +'<div class="m">'+ago(r.started)+'</div></td>'
      +'<td class="'+cls(r.status)+'">'+esc(NICE[r.status]||r.status)+died
      +'</td>'
      +'<td>'+(r.steps==null?'':r.steps)+'</td>'
      +'<td>'+(r.elapsed_sec==null?'':secs(r.elapsed_sec))+'</td>'
      +'<td class="m">'+(r.prompt_preview?'<span class="clip">'
          +esc(r.prompt_preview)+'</span>':'<span class="bad">(no prompt)</span>')
      +'<div class="m">'+(r.model?esc(r.model):'')+'</div></td>'
      +'<td class="m">'+(r.has_answer?esc(r.answer_preview)
          :'<span class="warn">no answer recorded</span>')+'</td></tr>';
  }).join('');
}

async function showRun(id){
  const el=document.getElementById('detail');
  el.innerHTML='<div class="m">loading '+esc(id)+'...</div>';
  let j=null;
  try{const r=await fetch('__AGENT_BASE__/run/'+encodeURIComponent(id));
      j=await r.json();}catch(e){}
  if(!j||j.error){el.innerHTML='<div class="bad">run not found: '+esc(id)
    +'</div>';return;}
  let h='<h3 style="margin-top:16px">run <span class="m">'+esc(j.session_id)
    +'</span> <span class="m" style="font-size:12px">'+esc(j.file)+'</span></h3>';
  h+='<div class="m">prompt sent to the agent</div><pre>'
    +(j.prompt?esc(j.prompt):'<span class="bad">EMPTY</span>')+'</pre>';
  h+='<div class="m">final answer</div><pre>'
    +(j.answer?esc(j.answer):'<span class="warn">none recorded</span>')+'</pre>';
  h+='<div class="m">steps and tool calls</div>';
  (j.events||[]).forEach(function(e){
    if(e.event==='action'){
      h+='<div class="'+((e.ok===false)?'bad':'ok')+'">'
        +'step '+(e.step)+'  '+esc(e.tool)
        +(e.seconds!=null?'  '+Number(e.seconds).toFixed(1)+'s':'')+'</div>';
      if(e.detail) h+='<div class="m">  '+esc(e.detail)+'</div>';
      if(e.result_preview) h+='<pre style="max-height:12em">'
        +esc(e.result_preview)+'</pre>';
    } else if(e.event==='assistant'){
      h+='<div class="m">says: '+esc(e.text)+'</div>';
    } else if(e.event==='model_call'){
      h+='<div class="m">model '+esc(e.model)+' attempt '+(e.attempt)+'</div>';
    } else if(e.event==='api_retry'){
      h+='<div class="warn">retry: '+esc(e.error||'')+'</div>';
    }
  });
  el.innerHTML=h;
}

loadSchedule(); loadRuns();
setInterval(loadSchedule,15000); setInterval(loadRuns,30000);
</script></div></body></html>"""


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>agent-harness</title>
<style>
 body{font:14px/1.5 ui-monospace,Consolas,monospace;margin:0;background:#111;
      color:#ddd}
 #wrap{max-width:960px;margin:0 auto;padding:16px}
 #log{white-space:pre-wrap;background:#181818;border:1px solid #333;
      border-radius:6px;padding:12px;min-height:60vh;max-height:70vh;
      overflow:auto}
 .u{color:#7fd1ff}.a{color:#b8f5b0}.e{color:#ff9d9d}.m{color:#888}
 form{display:flex;gap:8px;margin-top:12px}
 textarea{flex:1;background:#181818;color:#ddd;border:1px solid #333;
          border-radius:6px;padding:8px;font:inherit;min-height:64px}
 button{background:#2a6;background:#1f7a4d;color:#fff;border:0;
        border-radius:6px;padding:8px 16px;font:inherit;cursor:pointer}
 button:disabled{opacity:.4;cursor:default}
 .nav{margin:2px 0 10px}
 .nav a{display:inline-block;margin-right:6px;padding:3px 10px;font-size:12px;
        color:#7fd1ff;background:#16202c;border:1px solid #2f4f75;
        border-radius:5px;text-decoration:none}
 .nav a:hover{background:#1e2c3c}
</style></head><body><div id="wrap">
<h3>agent-harness <span class="m" id="state"></span></h3>
<div class="nav">
  <a href="__AGENT_BASE__/live">live view</a>
  <a href="__AGENT_BASE__/runs">schedule &amp; runs</a>
</div>
<div id="pick" class="m">
 <select id="prov"></select><select id="mod"></select>
 <span id="routes" style="font-size:11px"></span>
</div>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="task..."></textarea>
<button id="go">send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),
      go=document.getElementById('go'),state=document.getElementById('state'),
      prov=document.getElementById('prov'),mod=document.getElementById('mod');
let cur=null,timer=null,CAT=null;
function fillModels(){
  mod.innerHTML='';
  const p=CAT&&CAT.providers[prov.value], ids=(p&&p.model_ids)||{};
  ((p&&p.models)||[]).forEach(m=>{
    const o=document.createElement('option');o.value=m;
    // Show the real model id beside the alias: the alias alone hides which
    // model is actually selected on the wire.
    o.textContent=(ids[m]&&ids[m]!==m)?(m+' \u2192 '+ids[m]):m;
    mod.appendChild(o);});
  if(CAT&&CAT.default&&CAT.default[0]===prov.value&&CAT.default[1]){
    mod.value=CAT.default[1];}
}
async function loadCatalog(){
  try{const r=await fetch('__AGENT_BASE__/models');CAT=await r.json();}catch(e){CAT=null;}
  if(!CAT)return;
  prov.innerHTML='';
  Object.keys(CAT.providers).forEach(name=>{
    const o=document.createElement('option');o.value=name;
    o.textContent=name+(CAT.providers[name].key_configured?'':' (no key)');
    prov.appendChild(o);});
  if(CAT.default&&CAT.default[0])prov.value=CAT.default[0];
  fillModels();
  const rt=document.getElementById('routes');
  if(rt&&CAT.routes&&CAT.routes.length)rt.textContent='chain: '+CAT.routes.join(' \\u2192 ');
}
prov.onchange=fillModels;
loadCatalog();
function add(cls,text){const d=document.createElement('div');d.className=cls;
  d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function poll(){
  if(!cur)return;
  const r=await fetch('__AGENT_BASE__/task/'+cur);const j=await r.json();
  log.innerHTML='';(j.lines||[]).forEach(l=>add(l.role,l.text));
  if(j.stream){add('a',j.stream+'\u258c');}
  state.textContent=j.status;
  if(j.status==='done'||j.status==='error'){clearInterval(timer);cur=null;
    go.disabled=false;return;}
}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text||cur)return;
  q.value='';add('u','> '+text);go.disabled=true;state.textContent='queued';
  const r=await fetch('__AGENT_BASE__/chat',{method:'POST',
    headers:{'content-type':'application/json'},
    body:JSON.stringify({text, provider:prov.value||null,
                         model:mod.value||null})});
  const j=await r.json();cur=j.id;timer=setInterval(poll,1500);poll();
};
</script></div></body></html>"""


LIVE_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>agent-harness live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font:14px/1.5 ui-monospace,Consolas,monospace;margin:0;background:#0d1117;
      color:#ddd}
 #wrap{max-width:1000px;margin:0 auto;padding:12px}
 header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
        border-bottom:1px solid #222;padding-bottom:8px;margin-bottom:10px}
 h3{margin:0}
 select{background:#161b22;color:#ddd;border:1px solid #333;border-radius:6px;
        padding:4px;font:inherit}
 .m{color:#8b949e;font-size:12px}
 #chain{font-size:11px;color:#8b949e}
 #badge{font-size:11px;color:#d2a8ff;border:1px solid #3b2d52;background:#1c1529;
        border-radius:10px;padding:2px 10px;display:none}
 #msgs{background:#11151b;border:1px solid #222;border-radius:8px;padding:12px;
       min-height:52vh;max-height:64vh;overflow:auto}
 .row{display:flex;margin:8px 0}
 .row.u{justify-content:flex-end}
 .bub{max-width:85%;white-space:pre-wrap;border-radius:10px;padding:8px 12px}
 .row.u .bub{background:#1f3a5f;color:#cfe6ff}
 .row.a .bub{background:#16211a;color:#b8f5b0}
 .row.e .bub{background:#2d1618;color:#ff9d9d}
 .tools{margin:6px 0}
 .tools summary{cursor:pointer;color:#8b949e;font-size:12px}
 .tools pre{margin:4px 0 4px 12px;color:#7ee787;font-size:11px;
            white-space:pre-wrap;max-height:220px;overflow:auto}
 .caret::after{content:'\\258c';animation:bl 1s steps(1) infinite;color:#7fd1ff}
 @keyframes bl{50%{opacity:0}}
 form{display:flex;gap:8px;margin-top:10px}
 textarea{flex:1;background:#11151b;color:#ddd;border:1px solid #333;
          border-radius:6px;padding:8px;font:inherit;min-height:56px}
 button{background:#1f7a4d;color:#fff;border:0;border-radius:6px;
        padding:6px 14px;font:inherit;cursor:pointer}
 button.sec{background:#30363d}
 button.danger{background:#8d2b2b}
 button:disabled{opacity:.35;cursor:default}
 #bar{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap}
 label.as{font-size:12px;color:#8b949e;display:flex;gap:4px;align-items:center}
</style></head><body><div id="wrap">
<header>
 <h3>live</h3>
 <select id="prov"></select><select id="mod"></select>
 <span id="badge"></span>
 <span class="m" id="state"></span>
 <span id="chain"></span>
 <a href="__AGENT_BASE__/" class="m">simple view</a>
 <a href="__AGENT_BASE__/runs" style="font-size:12px;color:#7fd1ff;background:#16202c;border:1px solid #2f4f75;border-radius:5px;padding:2px 9px;text-decoration:none">schedule &amp; runs</a>
</header>
<div id="msgs"></div>
<form id="f"><textarea id="q" placeholder="task... (Ctrl+Enter to send)"></textarea>
<button id="go">send</button>
<button id="stop" class="danger" disabled>stop</button></form>
<div id="bar">
 <button id="copy" class="sec" disabled>copy answer</button>
 <button id="regen" class="sec" disabled>regenerate</button>
 <button id="clear" class="sec">clear</button>
 <label class="as"><input type="checkbox" id="as" checked>auto-scroll</label>
</div>
<script>
const msgs=document.getElementById('msgs'),q=document.getElementById('q'),
 go=document.getElementById('go'),stopB=document.getElementById('stop'),
 state=document.getElementById('state'),badge=document.getElementById('badge'),
 prov=document.getElementById('prov'),mod=document.getElementById('mod'),
 as=document.getElementById('as'),chain=document.getElementById('chain');
let cur=null,timer=null,CAT=null,lastText='',lastAnswer='',liveBub=null,liveTools=null;
function fillModels(){
 mod.innerHTML='';
 const p=CAT&&CAT.providers[prov.value], ids=(p&&p.model_ids)||{};
 ((p&&p.models)||[]).forEach(m=>{
  const o=document.createElement('option');o.value=m;
  o.textContent=(ids[m]&&ids[m]!==m)?(m+' \u2192 '+ids[m]):m;
  mod.appendChild(o);});
 if(CAT&&CAT.default&&CAT.default[0]===prov.value&&CAT.default[1])
  mod.value=CAT.default[1];
}
async function loadCatalog(){
 try{const r=await fetch('__AGENT_BASE__/models');CAT=await r.json();}catch(e){CAT=null;}
 if(!CAT)return;
 prov.innerHTML='';
 Object.keys(CAT.providers).forEach(name=>{
  const o=document.createElement('option');o.value=name;
  o.textContent=name+(CAT.providers[name].key_configured?'':' (no key)');
  prov.appendChild(o);});
 if(CAT.default&&CAT.default[0])prov.value=CAT.default[0];
 fillModels();
 if(CAT.routes&&CAT.routes.length)
  chain.textContent='route chain: '+CAT.routes.join(' \\u2192 ');
}
prov.onchange=fillModels;
loadCatalog();
function scroll(){if(as.checked)msgs.scrollTop=msgs.scrollHeight;}
function row(cls){const d=document.createElement('div');d.className='row '+cls;
 const b=document.createElement('div');b.className='bub';d.appendChild(b);
 msgs.appendChild(d);scroll();return b;}
function evLine(evt){
 const name=evt.event||'?';let s=name;
 if(evt.route)s+=' ['+evt.route+']';
 if(evt.tool)s+=' '+evt.tool;
 if(evt.to)s+=' \\u2192 '+evt.to;
 if(evt.reasoning_chars)s+=' '+evt.reasoning_chars+' reasoning chars';
 if(evt.error)s+=' :: '+String(evt.error).slice(0,140);
 return s;
}
function renderEvents(evs){
 if(!evs||!evs.length)return;
 if(!liveTools){
  const det=document.createElement('details');det.className='tools';
  const sm=document.createElement('summary');sm.textContent='agent activity';
  det.appendChild(sm);
  const pre=document.createElement('pre');det.appendChild(pre);
  const d=document.createElement('div');d.className='row a';
  d.appendChild(det);msgs.appendChild(d);
  liveTools={det,pre,count:0};
 }
 let added=false;
 evs.slice(liveTools.count).forEach(e=>{
  liveTools.pre.textContent+=evLine(e)+'\\n';liveTools.count++;added=true;
  if(e.event==='model_call'&&e.route){badge.textContent=e.route;
   badge.style.display='inline-block';}
  if(e.event==='route_switch'||e.event==='route_dead')
   {liveTools.det.open=true;}
 });
 if(added){liveTools.det.querySelector('summary').textContent=
  'agent activity ('+liveTools.count+')';scroll();}
}
async function poll(){
 if(!cur)return;
 try{
  const r=await fetch('__AGENT_BASE__/task/'+cur);const j=await r.json();
  renderEvents(j.events);
  if(j.stream){
   if(!liveBub)liveBub=row('a');
   liveBub.className='bub caret';liveBub.textContent=j.stream;scroll();
  }
  state.textContent=j.status;
  // A reasoning model streams reasoning_content for minutes before the first
  // answer token. Nothing to render then used to leave the page showing the
  // question and a silent "running" -- read, correctly, as a dead process.
  // Say what it is actually doing, with a number that keeps moving.
  if(!j.stream&&(j.status==='running'||j.status==='queued')){
   const ev=((j.events||[]).filter(e=>e.event==='thinking'
    ||e.event==='model_call')).pop();
   if(ev)state.textContent='thinking'+(ev.reasoning_chars?
    ' ('+ev.reasoning_chars+' reasoning chars)':'')+'...';
  }
  stopB.disabled=(j.status!=='running'&&j.status!=='stopping');
  go.disabled=(j.status==='running'||j.status==='stopping'||j.status==='queued');
  if(j.status==='done'||j.status==='error'||j.status==='stopped'){
   clearInterval(timer);timer=null;cur=null;
   const lines=j.lines||[];
   const last=lines.length?lines[lines.length-1]:null;
   if(last&&last.role==='a')lastAnswer=last.text;
   // The answer bubble only exists if some poll happened to catch a partial
   // stream. A task that FINISHES BETWEEN POLLS therefore left the transcript
   // showing the question and nothing else (live 2026-10-30: the answer was
   // in the payload and in lastAnswer, but never on screen). The final line
   // is authoritative, so render it whether or not a stream was ever seen.
   if(last&&last.text){
    if(!liveBub){liveBub=row(last.role);}
    else if(last.role==='e'){liveBub.parentNode.className='row e';}
    liveBub.className='bub';
    liveBub.textContent=last.text;
    if(last.role==='a')lastAnswer=last.text;
   }
   if(j.status!=='done'&&!last&&j.stream)lastAnswer=j.stream;
   scroll();
   document.getElementById('copy').disabled=!lastAnswer;
   document.getElementById('regen').disabled=!lastText;
   liveTools=null;liveBub=null;
  }
 }catch(e){}
}
async function send(text){
 if(!text||cur)return;
 lastText=text;q.value='';
 row('u').textContent=text;
 go.disabled=true;stopB.disabled=false;state.textContent='queued';
 lastAnswer='';
 const r=await fetch('__AGENT_BASE__/chat',{method:'POST',
  headers:{'content-type':'application/json'},
  body:JSON.stringify({text,provider:prov.value||null,model:mod.value||null})});
 const j=await r.json();cur=j.id;
 if(timer)clearInterval(timer);
 timer=setInterval(poll,800);poll();
}
document.getElementById('f').onsubmit=e=>{e.preventDefault();
 send(q.value.trim());};
q.addEventListener('keydown',e=>{
 if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();
  send(q.value.trim());}});
stopB.onclick=async()=>{if(!cur)return;
 await fetch('__AGENT_BASE__/stop/'+cur,{method:'POST'});state.textContent='stopping...';};
document.getElementById('copy').onclick=()=>{
 if(!lastAnswer)return;
 (navigator.clipboard?navigator.clipboard.writeText(lastAnswer)
  :Promise.reject()).then(()=>{
  document.getElementById('copy').textContent='copied!';
  setTimeout(()=>document.getElementById('copy').textContent='copy answer',900);
 }).catch(()=>{window.prompt('copy:',lastAnswer);});};
document.getElementById('regen').onclick=()=>send(lastText);
document.getElementById('clear').onclick=()=>{msgs.innerHTML='';
 liveBub=null;liveTools=null;lastAnswer='';
 document.getElementById('copy').disabled=true;};
</script></div></body></html>"""


def _start_task(text: str, provider=None, model=None) -> str:
    from agent import Agent

    task_id = uuid.uuid4().hex[:10]
    record = {"id": task_id, "status": "running", "lines": [], "answer": "",
              "events": [], "provider": provider, "model": model,
              "agent": None}
    with _LOCK:
        _TASKS[task_id] = record

    def work():
        def on_token(token: str) -> None:
            with _LOCK:
                record["stream"] = record.get("stream", "") + token

        def on_event(evt: Dict[str, Any]) -> None:
            # Cap the feed: long tasks log thousands of events and the poll
            # payload (and browser) must stay bounded.
            with _LOCK:
                feed = record["events"]
                feed.append(evt)
                if len(feed) > 400:
                    del feed[:len(feed) - 400]

        # Construction is INSIDE the try. Agent() resolves the route chain and
        # raises ProviderError for a missing key or an unknown alias; outside
        # the try that exception killed the worker thread and left the record
        # at "running" forever -- the page polled a spinner that never resolved
        # and never said what was misconfigured.
        agent = None
        try:
            agent = Agent(verbose=False, on_token=on_token, on_event=on_event,
                          provider=provider, model=model)
            record["agent"] = agent
            answer = agent.run(text)
            record["answer"] = answer
            record["stream"] = ""
            record["lines"].append({"role": "a", "text": answer})
            record["status"] = "stopped" if agent.stop_requested else "done"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["stream"] = ""
            record["lines"].append(
                {"role": "e", "text": f"{type(exc).__name__}: {exc}"})
        finally:
            record["agent"] = None
    threading.Thread(target=work, daemon=True).start()
    return task_id


def _stop_task(task_id: str) -> bool:
    """Request a cooperative stop. The agent finishes the current tool call
    and exits before its next model turn -- never mid-write."""
    with _LOCK:
        record = _TASKS.get(task_id)
        if not record or record["status"] != "running":
            return False
        record["status"] = "stopping"
        agent = record.get("agent")
    if agent is not None:
        agent.stop_requested = True
    return True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self, html: str):
        self._send(200, _render(html).encode(), "text/html; charset=utf-8")

    def _route(self) -> str:
        """The request path with the configured prefix removed.

        Returns ``""`` for anything OUTSIDE the prefix, which is how a console
        mounted at ``/agent`` refuses to also answer at ``/models``. Answering
        both would mean two paths to the same shell-capable agent, only one of
        which the reverse proxy authenticates -- and the one that is not the
        documented path is the one nobody reviews.
        """
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        base = getattr(config, "WEBUI_BASE_PATH", "")
        if not base:
            return path
        if path == base:
            return "/"                       # "/agent" means the console root
        if path.startswith(base + "/"):
            return path[len(base):] or "/"
        return ""

    def do_GET(self):  # noqa: N802
        route = self._route()
        if route in ("/", "/index.html"):
            self._send_page(PAGE)
        elif route in ("/live", "/live/"):
            self._send_page(LIVE_PAGE)
        elif route in ("/runs", "/runs/"):
            self._send_page(RUNS_PAGE)
        elif route == "/schedule":
            #: Read live rather than cached: the whole point is to see whether
            #: the loop is alive RIGHT NOW, and a stale answer would be the
            #: same blindness the page exists to fix.
            import scheduler
            try:
                payload = scheduler.status()
            except Exception as exc:  # noqa: BLE001
                payload = {"error": f"{type(exc).__name__}: {exc}",
                           "jobs": [], "paused": False}
            self._send(200, json.dumps(payload, default=str).encode(),
                       "application/json")
        elif route == "/runs.json":
            limit = 60
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            for part in query.split("&"):
                if part.startswith("limit="):
                    try:
                        limit = max(1, min(500, int(part.split("=", 1)[1])))
                    except ValueError:
                        pass
            payload = {"runs": run_summaries(limit),
                       "sessions_dir": str(_sessions_dir()),
                       "held": retention_facts(),
                       #: THE SCHEDULER'S OWN ANSWER, next to this page's.
                       #:
                       #: The two are computed differently and CAN disagree:
                       #: this page calls a run live while its session file was
                       #: touched in the last 600 s, and the scheduler calls it
                       #: live while last_run has no matching finished. A run
                       #: here lasts 383-470 s, so a finished run keeps reading
                       #: as `running` for minutes after it ended -- and the
                       #: mission brief told a pass "ask the scheduler first; it
                       #: is the same verdict the console shows", which was
                       #: simply untrue.
                       #:
                       #: The dangerous direction is the other one: this page
                       #: says `no_end_record` (idle) once the window lapses,
                       #: while the scheduler may still be mid-run -- and a pass
                       #: that trusts only this field would recreate the sidecar
                       #: and kill it. So both are published, and their
                       #: disagreement is stated rather than left to be noticed.
                       "scheduler": _scheduler_view()}
            self._send(200, json.dumps(payload, default=str).encode(),
                       "application/json")
        elif route.startswith("/run/"):
            wanted = route.split("/run/", 1)[1].strip("/")
            detail = run_detail(wanted) if wanted else None
            if detail is None:
                self._send(404, json.dumps(
                    {"error": "unknown run", "session_id": wanted}
                ).encode(), "application/json")
            else:
                self._send(200, json.dumps(detail, default=str).encode(),
                           "application/json")
        elif route.startswith("/task/"):
            task_id = route.split("/task/", 1)[1]
            with _LOCK:
                record = _TASKS.get(task_id)
                payload = dict(record) if record else None
            if payload is None:
                self._send(404, b'{"error":"unknown task"}',
                           "application/json")
            else:
                payload.pop("agent", None)
                self._send(200, json.dumps(payload, default=str).encode(),
                           "application/json")
        elif route == "/models":
            import llm
            try:
                payload = llm.catalog()
            except Exception as exc:  # noqa: BLE001
                payload = {"providers": {}, "default": [],
                           "error": str(exc)[:300]}
            self._send(200, json.dumps(payload).encode(),
                       "application/json")
        elif route == "/health":
            self._send(200, json.dumps({"ok": True,
                                        "tasks": len(_TASKS)}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        route = self._route()
        if route.startswith("/stop/"):
            task_id = route.split("/stop/", 1)[1]
            ok = _stop_task(task_id)
            self._send(200, json.dumps({"stopped": ok}).encode(),
                       "application/json")
            return
        if route != "/chat":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
            text = (body.get("text") or "").strip()
            provider = body.get("provider") or None
            model = body.get("model") or None
        except ValueError:
            text, provider, model = "", None, None
        if not text:
            self._send(400, b'{"error":"empty text"}', "application/json")
            return
        self._send(200,
                   json.dumps({"id": _start_task(text, provider,
                                                  model)}).encode(),
                   "application/json")


def main() -> int:
    server = ThreadingHTTPServer((config.WEBUI_HOST, config.WEBUI_PORT),
                                 Handler)
    base = getattr(config, "WEBUI_BASE_PATH", "")
    #: base is "" or "/agent", so this prints the streaming page's real URL
    #: either way -- the one line worth having when you are checking whether a
    #: proxy is pointed at the right path.
    print(f"[webui] http://{config.WEBUI_HOST}:{config.WEBUI_PORT}{base}/live",
          flush=True)
    if base:
        # Say it plainly: with a prefix set, the root of this port is a dead
        # end, and a reader who curls "/" and gets a 404 would otherwise spend
        # the next few minutes wondering whether the console died.
        print(f"[webui] mounted under {base!r}; unprefixed paths are refused "
              f"by design, so a proxy must forward the prefix unchanged.",
              flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
