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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import config

_TASKS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

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
</style></head><body><div id="wrap">
<h3>agent-harness <span class="m" id="state"></span></h3>
<div id="log"></div>
<form id="f"><textarea id="q" placeholder="task..."></textarea>
<button id="go">send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),
      go=document.getElementById('go'),state=document.getElementById('state');
let cur=null,timer=null;
function add(cls,text){const d=document.createElement('div');d.className=cls;
  d.textContent=text;log.appendChild(d);log.scrollTop=log.scrollHeight;}
async function poll(){
  if(!cur)return;
  const r=await fetch('/task/'+cur);const j=await r.json();
  log.innerHTML='';(j.lines||[]).forEach(l=>add(l.role,l.text));
  if(j.stream){add('a',j.stream+'\u258c');}
  state.textContent=j.status;
  if(j.status==='done'||j.status==='error'){clearInterval(timer);cur=null;
    go.disabled=false;return;}
}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text||cur)return;
  q.value='';add('u','> '+text);go.disabled=true;state.textContent='queued';
  const r=await fetch('/chat',{method:'POST',
    headers:{'content-type':'application/json'},body:JSON.stringify({text})});
  const j=await r.json();cur=j.id;timer=setInterval(poll,1500);poll();
};
</script></div></body></html>"""


def _start_task(text: str) -> str:
    from agent import Agent

    task_id = uuid.uuid4().hex[:10]
    record = {"id": task_id, "status": "running", "lines": [], "answer": ""}
    with _LOCK:
        _TASKS[task_id] = record

    def work():
        def on_token(token: str) -> None:
            with _LOCK:
                record["stream"] = record.get("stream", "") + token
        agent = Agent(verbose=False, on_token=on_token)
        try:
            answer = agent.run(text)
            record["answer"] = answer
            record["stream"] = ""
            record["lines"].append({"role": "a", "text": answer})
            record["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["stream"] = ""
            record["lines"].append(
                {"role": "e", "text": f"{type(exc).__name__}: {exc}"})
    threading.Thread(target=work, daemon=True).start()
    return task_id


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

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/task/"):
            task_id = self.path.split("/task/", 1)[1]
            with _LOCK:
                record = _TASKS.get(task_id)
                payload = dict(record) if record else None
            if payload is None:
                self._send(404, b'{"error":"unknown task"}',
                           "application/json")
            else:
                self._send(200, json.dumps(payload).encode(),
                           "application/json")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True,
                                        "tasks": len(_TASKS)}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/chat":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            text = (json.loads(raw or b"{}").get("text") or "").strip()
        except ValueError:
            text = ""
        if not text:
            self._send(400, b'{"error":"empty text"}', "application/json")
            return
        self._send(200, json.dumps({"id": _start_task(text)}).encode(),
                   "application/json")


def main() -> int:
    server = ThreadingHTTPServer((config.WEBUI_HOST, config.WEBUI_PORT),
                                 Handler)
    print(f"[webui] http://{config.WEBUI_HOST}:{config.WEBUI_PORT}",
          flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
