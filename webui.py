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
<h3>agent-harness <span class="m" id="state"></span>
  <a href="/live" style="color:#7fd1ff;font-size:12px">live view -></a></h3>
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
  const p=CAT&&CAT.providers[prov.value];
  ((p&&p.models)||[]).forEach(m=>{
    const o=document.createElement('option');o.value=m;o.textContent=m;
    mod.appendChild(o);});
  if(CAT&&CAT.default&&CAT.default[0]===prov.value&&CAT.default[1]){
    mod.value=CAT.default[1];}
}
async function loadCatalog(){
  try{const r=await fetch('/models');CAT=await r.json();}catch(e){CAT=null;}
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
 <a href="/" class="m">simple view</a>
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
 const p=CAT&&CAT.providers[prov.value];
 ((p&&p.models)||[]).forEach(m=>{
  const o=document.createElement('option');o.value=m;o.textContent=m;
  mod.appendChild(o);});
 if(CAT&&CAT.default&&CAT.default[0]===prov.value&&CAT.default[1])
  mod.value=CAT.default[1];
}
async function loadCatalog(){
 try{const r=await fetch('/models');CAT=await r.json();}catch(e){CAT=null;}
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
  const r=await fetch('/task/'+cur);const j=await r.json();
  renderEvents(j.events);
  if(j.stream){
   if(!liveBub)liveBub=row('a');
   liveBub.className='bub caret';liveBub.textContent=j.stream;scroll();
  }
  state.textContent=j.status;
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
 const r=await fetch('/chat',{method:'POST',
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
 await fetch('/stop/'+cur,{method:'POST'});state.textContent='stopping...';};
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

        agent = Agent(verbose=False, on_token=on_token, on_event=on_event,
                      provider=provider, model=model)
        record["agent"] = agent
        try:
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

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path in ("/live", "/live/"):
            self._send(200, LIVE_PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/task/"):
            task_id = self.path.split("/task/", 1)[1]
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
        elif self.path == "/models":
            import llm
            try:
                payload = llm.catalog()
            except Exception as exc:  # noqa: BLE001
                payload = {"providers": {}, "default": [],
                           "error": str(exc)[:300]}
            self._send(200, json.dumps(payload).encode(),
                       "application/json")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True,
                                        "tasks": len(_TASKS)}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/stop/"):
            task_id = self.path.split("/stop/", 1)[1]
            ok = _stop_task(task_id)
            self._send(200, json.dumps({"stopped": ok}).encode(),
                       "application/json")
            return
        if self.path != "/chat":
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
    print(f"[webui] http://{config.WEBUI_HOST}:{config.WEBUI_PORT}",
          flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
