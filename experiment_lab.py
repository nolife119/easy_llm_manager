"""Windows-native inference experiment dashboard. Standard library, localhost only.

Controls already-installed binaries. It does NOT implement DS4 VQ kernels or
multi-GPU model parallelism. Never claims that configured 1M == tested 1M.
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if __name__ == "__main__":
    sys.modules.setdefault("experiment_lab", sys.modules[__name__])

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "InklingManager" / "experiment-platform"
STATE.mkdir(parents=True, exist_ok=True)
RUNS = STATE / "runs"
RUNS.mkdir(exist_ok=True)
CONFIG = STATE / "platform.json"
EXAMPLE = ROOT / "experiment_profiles.json"
LOCK = threading.RLock()
PROCESS = None
CURRENT = {}
TRIAL = None


def config():
    if not CONFIG.exists():
        shutil.copy2(EXAMPLE, CONFIG)
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def save_config(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get("profiles"), dict):
        raise ValueError("profiles object required")
    for name, profile in obj["profiles"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}", name):
            raise ValueError("invalid profile name")
        validate(profile)
    tmp = CONFIG.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CONFIG)


def validate(p):
    if p.get("backend") not in ("llama", "ds4"):
        raise ValueError("backend must be llama or ds4")
    if not isinstance(p.get("executable"), str) or not p["executable"]:
        raise ValueError("executable is required")
    if not isinstance(p.get("model"), str) or not p["model"]:
        raise ValueError("model path required")
    args = p.get("args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("args must be a list of strings")
    if not 1 <= int(p.get("port", 8000)) <= 65535:
        raise ValueError("invalid port")
    if p.get("host", "127.0.0.1") != "127.0.0.1":
        raise ValueError("bind to localhost only")
    if "API_KEY" in str(p.get("env", {})).upper():
        raise ValueError("do not store credentials in platform profiles")


def gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"error": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode:
            return {"error": out.stderr[-500:]}
        return {"columns": ["id", "name", "total_mib", "used_mib", "free_mib", "utilization_pct"],
                "rows": [x.strip().split(", ") for x in out.stdout.splitlines() if x.strip()]}
    except Exception as exc:
        return {"error": str(exc)}


def make_command(p):
    validate(p)
    exe = Path(os.path.expandvars(p["executable"])).expanduser()
    model = Path(os.path.expandvars(p["model"])).expanduser()
    if not exe.is_file():
        raise ValueError(f"Executable missing: {exe}")
    if not model.exists():
        raise ValueError(f"Model missing: {model}")
    if p["backend"] == "ds4":
        if not model.is_dir() or not (model / "release.json").is_file():
            raise ValueError("DS4 VQ requires model directory containing release.json")
    elif not model.is_file():
        raise ValueError("llama requires model GGUF file")
    substitutions = {"{model}": str(model), "{port}": str(int(p.get("port", 8000))),
                     "{host}": "127.0.0.1", "{alias}": str(p.get("alias", "local-model"))}
    cmd = [str(exe)]
    for part in p["args"]:
        for key, val in substitutions.items():
            part = part.replace(key, val)
        cmd.append(part)
    return cmd


def _readlog(proc, file):
    with file.open("a", encoding="utf-8", errors="replace") as out:
        for line in proc.stdout:
            out.write(line)
            out.flush()
    proc.wait()
    with LOCK:
        CURRENT["exit_code"] = proc.returncode
        CURRENT["state"] = "exited"


def start(name, check_only=False):
    global PROCESS
    with LOCK:
        if PROCESS and PROCESS.poll() is None:
            raise ValueError("A server is already running; stop it first")
        obj = config()
        if name not in obj["profiles"]:
            raise ValueError("Unknown profile")
        p = dict(obj["profiles"][name])
        cmd = make_command(p)
        rid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        folder = RUNS / rid
        folder.mkdir()
        meta = {"id": rid, "profile": name, "backend": p["backend"], "command": cmd,
                "started": datetime.now(timezone.utc).isoformat(), "state": "preflight" if check_only else "starting",
                "port": int(p.get("port", 8000)), "alias": p.get("alias", "local-model"),
                "context_requested": p.get("context_requested"),
                "gpu_at_start": gpu(), "check_only": check_only}
        CURRENT.clear()
        CURRENT.update(meta)
        (folder / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        # Never use a shell: profiles supply exact executable and argument vector.
        if check_only:
            if p["backend"] != "ds4":
                raise ValueError("DS4 --check-config supported in this mode only")
            cmd = cmd + ["--check-config"]
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in p.get("env", {}).items()})
        PROCESS = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   cwd=str(Path(cmd[0]).parent), env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                   bufsize=1)
        CURRENT["pid"] = PROCESS.pid
        threading.Thread(target=_readlog, args=(PROCESS, folder / "server.log"), daemon=True).start()
        return dict(CURRENT)


def stop():
    global PROCESS
    with LOCK:
        if not PROCESS or PROCESS.poll() is not None:
            return {"stopped": False}
        PROCESS.terminate()
        try:
            PROCESS.wait(timeout=8)
        except subprocess.TimeoutExpired:
            PROCESS.kill()
        CURRENT["state"] = "stopped"
        CURRENT["exit_code"] = PROCESS.returncode
        return {"stopped": True}


def api(port, path, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def probe():
    with LOCK:
        p = CURRENT.get("port")
    if not p:
        return {"ready": False, "reason": "No server"}
    try:
        return {"ready": True, "models": api(p, "/v1/models", timeout=3)}
    except Exception as exc:
        return {"ready": False, "reason": str(exc)[:300]}


def benchmark(prompt, n=256, image=None):
    with LOCK:
        meta = dict(CURRENT)
    if not meta.get("port"):
        raise ValueError("No active run")
    n = int(n)
    if not 16 <= n <= 2048:
        raise ValueError("output length must be 16..2048")
    message = {"role": "user", "content": prompt}
    if image:
        if not isinstance(image, str) or not image.startswith("data:image/"):
            raise ValueError("Only data:image/... input is supported")
        if len(image) > 8 * 1024 * 1024:
            raise ValueError("Image exceeds 8 MB encoded")
        message["content"] = [{"type": "text", "text": prompt},
                              {"type": "image_url", "image_url": {"url": image}}]
    payload = {"model": meta["alias"], "messages": [message],
               "max_tokens": n, "temperature": 0, "stream": False}
    now = time.perf_counter()
    output = api(meta["port"], "/v1/chat/completions", payload, timeout=1200)
    duration = time.perf_counter() - now
    usage = output.get("usage") or {}
    tokens = usage.get("completion_tokens")
    sample = output.get("choices", [{}])[0].get("message", {}).get("content", "")
    result = {"time": datetime.now(timezone.utc).isoformat(),
              "run_id": meta["id"], "prompt_characters": len(prompt),
              "image": bool(image), "requested_output": n,
              "completion_tokens": tokens, "elapsed_seconds": round(duration, 3),
              "end_to_end_tok_s": round(tokens / duration, 2) if isinstance(tokens, int) and duration else None,
              "gpu_after": gpu(), "sample": str(sample)[:1000],
              "warning": "End-to-end rate includes prefill, HTTP and scheduling; NOT decode-only throughput."}
    with (RUNS / meta["id"] / "benchmarks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def report():
    with LOCK:
        meta = dict(CURRENT)
    if not meta.get("id"):
        raise ValueError("No run yet")
    folder = RUNS / meta["id"]
    name = STATE / f"report-{meta['id']}.zip"
    # Never include API keys, full prompts or raw server state/config.
    manifest = {"run": {k: v for k, v in meta.items() if k != "command"},
                "command": [re.sub(r"(?i)(bearer|api.key|token)=\S+", r"\1=REDACTED", x)
                            for x in meta.get("command", [])],
                "gpu_now": gpu(), "generated": datetime.now(timezone.utc).isoformat()}
    (folder / "diagnostics.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for x in ("diagnostics.json", "server.log", "benchmarks.jsonl"):
            src = folder / x
            if src.is_file():
                # Logs may contain private prompts; review archive before sharing.
                z.write(src, arcname=x)
    return {"path": str(name), "warning": "Review server.log for confidential prompts before sharing"}


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>Inkling Lab</title>
<style>body{font-family:system-ui;background:#10141c;color:#edf2f8;margin:0 auto;max-width:1100px;padding:24px}
h1{margin-bottom:4px}small{color:#9daab9}.panel{background:#1c2430;padding:18px;border-radius:12px;margin:14px 0}
input,textarea,select,button{font:inherit;color:inherit;background:#293646;border:1px solid #506070;padding:8px;border-radius:6px}
textarea{width:98%;height:200px}button{cursor:pointer;margin:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #45505c;text-align:left;padding:6px}
strong{color:#8ed5ff}</style></head><body>
<h1>Inkling Experiment Lab</h1><small>localhost-only · Windows · native processes · DOES NOT implement VQ multi-GPU inference</small>
<div class="panel"><h3>실행 설정</h3><select id="profile"></select>
<button onclick="call('start')">시작</button><button onclick="call('check')">DS4 사전검사</button>
<button onclick="call('stop')">정지</button><button onclick="call('probe')">API 검사</button>
<button onclick="call('report')">진단 ZIP 생성</button><button onclick="runSweep()">전체 프로필 순차 시험</button><button onclick="sweepStatus()">자동 시험 결과</button>
<p><small>환경 설정은 실행 전에 JSON에서 바꿀 수 있습니다. 모델/실행 파일 경로는 실제 경로여야 합니다.</small></p>
<textarea id="config"></textarea><br><button onclick="save()">설정 저장 · 재검사</button></div>
<div class="panel"><h3>벤치마크</h3><input id="tokens" type="number" value="256" min="16" max="2048">
<input id="prompt" size="55" value="Explain an efficient PID controller architecture.">
<button onclick="runBench()">요청 · 성능 측정</button><p><small>측정값은 end-to-end입니다. decode-only 값이 아닙니다. 이미지 입력은 API를 통해 별도 테스트할 수 있습니다.</small></p></div>
<div class="panel"><h3>상태 · GPU · 최근 실행</h3><pre id="status"></pre></div>
<div class="panel"><h3>출력</h3><pre id="out"></pre></div>
<script>
async function request(url,payload){let r=await fetch(url,{method:payload?'POST':'GET',headers:{'Content-Type':'application/json'},body:payload?JSON.stringify(payload):undefined});
let j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
function show(j){document.getElementById('out').textContent=JSON.stringify(j,null,2)}
async function refresh(){try{let c=await request('/api/config');document.getElementById('config').value=JSON.stringify(c,null,2);
let p=document.getElementById('profile');let old=p.value;p.innerHTML='';
Object.keys(c.profiles).forEach(n=>p.add(new Option(n,n)));if(old&&c.profiles[old])p.value=old;
await status()}catch(e){show({error:String(e)})}}
async function status(){document.getElementById('status').textContent=JSON.stringify(await request('/api/status'),null,2)}
async function call(op){try{let n=document.getElementById('profile').value;show(await request('/api/'+op,op==='start'||op==='check'?{profile:n}:{}));await status()}catch(e){show({error:String(e)})}}
async function save(){try{show(await request('/api/config',JSON.parse(document.getElementById('config').value)));await refresh()}catch(e){show({error:String(e)})}}
async function runSweep(){try{let raw=JSON.parse(document.getElementById('config').value);let names=Object.keys(raw.profiles);if(!confirm(names.length+'개 프로필을 순차 시험합니다. 기존 서버가 실행 중이면 중지하세요. 진행할까요?'))return;show(await request('/api/sweep',{profiles:names,output:Number(document.getElementById('tokens').value),prompt:document.getElementById('prompt').value}));}catch(e){show({error:String(e)})}}
async function sweepStatus(){try{show(await request('/api/sweep-status'))}catch(e){show({error:String(e)})}}
async function runBench(){try{show(await request('/api/benchmark',{prompt:document.getElementById('prompt').value,tokens:Number(document.getElementById('tokens').value)}));await status()}catch(e){show({error:String(e)})}}
refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, mime="application/json"):
        buf = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
               if mime == "application/json" else payload.encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        self.wfile.write(buf)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        if n > 10 * 1024 * 1024:
            raise ValueError("Request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        try:
            if self.path == "/":
                return self._send(200, HTML, "text/html")
            if self.path == "/api/config":
                return self._send(200, config())
            if self.path == "/api/sweep-status":
                import experiment_sweep
                return self._send(200, experiment_sweep.status())
            if self.path == "/api/status":
                with LOCK:
                    current = dict(CURRENT)
                    if PROCESS and PROCESS.poll() is not None:
                        current["exit_code"] = PROCESS.returncode
                return self._send(200, {"run": current, "gpu": gpu(), "probe": probe()})
            self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
                raise ValueError("Cross-origin requests forbidden")
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                raise ValueError("application/json required")
            data = self._body()
            if self.path == "/api/config":
                save_config(data)
                return self._send(200, {"saved": True, "file": str(CONFIG)})
            if self.path == "/api/start":
                return self._send(200, start(data["profile"]))
            if self.path == "/api/check":
                return self._send(200, start(data["profile"], True))
            if self.path == "/api/stop":
                return self._send(200, stop())
            if self.path == "/api/probe":
                return self._send(200, probe())
            if self.path == "/api/sweep":
                import experiment_sweep
                return self._send(200, experiment_sweep.begin(data.get("profiles", []), data.get("prompt", "Explain PID control."), data.get("output", 256)))
            if self.path == "/api/benchmark":
                return self._send(200, benchmark(data.get("prompt", "Hello"),
                                                 data.get("tokens", 256), data.get("image")))
            if self.path == "/api/report":
                return self._send(200, report())
            self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    config()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Inkling Experiment Lab: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop()
        server.server_close()


if __name__ == "__main__":
    main()
