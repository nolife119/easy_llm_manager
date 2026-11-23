"""Localhost-only Windows control plane for Unsloth Inkling 2 Mini Q8 + DSpark.

No model is loaded by this Python process. llama-server provides native OpenAI
and Anthropic APIs; Claude Code talks to it directly, without a proxy.
"""
from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zlib
import struct
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mini_runtime as rt

PORT = 8766
ROOT = Path(__file__).resolve().parent
STATE = None


def json_request(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", "User-Agent": "InklingMiniManager/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read(8 * 1024 * 1024).decode("utf-8"))


def png_sample():
    """64x64 RGB PNG for a real image+text inference probe, no dependencies."""
    width = height = 64
    raw = b"".join(b"\x00" + b"\x34\x83\xce" * width for _ in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tail(path, limit=120):
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size - 72_000))
            return file.read().decode("utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


class Manager:
    def __init__(self):
        self.cfg = rt.load_config()
        if not self.cfg["engine"]:
            self.cfg["engine"] = rt.find_engine()
        self.proc = None
        self.lock = threading.RLock()
        self.logs = deque(maxlen=160)
        self.history = deque(maxlen=100)
        self.task = None
        self.task_status = "idle"
        self.detail = ""
        self.progress = {"done": 0, "total": 3, "file": "", "stage": ""}
        self.features = None
        self.test_results = {}
        self.metrics_prev = {}
        self.health_ok = False
        self.last_sample = 0
        self.last_gpu = {}
        self.last_metrics = {}
        self.server_log = rt.root() / "mini-server.log"

    def add_log(self, msg):
        with self.lock:
            self.logs.append(time.strftime("%H:%M:%S ") + str(msg))

    def busy(self):
        return self.task is not None and self.task.is_alive()

    def job(self, label, fn):
        with self.lock:
            if self.busy():
                raise ValueError("이미 다른 작업이 실행 중입니다")
            self.task_status = "running"
            self.detail = label
            def work():
                try:
                    result = fn()
                    with self.lock:
                        self.task_status = "done"
                        self.detail = label + " 완료"
                    self.add_log(f"{label} 완료: {result if isinstance(result, str) else 'OK'}")
                except Exception as exc:
                    with self.lock:
                        self.task_status = "error"
                        self.detail = f"{label} 오류: {exc}"
                    self.add_log(self.detail)
            self.task = threading.Thread(target=work, daemon=True)
            self.task.start()

    def save(self, changes):
        with self.lock:
            if self.busy() or (self.proc and self.proc.poll() is None):
                raise ValueError("설정은 서버와 작업을 중지한 후 변경하세요")
            self.cfg = rt.save_config(rt.validate({**self.cfg, **changes}))
            self.features = None
            self.add_log("설정 저장 완료")
            return self.cfg

    def probe(self):
        def work():
            if platform.system() != "Windows":
                raise ValueError("Windows x64 전용 플랫폼입니다")
            if platform.machine().lower() not in ("amd64", "x86_64"):
                raise ValueError("Windows x64 시스템에서 실행하세요")
            info = rt.gpu_info()
            if not info.get("ok"):
                raise ValueError(info.get("error", "GPU 점검 실패"))
            devices = info.get("devices", [])
            gpu = next((x for x in devices if x["index"] == self.cfg["gpu_index"]), None)
            if gpu is None:
                raise ValueError("GPU 인덱스를 찾을 수 없습니다")
            if "H100" not in gpu["name"].upper():
                self.add_log("주의: 대상은 H100 NVL이며 현재 GPU는 " + gpu["name"])
            features = rt.probe_engine(self.cfg["engine"])
            with self.lock:
                self.features = features
            self.add_log(f"기존 llama-server 사용: {features['path']}")
            return "GPU / DSpark 엔진 옵션 검증"
        self.job("엔진 진단", work)

    def download(self):
        def work():
            def progress(done, total, file, stage):
                with self.lock:
                    self.progress = {"done": done, "total": total, "file": file, "stage": stage}
            result = rt.download_models(self.cfg.copy(), self.add_log, progress)
            return str(result["directory"])
        self.job("Unsloth Q8 + DSpark + Vision 다운로드 및 SHA-256 검증", work)

    def base(self):
        return f"http://127.0.0.1:{self.cfg['port']}"

    def health(self):
        if not self.proc or self.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(self.base() + "/health", timeout=1) as resp:
                payload = json.loads(resp.read(2048))
            return resp.status == 200 and payload.get("status") == "ok"
        except (OSError, ValueError, urllib.error.URLError):
            return False

    def start(self):
        with self.lock:
            if self.busy():
                raise ValueError("다른 작업 중에는 시작할 수 없습니다")
            if self.proc is not None and self.proc.poll() is None:
                raise ValueError("서버가 이미 실행 중입니다")
            cfg = self.cfg.copy()
        if platform.system() != "Windows":
            raise ValueError("llama-server 실행은 Windows에서만 지원합니다")
        if not rt.is_verified(cfg):
            raise ValueError("먼저 8-bit 모델, DSpark, 비전 파일을 다운로드/검증하세요")
        gpu = rt.gpu_info()
        if not gpu.get("ok"):
            raise ValueError("nvidia-smi 진단 실패: " + str(gpu.get("error")))
        chosen = next((x for x in gpu["devices"] if x["index"] == cfg["gpu_index"]), None)
        if chosen is None:
            raise ValueError("선택한 GPU 인덱스가 없습니다")
        caps = rt.probe_engine(cfg["engine"])
        cmd = rt.build_command(cfg, caps)
        rt.root().mkdir(parents=True, exist_ok=True)
        self.server_log.write_text("Inkling Mini llama-server started " + time.ctime() +
                                   "\nArguments: " + subprocess.list2cmdline(cmd) + "\n",
                                   encoding="utf-8")
        env = os.environ.copy()
        # Logical CUDA0 is the single selected physical GPU. Never allocate another GPU.
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_index"])
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with self.server_log.open("ab") as output:
            proc = subprocess.Popen(cmd, cwd=str(Path(cfg["engine"]).parent),
                                    env=env, stdout=output, stderr=subprocess.STDOUT,
                                    creationflags=flags)
        with self.lock:
            self.proc = proc
            self.features = caps
            self.metrics_prev = {}
            self.history.clear()
            self.health_ok = False
            self.test_results = {}
            self.task_status = "running"
            self.detail = "GPU에 모델 적재 중"
        self.add_log(f"llama-server PID={proc.pid}; CUDA_VISIBLE_DEVICES={cfg['gpu_index']}")
        def readiness():
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    self.add_log("llama-server가 종료되었습니다. 서버 로그를 확인하세요.")
                    with self.lock:
                        self.task_status = "error"
                        self.detail = f"llama-server 종료 코드 {proc.returncode}"
                    return
                if self.health():
                    with self.lock:
                        self.task_status = "done"
                        self.detail = "서버 정상 응답. 검증 실행을 눌러 출력/도구/이미지를 점검하세요."
                        self.health_ok = True
                    self.add_log("Health OK: " + self.base())
                    return
                time.sleep(3)
            with self.lock:
                self.task_status = "error"
                self.detail = "모델 로드가 10분을 초과했습니다. 로그를 확인하세요."
        threading.Thread(target=readiness, daemon=True).start()

    def stop(self):
        with self.lock:
            proc = self.proc
            self.proc = None
            self.health_ok = False
            self.task_status = "idle"
            self.detail = "서버 종료"
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=15)
            else:
                proc.terminate()
        self.add_log("llama-server 종료 요청")

    def test(self):
        if not self.health():
            raise ValueError("Health OK 이후 검증할 수 있습니다")
        cfg = self.cfg.copy()
        def work():
            base = self.base()
            tests = {}
            def request(payload):
                return json_request(base + "/v1/chat/completions", payload, timeout=180)
            def check_text():
                out = request({"model": cfg.get("alias", "inkling-mini"),
                               "messages": [{"role": "user", "content": "Reply with one word: OK"}],
                               "max_tokens": 64, "stream": False})
                assert out["choices"][0]["message"].get("content"), "empty completion"
                return {"ok": True, "result": out["choices"][0]["message"]["content"][:240],
                        "usage": out.get("usage")}
            def check_anthropic():
                return json_request(base + "/v1/messages/count_tokens",
                    {"model": "inkling-mini",
                     "messages": [{"role": "user", "content": "Count tokens"}]}, timeout=30)
            def check_tool():
                out = request({"model": "inkling-mini",
                    "messages": [{"role": "user",
                                  "content": "Call the lookup function with word=hello. Do not reply in text."}],
                    "tools": [{"type": "function", "function": {"name": "lookup",
                               "description": "Return a dictionary word",
                               "parameters": {"type": "object", "properties": {
                                   "word": {"type": "string"}}, "required": ["word"]}}}],
                    "tool_choice": "required", "max_tokens": 256})
                calls = out["choices"][0]["message"].get("tool_calls") or []
                return {"ok": bool(calls), "tool_calls": calls[:1],
                        "warning": None if calls else "도구 호출 미생성: 파서/모델 설정 확인"}
            def check_image():
                uri = "data:image/png;base64," + base64.b64encode(png_sample()).decode("ascii")
                out = request({"model": "inkling-mini", "messages": [{"role": "user",
                    "content": [{"type": "text", "text": "What is the main color? Answer in one word."},
                                {"type": "image_url", "image_url": {"url": uri}}]}],
                    "max_tokens": 64})
                result = out["choices"][0]["message"].get("content", "")
                return {"ok": bool(result), "result": str(result)[:240]}
            for name, fn in (("text", check_text), ("anthropic_count_tokens", check_anthropic),
                             ("tool_call", check_tool),
                             ("image", check_image if cfg["vision"] else lambda: {"skipped": True})):
                try:
                    result = fn()
                    tests[name] = result
                    self.add_log(f"검증 {name}: {'OK' if result.get('ok', True) else '주의'}")
                except Exception as exc:
                    tests[name] = {"ok": False, "error": str(exc)}
                    self.add_log(f"검증 {name}: 오류 {exc}")
                with self.lock:
                    self.test_results = tests.copy()
            return "완료 (각 검증 결과를 확인하세요)"
        self.job("텍스트 / Anthropic / Tool / 이미지 실동작 검증", work)

    def status(self):
        with self.lock:
            cfg = self.cfg.copy()
            proc = self.proc
            task = {"state": self.task_status, "detail": self.detail, **self.progress}
            logs = list(self.logs)[-40:]
            tests = self.test_results.copy()
            features = self.features.copy() if self.features else None
        running = proc is not None and proc.poll() is None
        health = self.health() if running else False
        now = time.time()
        if now - self.last_sample > 1.5:
            self.last_gpu = rt.gpu_info()
            self.last_sample = now
            if running and health and (features or {}).get("metrics", False):
                try:
                    with urllib.request.urlopen(self.base() + "/metrics", timeout=2) as resp:
                        metrics = rt.parse_metrics(resp.read(512000).decode("utf-8", errors="replace"))
                    prompt = rt.rate(metrics, "llamacpp:prompt_tokens_total",
                                     "llamacpp:prompt_seconds_total", self.metrics_prev)
                    decode = rt.rate(metrics, "llamacpp:tokens_predicted_total",
                                     "llamacpp:tokens_predicted_seconds_total", self.metrics_prev)
                    drafted = metrics.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
                    accepted = metrics.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
                    before_drafted = self.metrics_prev.get("llamacpp:spec_decode_num_draft_tokens_total", 0)
                    before_accepted = self.metrics_prev.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)
                    d_drafted = max(0, drafted - before_drafted)
                    d_accepted = max(0, accepted - before_accepted)
                    acceptance = round(100 * d_accepted / d_drafted, 1) if d_drafted > 0 else None
                    self.last_metrics = {"prefill_tps": prompt, "decode_tps": decode,
                                         "dspark_acceptance": acceptance,
                                         "drafted": d_drafted, "accepted": d_accepted,
                                         "tokens": metrics.get("llamacpp:tokens_predicted_total"),
                                         "prompt_tokens": metrics.get("llamacpp:prompt_tokens_total")}
                    self.metrics_prev = metrics
                except (OSError, ValueError):
                    self.last_metrics = {"error": "metrics unavailable"}
            if running:
                with self.lock:
                    gpu_used = None
                    if self.last_gpu.get("ok"):
                        chosen = next((x for x in self.last_gpu.get("devices", [])
                                      if x["index"] == cfg["gpu_index"]), None)
                        if chosen:
                            gpu_used = round(chosen["total_mb"] - chosen["free_mb"])
                    self.history.append({"t": round(now), "gpu_mb": gpu_used,
                                         **self.last_metrics})
        with self.lock:
            return {
                "config": cfg, "running": running, "health": health,
                "pid": proc.pid if running else None, "verified": rt.is_verified(cfg),
                "gpu": self.last_gpu, "features": features, "task": task, "tests": tests,
                "metrics": self.last_metrics, "history": list(self.history),
                "logs": logs, "server_log": tail(self.server_log, 35),
                "url": self.base(), "manager": f"http://127.0.0.1:{PORT}",
            }


class Handler(BaseHTTPRequestHandler):
    server_version = "InklingMini/1.0"

    def log_message(self, fmt, *args):
        return

    def authorized(self):
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            self.fail(403, "localhost Host header required")
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            self.fail(403, "cross-origin requests denied")
            return False
        return True

    def output(self, status, body, kind="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(data)

    def fail(self, status, message):
        self.output(status, {"error": str(message)})

    def do_GET(self):
        if not self.authorized():
            return
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            self.output(200, (ROOT / "mini_ui.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.output(200, STATE.status())
        else:
            self.fail(404, "not found")

    def do_POST(self):
        if not self.authorized():
            return
        if self.headers.get("Content-Type", "").split(";")[0].lower() != "application/json":
            return self.fail(415, "Content-Type application/json required")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n < 0 or n > 100_000:
                raise ValueError("요청 크기가 잘못되었습니다")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("JSON object 필수")
            path = urllib.parse.urlsplit(self.path).path
            if path == "/api/save":
                result = STATE.save(body)
            elif path == "/api/probe":
                STATE.probe()
                result = {"started": True}
            elif path == "/api/download":
                STATE.download()
                result = {"started": True}
            elif path == "/api/start":
                STATE.start()
                result = {"started": True}
            elif path == "/api/stop":
                STATE.stop()
                result = {"stopped": True}
            elif path == "/api/test":
                STATE.test()
                result = {"started": True}
            else:
                return self.fail(404, "not found")
            self.output(200, result)
        except (ValueError, OSError, RuntimeError) as exc:
            self.fail(400, str(exc))
        except Exception as exc:
            STATE.add_log(traceback.format_exc(limit=4))
            self.fail(500, str(exc))


def main():
    global STATE
    if platform.system() != "Windows":
        print("NOTE: UI may run for tests elsewhere; GPU execution is Windows only.")
    STATE = Manager()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Inkling 2 Mini Manager: {url}")
    webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        STATE.stop()
        server.server_close()


if __name__ == "__main__":
    main()
