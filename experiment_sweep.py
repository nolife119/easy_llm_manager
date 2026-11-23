"""Serial profile sweep. Uses exact profile definitions; no shell expansion.

Do not confuse end-to-end throughput with decode-only speed.
"""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import experiment_lab as lab

LOCK = threading.Lock()
JOB = {"state": "idle"}


def status():
    with LOCK:
        return dict(JOB)


def worker(names, prompt, output, startup_timeout):
    results = []
    try:
        profiles = lab.config()["profiles"]
        for name in names:
            if name not in profiles:
                results.append({"profile": name, "status": "unknown profile"})
                continue
            with LOCK:
                JOB["current"] = name
            item = {"profile": name, "time": datetime.now().isoformat()}
            try:
                lab.start(name)
                end = time.monotonic() + startup_timeout
                while time.monotonic() < end:
                    ready = lab.probe()
                    if ready["ready"]:
                        break
                    with lab.LOCK:
                        proc = lab.PROCESS
                        if proc and proc.poll() is not None:
                            break
                    time.sleep(2)
                if not lab.probe()["ready"]:
                    item["status"] = "server not ready"
                    item["error"] = lab.CURRENT.get("exit_code")
                else:
                    item.update(lab.benchmark(prompt, output))
                    item["status"] = "measured"
            except Exception as exc:
                item["status"] = "failed"
                item["error"] = str(exc)[:1000]
            finally:
                try:
                    item["diagnostic"] = lab.report()
                except Exception:
                    pass
                lab.stop()
            results.append(item)
            with LOCK:
                JOB["results"] = list(results)
    finally:
        with LOCK:
            JOB["state"] = "completed"
            JOB["current"] = None


def begin(names, prompt="Provide a short explanation of PID anti-windup.", output=256, startup_timeout=180):
    with LOCK:
        if JOB.get("state") == "running":
            raise ValueError("Sweep already running")
        if lab.PROCESS and lab.PROCESS.poll() is None:
            raise ValueError("Stop the interactive server before sweep")
        if not isinstance(names, list) or not (1 <= len(names) <= 24):
            raise ValueError("Choose 1..24 profiles")
        if not all(isinstance(x, str) and x in lab.config()["profiles"] for x in names):
            raise ValueError("Unknown profile in sweep")
        if not 16 <= int(output) <= 2048:
            raise ValueError("output 16..2048")
        if not 10 <= int(startup_timeout) <= 1800:
            raise ValueError("startup_timeout 10..1800")
        JOB.clear()
        JOB.update(state="running", current=None, profiles=names, results=[])
    threading.Thread(target=worker, args=(names, prompt, int(output), int(startup_timeout)),
                     daemon=True).start()
    return status()
