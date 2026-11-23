"""Experimental DS4 Inkling-2 VQ launcher, isolated from llama.cpp.

This module does not implement GPU tensor parallelism. DS4's ability to use
two PCIe H100 NVL devices must be validated separately. It will never assume
that CUDA_VISIBLE_DEVICES=0,1 implies model sharding.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


PARTS = 52
EXPERTS = 48
EXPECTED_ARCH = "inkling2_vq"


def inspect_release(model_dir: str | Path) -> dict:
    root = Path(model_dir).expanduser().resolve()
    manifest = root / "release.json"
    if not manifest.is_file():
        raise ValueError("Missing release.json: DS4 VQ is not a llama.cpp GGUF shard")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid release.json") from exc
    if not isinstance(data, dict):
        raise ValueError("release.json must contain an object")
    experts = sorted(root.glob("expert-*.gguf"))
    # The exact release file naming may differ; preserve the manifest as the authority.
    ggufs = sorted(root.glob("*.gguf"))
    known = {
        "body-nonexpert.gguf",
        "embed-tokens-bf16.gguf",
        "mtp-q8_0.gguf",
        "vision-bf16.gguf",
    }
    missing = sorted(name for name in known if not (root / name).is_file())
    tokenizer = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "processor_config.json",
    ]
    missing += [name for name in tokenizer if not (root / name).is_file()]
    if len(ggufs) != PARTS:
        missing.append(f"GGUF parts: found {len(ggufs)}, expected {PARTS}")
    return {
        "root": str(root), "manifest": str(manifest),
        "part_count": len(ggufs), "expert_candidates": len(experts),
        "manifest_keys": sorted(data), "missing": missing,
        "ready_for_engine_validation": not missing,
        "note": "This checks local presence only. DS4 performs authoritative SHA-256 and layout validation.",
    }


def build_command(exe: str | Path, model_dir: str | Path, *,
                  context: int = 32768, port: int = 8000,
                  host: str = "127.0.0.1", model_id: str = "inkling-2-vq",
                  mtp_draft: int = 0, mem_floor_gb: int = 4) -> list[str]:
    path = Path(exe).expanduser().resolve()
    if not path.is_file():
        raise ValueError("DS4 executable not found")
    report = inspect_release(model_dir)
    if not report["ready_for_engine_validation"]:
        raise ValueError("Incomplete VQ release: " + ", ".join(report["missing"]))
    if not 1 <= context <= 1048576:
        raise ValueError("context must be 1..1048576")
    if not 1 <= port <= 65535:
        raise ValueError("invalid port")
    if host != "127.0.0.1":
        raise ValueError("Localhost only until DS4 authentication is verified")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", model_id):
        raise ValueError("unsafe model id")
    if mtp_draft not in (0, 2, 3, 4, 5, 6, 7):
        raise ValueError("mtp_draft must be 0 or 2..7")
    cmd = [str(path), "--cuda", "-m", report["root"], "-c", str(context),
           "--host", host, "--port", str(port), "--model-id", model_id,
           "--mem-floor-gb", str(mem_floor_gb), "--no-update-check"]
    if mtp_draft:
        cmd += ["--mtp-mode", "on", "--mtp-draft", str(mtp_draft)]
    else:
        cmd += ["--mtp-mode", "off"]
    return cmd


def health(port: int = 8000, model_id: str = "inkling-2-vq", timeout: float = 3) -> dict:
    """Read-only OpenAI discovery; does not assume llama.cpp /health semantics."""
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url), timeout=timeout) as response:
        result = json.load(response)
    ids = [m.get("id") for m in result.get("data", []) if isinstance(m, dict)]
    return {"reachable": True, "models": ids, "model_found": model_id in ids}


def benchmark(port: int = 8000, model_id: str = "inkling-2-vq",
              prompt: str = "Explain the role of a PID controller.", output: int = 128,
              timeout: float = 300) -> dict:
    """Wall-clock end-to-end speed; NOT a kernel-only decode rate."""
    if not 1 <= output <= 4096:
        raise ValueError("output out of range")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.perf_counter()
    with opener.open(req, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    usage = result.get("usage", {})
    tokens = usage.get("completion_tokens")
    return {
        "elapsed_seconds": round(elapsed, 3),
        "completion_tokens": tokens,
        "end_to_end_tokens_per_second": round(tokens / elapsed, 3) if isinstance(tokens, int) and elapsed > 0 else None,
        "note": "Includes prefill and HTTP overhead. Compare DS4 native decode metrics for >=30 tok/s goal.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inkling 2 VQ / DS4 experimental adapter")
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check")
    check.add_argument("--model-dir", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--model-dir", required=True)
    launch.add_argument("--exe", required=True)
    launch.add_argument("--context", type=int, default=32768)
    launch.add_argument("--port", type=int, default=8000)
    launch.add_argument("--mtp-draft", type=int, default=0)
    launch.add_argument("--mem-floor-gb", type=int, default=4)
    probe = sub.add_parser("health")
    probe.add_argument("--port", type=int, default=8000)
    probe.add_argument("--model-id", default="inkling-2-vq")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--port", type=int, default=8000)
    bench.add_argument("--model-id", default="inkling-2-vq")
    bench.add_argument("--output", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            answer = inspect_release(args.model_dir)
            print(json.dumps(answer, ensure_ascii=False, indent=2))
            return 0 if answer["ready_for_engine_validation"] else 2
        if args.action == "run":
            command = build_command(args.exe, args.model_dir, context=args.context,
                                    port=args.port, mtp_draft=args.mtp_draft,
                                    mem_floor_gb=args.mem_floor_gb)
            print("DS4 launch:", subprocess.list2cmdline(command), flush=True)
            print("WARNING: 2-GPU H100 execution is NOT validated by this adapter.", flush=True)
            return subprocess.call(command, env=os.environ.copy())
        if args.action == "health":
            result = health(args.port, args.model_id)
        else:
            result = benchmark(args.port, args.model_id, output=args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, urllib.error.URLError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
