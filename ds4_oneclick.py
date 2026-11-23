"""One-click *preflight and launch* for DS4 Inkling-2 VQ.

This script deliberately fails closed when the CUDA engine cannot demonstrate
support for the intended settings. It does NOT provide GPU sharding by itself.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from ds4_vq import build_command, inspect_release

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "ds4_oneclick.local.json"
EXAMPLE = ROOT / "ds4_oneclick.example.json"

def main():
    if not CONFIG.exists():
        print("CONFIG REQUIRED: copy ds4_oneclick.example.json to ds4_oneclick.local.json")
        print("Set ds4_exe and model_dir. Existing llama.cpp installation is untouched.")
        return 2
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    exe = Path(os.path.expandvars(data["ds4_exe"])).expanduser()
    model = Path(os.path.expandvars(data["model_dir"])).expanduser()
    if not exe.is_file():
        print(f"DS4 engine not found: {exe}")
        return 2
    report = inspect_release(model)
    if not report["ready_for_engine_validation"]:
        print("VQ model incomplete:", report["missing"])
        return 2
    context = int(data.get("context", 1048576))
    draft = int(data.get("mtp_draft", 3))
    cmd = build_command(
        exe, model, context=context, port=int(data.get("port", 8000)),
        mtp_draft=draft, mem_floor_gb=int(data.get("mem_floor_gb", 4)))
    print("Checking requested 1M context + DSpark using engine's own planner...")
    check = subprocess.run(cmd + ["--check-config"], check=False)
    if check.returncode:
        print("PRECHECK FAILED. No model server started.")
        print("Try mtp_draft=0, or context=131072, and inspect the planner output.")
        return check.returncode
    if not data.get("acknowledge_unverified_multi_gpu", False):
        print("STOP: Two H100 PCIe devices cannot be assumed to work as one VRAM pool.")
        print("Set acknowledge_unverified_multi_gpu=true ONLY after independently")
        print("validating DS4 device sharding on this host. This flag does not enable it.")
        return 3
    print("Preflight passed; starting server. This does NOT certify multi-GPU residency.")
    print("Verify both GPUs in nvidia-smi and measure actual decode throughput.")
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
